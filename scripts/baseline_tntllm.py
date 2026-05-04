#!/usr/bin/env python3
"""
TnT-LLM-style baseline (Wan et al., KDD 2024)
===============================================
Same interface as concept_knn_v5.py:
  --features-file features_v4.json
  --gt-dir ground_truth/

Pipeline:
  Phase 1: Taxonomy Generation
    1a. Sample N prompts, summarize with LLM
    1b. Iteratively generate & refine taxonomy from summaries
    1c. Hierarchical subdivision — subdivide each category into subcategories
  Phase 2: Classification at Scale
    2a. LLM labels a training subset using taxonomy
    2b. Train LogReg classifier on SBERT embeddings
    2c. Classify full dataset → cluster labels
  Evaluate with same evaluate_gt()

Fixes over v1:
  - Guard against taxonomy collapse (reject refinement if categories drop >50%)
  - Hierarchical subdivision to reach 200-500+ leaf categories
  - Larger label sample for more classes
  - Better prompts with explicit category count targets

Usage:
  python baseline_tntllm.py \\
    --features-file features_v4.json \\
    --gt-dir ground_truth/ \\
    --output-dir results/tntllm \\
    --api-key YOUR_OPENAI_KEY

Requirements:
  pip install openai sentence-transformers scikit-learn numpy tqdm
"""

import json, os, sys, time, random, argparse, re
import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from eval_shared import load_features, evaluate_gt, compute_metrics, report


# ═══════════════════════════════════════════════════════════════
# PHASE 1: TAXONOMY GENERATION
# ═══════════════════════════════════════════════════════════════

def summarize_batch(prompts_batch, client, model):
    """Summarize prompts into short intent descriptions."""
    summaries = []
    for prompt in prompts_batch:
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=50, temperature=0,
                messages=[
                    {"role": "system",
                     "content": "Summarize the user's intent in one sentence (max 20 words). "
                                "Focus on topic and task."},
                    {"role": "user", "content": prompt[:1000]}])
            summaries.append(resp.choices[0].message.content.strip())
        except Exception as e:
            summaries.append(prompt[:100])
    return summaries


def generate_taxonomy(summaries_batch, client, model):
    """Generate initial taxonomy from summaries."""
    text = "\n".join(f"- {s}" for s in summaries_batch[:100])
    resp = client.chat.completions.create(
        model=model, max_tokens=4000, temperature=0.3,
        messages=[
            {"role": "system",
             "content": """You are a taxonomy generation expert. Given conversation summaries,
generate a FLAT taxonomy of categories for classifying these conversations.

Output a JSON array of objects:
[{"name": "Category Name", "description": "1-sentence description"}, ...]

IMPORTANT:
- Generate exactly 30-50 categories
- Categories should be SPECIFIC (e.g., "Python Programming", not just "Programming")
- Categories must be mutually exclusive
- Include an "Other" category for edge cases
- Output ONLY the JSON array, no other text"""},
            {"role": "user", "content": f"Conversation summaries:\n{text}"}])
    return _parse_taxonomy(resp.choices[0].message.content.strip())


def refine_taxonomy(taxonomy, new_summaries, client, model):
    """Refine taxonomy with new batch. Returns None if refinement degrades."""
    tax_text = json.dumps([{"name": t.get("name",""), "description": t.get("description","")}
                           for t in taxonomy[:80]], indent=1)
    text = "\n".join(f"- {s}" for s in new_summaries[:100])
    resp = client.chat.completions.create(
        model=model, max_tokens=4000, temperature=0.2,
        messages=[
            {"role": "system",
             "content": f"""You are a taxonomy refinement expert. Given an existing taxonomy
(currently {len(taxonomy)} categories) and new conversation summaries:

1. ADD categories for uncovered topics in the new summaries
2. SPLIT overly broad categories if evidence warrants it
3. MERGE only near-duplicate categories (e.g., "Python Coding" + "Python Programming")
4. Keep total between 30-60 categories

CRITICAL: Do NOT collapse categories aggressively. The goal is COVERAGE, not minimalism.
Output the refined taxonomy as a JSON array:
[{{"name": "...", "description": "..."}}, ...]
Output ONLY the JSON array."""},
            {"role": "user",
             "content": f"Current taxonomy ({len(taxonomy)} categories):\n{tax_text}\n\n"
                        f"New conversation summaries:\n{text}"}])
    return _parse_taxonomy(resp.choices[0].message.content.strip())


def subdivide_taxonomy(taxonomy, summaries, client, model, target_leaves=300):
    """
    Hierarchical subdivision: for each top-level category, generate subcategories.
    This is faithful to TnT-LLM's hierarchical approach (Section 3.2 of the paper).
    """
    print(f"\n  Hierarchical subdivision (target ~{target_leaves} leaves)...")

    # Classify summaries into top-level categories first
    cat_names = [t.get("name", "?") for t in taxonomy]
    cat_list = "\n".join(f"- {name}" for name in cat_names)

    # Batch-classify summaries
    cat_to_summaries = defaultdict(list)
    for i in range(0, len(summaries), 50):
        batch = summaries[i:i+50]
        batch_text = "\n".join(f"{j+1}. {s}" for j, s in enumerate(batch))
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=2000, temperature=0,
                messages=[
                    {"role": "system",
                     "content": f"Classify each numbered summary into exactly one category.\n\n"
                                f"Categories:\n{cat_list}\n\n"
                                f"Output JSON: {{\"1\": \"category\", \"2\": \"category\", ...}}\n"
                                f"Output ONLY the JSON."},
                    {"role": "user", "content": batch_text}])
            text = resp.choices[0].message.content.strip()
            try:
                m = re.search(r'\{.*\}', text, re.DOTALL)
                assignments = json.loads(m.group()) if m else json.loads(text)
                for k, v in assignments.items():
                    idx = int(k) - 1
                    if 0 <= idx < len(batch):
                        cat_to_summaries[v].append(batch[idx])
            except:
                pass
        except:
            pass

    # Compute subcategories per top-level category
    n_top = len(taxonomy)
    subs_per_cat = max(3, target_leaves // max(n_top, 1))

    all_leaves = []
    for cat in taxonomy:
        name = cat.get("name", "?")
        samps = cat_to_summaries.get(name, [])

        if len(samps) < 5:
            # Too few samples — keep as leaf
            all_leaves.append({"name": name, "description": cat.get("description", ""),
                               "parent": name})
            continue

        # Ask LLM to subdivide
        samp_text = "\n".join(f"- {s}" for s in samps[:30])
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=2000, temperature=0.3,
                messages=[
                    {"role": "system",
                     "content": f"""Subdivide the category "{name}" into {subs_per_cat}-{subs_per_cat+5} specific subcategories.

Based on these example conversations in this category:
{samp_text}

Output JSON array:
[{{"name": "{name}: Subcategory", "description": "brief description"}}, ...]

Rules:
- Prefix each subcategory with "{name}: "
- Be specific (e.g., "{name}: Data Visualization" not just "{name}: General")
- Include "{name}: Other" for edge cases
- Output ONLY the JSON array."""},
                    {"role": "user", "content": f"Subdivide '{name}' into subcategories."}])
            subs = _parse_taxonomy(resp.choices[0].message.content.strip())
            if subs and len(subs) >= 2:
                for s in subs:
                    s["parent"] = name
                all_leaves.extend(subs)
                print(f"    {name}: {len(subs)} subcategories")
            else:
                all_leaves.append({"name": name, "description": cat.get("description",""),
                                   "parent": name})
        except Exception as e:
            all_leaves.append({"name": name, "description": cat.get("description",""),
                               "parent": name})
            print(f"    {name}: subdivision failed ({e})")

    print(f"  Subdivision complete: {n_top} top-level -> {len(all_leaves)} leaf categories")
    return all_leaves


def _parse_taxonomy(text):
    try:
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m: return json.loads(m.group())
        return json.loads(text)
    except json.JSONDecodeError:
        # Try fixing common issues
        try:
            cleaned = re.sub(r',\s*\]', ']', text)
            m = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if m: return json.loads(m.group())
        except:
            pass
        # Line-based fallback
        entries = []
        for line in text.split("\n"):
            line = line.strip().strip("-*").strip()
            if line and len(line) > 3:
                entries.append({"name": line[:100], "description": "", "parent_id": None})
        return entries if entries else []


# ═══════════════════════════════════════════════════════════════
# PHASE 2: CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

def build_taxonomy_prompt(taxonomy, max_cats=300):
    """Build compact label list for classification prompt."""
    cats = taxonomy[:max_cats]
    lines = [f"- {t.get('name','?')}" for t in cats]
    return "\n".join(lines), [t.get("name", "?") for t in cats]


def label_with_llm(prompts_batch, taxonomy, client, model):
    """LLM-label prompts using taxonomy."""
    tax_prompt, cat_names = build_taxonomy_prompt(taxonomy)
    cat_names_lower = {c.lower(): c for c in cat_names}
    labels = []
    for prompt in prompts_batch:
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=80, temperature=0,
                messages=[
                    {"role": "system",
                     "content": f"Classify into exactly one category. Reply with ONLY the category name.\n\n"
                                f"Categories:\n{tax_prompt}"},
                    {"role": "user", "content": prompt[:500]}])
            label = resp.choices[0].message.content.strip().strip('"\'')
            # Exact match
            if label in cat_names:
                labels.append(label)
                continue
            # Case-insensitive
            ll = label.lower()
            if ll in cat_names_lower:
                labels.append(cat_names_lower[ll])
                continue
            # Substring match
            matched = next((c for c in cat_names if c.lower() in ll or ll in c.lower()), None)
            labels.append(matched or label)
        except Exception as e:
            labels.append("Other")
            if "rate" in str(e).lower(): time.sleep(5)
    return labels


def train_classifier(embeddings_train, labels_train, embeddings_all):
    """Train LogReg on LLM-labeled subset, predict all."""
    le = LabelEncoder()
    y = le.fit_transform(labels_train)
    n_classes = len(le.classes_)
    print(f"  Training LogReg: {len(y)} samples, {n_classes} classes...")

    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', n_jobs=-1)
    clf.fit(embeddings_train, y)
    print(f"  Train acc: {clf.score(embeddings_train, y):.4f}")
    y_pred = clf.predict(embeddings_all)
    return y_pred


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='results/tntllm')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--embed-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--taxonomy-sample', type=int, default=1000)
    parser.add_argument('--label-sample', type=int, default=5000,
                        help="More labels needed for fine-grained taxonomy")
    parser.add_argument('--n-refine', type=int, default=4)
    parser.add_argument('--target-leaves', type=int, default=300,
                        help="Target leaf categories after subdivision")
    parser.add_argument('--skip-subdivide', action='store_true',
                        help="Skip hierarchical subdivision (coarse only)")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key: raise ValueError("Set --api-key or OPENAI_API_KEY")
    t0 = time.time()

    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    data, prompts = load_features(args.features_file)
    n = len(prompts)

    # ── Phase 1: Taxonomy ──
    print(f"\n{'='*70}")
    print(f"PHASE 1: Taxonomy Generation")
    print(f"{'='*70}")

    tax_idx = random.sample(range(n), min(args.taxonomy_sample, n))
    tax_prompts = [prompts[i] for i in tax_idx]

    print(f"  Summarizing {len(tax_prompts)} prompts...")
    summaries = []
    for i in range(0, len(tax_prompts), 50):
        summaries.extend(summarize_batch(tax_prompts[i:i+50], client, args.model))
        if (i // 50 + 1) % 4 == 0 or i + 50 >= len(tax_prompts):
            print(f"    {min(i+50, len(tax_prompts))}/{len(tax_prompts)}")

    with open(os.path.join(args.output_dir, "summaries.json"), "w") as f:
        json.dump(summaries, f)

    # Generate + refine taxonomy (with collapse guard)
    batches = [summaries[i:i+200] for i in range(0, len(summaries), 200)]
    print(f"  Generating taxonomy from {len(batches)} batches...")
    taxonomy = generate_taxonomy(batches[0], client, args.model)
    print(f"    Initial: {len(taxonomy)} categories")

    for i, batch in enumerate(batches[1:args.n_refine], 1):
        refined = refine_taxonomy(taxonomy, batch, client, args.model)
        if refined and len(refined) >= len(taxonomy) * 0.5:
            taxonomy = refined
            print(f"    Refinement {i}: {len(taxonomy)} categories (accepted)")
        else:
            n_ref = len(refined) if refined else 0
            print(f"    Refinement {i}: {n_ref} categories — REJECTED (collapse guard, "
                  f"keeping {len(taxonomy)})")

    # Hierarchical subdivision
    if not args.skip_subdivide:
        leaf_taxonomy = subdivide_taxonomy(
            taxonomy, summaries, client, args.model,
            target_leaves=args.target_leaves)
    else:
        leaf_taxonomy = taxonomy
        print(f"  Skipping subdivision (--skip-subdivide)")

    with open(os.path.join(args.output_dir, "taxonomy.json"), "w") as f:
        json.dump(leaf_taxonomy, f, indent=2)
    print(f"\n  Final taxonomy: {len(leaf_taxonomy)} leaf categories "
          f"(from {len(taxonomy)} top-level)")

    # ── Phase 2: Classification ──
    print(f"\n{'='*70}")
    print(f"PHASE 2: Classification")
    print(f"{'='*70}")

    from sentence_transformers import SentenceTransformer
    print(f"  SBERT encode ({args.embed_model})...")
    embed_model = SentenceTransformer(args.embed_model)
    all_emb = embed_model.encode(prompts, batch_size=256,
                                  show_progress_bar=True, normalize_embeddings=True)

    label_idx = random.sample(range(n), min(args.label_sample, n))
    label_prompts = [prompts[i] for i in label_idx]

    print(f"  LLM-labeling {len(label_prompts)} prompts "
          f"into {len(leaf_taxonomy)} categories...")
    llm_labels = []
    for i in range(0, len(label_prompts), 20):
        llm_labels.extend(label_with_llm(label_prompts[i:i+20], leaf_taxonomy,
                                          client, args.model))
        if (i // 20) % 25 == 0:
            print(f"    {min(i+20, len(label_prompts))}/{len(label_prompts)} "
                  f"({len(set(llm_labels))} unique labels so far)")

    with open(os.path.join(args.output_dir, "llm_labels.json"), "w") as f:
        json.dump({"indices": label_idx, "labels": llm_labels}, f)

    n_unique = len(set(llm_labels))
    print(f"  LLM assigned {n_unique} unique labels from {len(leaf_taxonomy)} categories")

    # Train classifier
    pred_labels = train_classifier(all_emb[label_idx], llm_labels, all_emb)
    met = compute_metrics(pred_labels, n)
    gt = evaluate_gt(pred_labels, args.gt_dir)
    bal = report("TnT-LLM", met, gt)

    # Also report coarse (top-level only) result
    if not args.skip_subdivide:
        print(f"\n── Also trying coarse (top-level only) ──")
        coarse_labels = []
        for lbl in llm_labels:
            if ": " in lbl:
                coarse_labels.append(lbl.split(": ")[0])
            else:
                coarse_labels.append(lbl)
        pred_coarse = train_classifier(all_emb[label_idx], coarse_labels, all_emb)
        met_c = compute_metrics(pred_coarse, n)
        gt_c = evaluate_gt(pred_coarse, args.gt_dir)
        bal_c = report("TnT-LLM (coarse)", met_c, gt_c)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  TnT-LLM: bal={bal:.4f}")
    print(f"  Taxonomy: {len(taxonomy)} top-level -> {len(leaf_taxonomy)} leaves")
    print(f"  Classifier: {n_unique} classes from {len(llm_labels)} LLM labels")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*70}")

    results = {"method": "TnT-LLM", "model": args.model, "bal": bal,
               "n_top_categories": len(taxonomy),
               "n_leaf_categories": len(leaf_taxonomy),
               "n_classifier_classes": n_unique,
               "label_sample": len(llm_labels),
               **met, **gt, "elapsed_s": round(elapsed)}
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()