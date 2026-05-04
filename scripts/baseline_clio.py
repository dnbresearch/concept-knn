#!/usr/bin/env python3
"""
Clio-style baseline (Tamkin et al., 2024)
=========================================
Same interface as concept_knn_v5.py:
  --features-file features_v4.json
  --gt-dir ground_truth/

Pipeline:
  1. LLM extracts structured facets (topic, task, domain) per conversation
  2. Embed facet text with SBERT
  3. KMeans clustering on facet embeddings
  Evaluate with same evaluate_gt()

Usage:
  python baseline_clio.py \\
    --features-file features_v4.json \\
    --gt-dir ground_truth/ \\
    --output-dir results/clio \\
    --api-key YOUR_OPENAI_KEY \\
    --facet-sample 5000

  --facet-sample 0  → process ALL items (expensive!)
  --facet-sample 5000 → sample 5K, fallback to raw prompt for rest

Requirements:
  pip install openai sentence-transformers scikit-learn numpy tqdm
"""

import json, os, sys, time, random, argparse, re
import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

from eval_shared import load_features, evaluate_gt, compute_metrics, report


# ═══════════════════════════════════════════════════════════════
# FACET EXTRACTION (core Clio innovation)
# ═══════════════════════════════════════════════════════════════

FACET_PROMPT = """Analyze this conversation prompt and extract structured facets.

Conversation: {text}

Extract as JSON:
{{"topic": "main subject (e.g. machine learning, cooking, legal advice)",
  "task": "what user wants (e.g. code debugging, creative writing, info seeking)",
  "domain": "broad domain (e.g. technology, education, entertainment)",
  "combined": "concise 5-10 word description of topic+task"}}

Reply ONLY with the JSON object."""


def extract_facets(prompts, indices, client, model, output_dir, resume=False):
    """Extract Clio-style facets for given indices."""
    facets_file = os.path.join(output_dir, "facets.json")
    existing = {}
    if resume and os.path.exists(facets_file):
        with open(facets_file) as f:
            existing = {int(k): v for k, v in json.load(f).items()}
        print(f"  Resuming: {len(existing)} existing facets")

    remaining = [i for i in indices if i not in existing]
    print(f"  Extracting facets: {len(remaining)} to process "
          f"({len(existing)} cached)")

    all_facets = dict(existing)

    for batch_start in tqdm(range(0, len(remaining), 20), desc="Facets"):
        batch_idx = remaining[batch_start:batch_start+20]
        for idx in batch_idx:
            text = prompts[idx][:800]
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=200, temperature=0,
                    messages=[{"role": "user",
                               "content": FACET_PROMPT.format(text=text)}])
                content = resp.choices[0].message.content.strip()
                try:
                    m = re.search(r'\{.*\}', content, re.DOTALL)
                    facets = json.loads(m.group()) if m else json.loads(content)
                except json.JSONDecodeError:
                    facets = {"combined": content[:100], "topic": content[:50],
                              "task": "unknown", "domain": "unknown"}
                all_facets[idx] = facets
            except Exception as e:
                all_facets[idx] = {"combined": text[:50], "topic": "error",
                                   "task": "error", "domain": "error"}
                if "rate" in str(e).lower(): time.sleep(5)

        # Checkpoint every 500
        if (batch_start // 20) % 25 == 0 and batch_start > 0:
            with open(facets_file, "w") as f:
                json.dump({str(k): v for k, v in all_facets.items()}, f)

    with open(facets_file, "w") as f:
        json.dump({str(k): v for k, v in all_facets.items()}, f)
    print(f"  Extracted {len(all_facets)} facets total")
    return all_facets


# ═══════════════════════════════════════════════════════════════
# EMBED + CLUSTER
# ═══════════════════════════════════════════════════════════════

def embed_facets(facets, prompts, embed_model_name):
    """Embed facet text; fallback to raw prompt for items without facets."""
    from sentence_transformers import SentenceTransformer
    print(f"  Embedding facets ({embed_model_name})...")
    model = SentenceTransformer(embed_model_name)

    texts = []
    for i in range(len(prompts)):
        if i in facets and "combined" in facets[i]:
            f = facets[i]
            texts.append(f"{f['combined']}. Topic: {f.get('topic','')}. "
                         f"Task: {f.get('task','')}")
        else:
            texts.append(prompts[i][:200])  # Fallback

    emb = model.encode(texts, batch_size=256, show_progress_bar=True,
                       normalize_embeddings=True)
    print(f"  Facet embeddings: {emb.shape}")
    return emb


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='results/clio')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--embed-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--k', nargs='+', type=int, default=[5000, 10000, 15000])
    parser.add_argument('--facet-sample', type=int, default=0,
                        help="0=all items (expensive), else sample N")
    parser.add_argument('--resume', action='store_true')
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

    # Determine which indices to extract facets for
    if args.facet_sample > 0 and args.facet_sample < n:
        facet_indices = random.sample(range(n), args.facet_sample)
        print(f"  Sampling {args.facet_sample}/{n} items for facet extraction")
    else:
        facet_indices = list(range(n))
        print(f"  Processing ALL {n} items (this will be expensive!)")

    # Estimate cost
    est_tokens = len(facet_indices) * 300
    est_cost = est_tokens * 0.0003 / 1000
    print(f"  Est. cost: ~{len(facet_indices)} items × 300 tok = "
          f"~{est_tokens/1e6:.1f}M tokens ≈ ${est_cost:.2f}")

    # Step 1: Extract facets
    print(f"\n{'='*70}")
    print(f"STEP 1: Facet Extraction (Clio-style)")
    print(f"{'='*70}")
    facets = extract_facets(prompts, facet_indices, client, args.model,
                            args.output_dir, args.resume)

    # Step 2: Embed facets
    print(f"\n{'='*70}")
    print(f"STEP 2: Embed Facets")
    print(f"{'='*70}")
    emb = embed_facets(facets, prompts, args.embed_model)

    # Step 3: KMeans sweep
    print(f"\n{'='*70}")
    print(f"STEP 3: KMeans Clustering")
    print(f"{'='*70}")

    best_bal, best_k = 0, 0
    all_res = []
    for k in args.k:
        labels = MiniBatchKMeans(n_clusters=k, random_state=args.seed,
                                 batch_size=4096, n_init=3).fit_predict(emb)
        met = compute_metrics(labels, n)
        gt = evaluate_gt(labels, args.gt_dir)
        bal = report(f"  Clio k={k}", met, gt)
        all_res.append({"k": k, **met, **gt, "bal": bal})
        if bal > best_bal: best_bal, best_k = bal, k

    # Also run SBERT baseline for direct comparison
    print(f"\n── SBERT baseline (no LLM facets) ──")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(args.embed_model)
    sbert_emb = sbert.encode(prompts, batch_size=256, show_progress_bar=True,
                             normalize_embeddings=True)
    best_sbert_bal = 0
    for k in args.k:
        labels = MiniBatchKMeans(n_clusters=k, random_state=args.seed,
                                 batch_size=4096, n_init=3).fit_predict(sbert_emb)
        met = compute_metrics(labels, n)
        gt = evaluate_gt(labels, args.gt_dir)
        bal = report(f"  SBERT k={k}", met, gt)
        if bal > best_sbert_bal: best_sbert_bal = bal

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Clio-style:      bal={best_bal:.4f} (k={best_k})")
    print(f"  SBERT baseline:  bal={best_sbert_bal:.4f}")
    print(f"  Δ from facets: {best_bal-best_sbert_bal:+.4f}")
    print(f"  Facets: {len(facets)}/{n} items, ~${est_cost:.2f}")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*70}")

    results = {"method": "Clio-style", "model": args.model,
               "clio_best": {"bal": best_bal, "k": best_k},
               "sbert_best": {"bal": best_sbert_bal},
               "n_facets": len(facets), "n_total": n,
               "all_k": all_res, "est_cost_usd": est_cost,
               "elapsed_s": round(elapsed)}
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
