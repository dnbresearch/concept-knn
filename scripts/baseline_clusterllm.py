#!/usr/bin/env python3
"""
ClusterLLM-style baseline (Zhang et al., EMNLP 2023)
=====================================================
Same interface as concept_knn_v5.py:
  --features-file features_v4.json
  --gt-dir ground_truth/

Pipeline:
  1. SBERT embed prompts from features JSON
  2. Initial KMeans clustering
  3. Entropy-based hard triplet sampling (top 40% entropy)
  4. Query LLM for triplet judgments (budget: 1024)
  5. Fine-tune SBERT with triplet loss
  6. Re-cluster with KMeans on fine-tuned embeddings
  7. Evaluate with same evaluate_gt() as concept_knn_v5

Usage:
  python baseline_clusterllm.py \\
    --features-file features_v4.json \\
    --gt-dir ground_truth/ \\
    --output-dir results/clusterllm \\
    --api-key YOUR_OPENAI_KEY

Requirements:
  pip install openai sentence-transformers scikit-learn numpy tqdm
"""

import json, os, sys, time, random, argparse
import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from eval_shared import load_features, evaluate_gt, compute_metrics, report


def compute_embeddings(prompts, model_name, batch_size=256):
    from sentence_transformers import SentenceTransformer
    print(f"  SBERT encode ({model_name})...")
    model = SentenceTransformer(model_name)
    emb = model.encode(prompts, batch_size=batch_size,
                       show_progress_bar=True, normalize_embeddings=True)
    print(f"  shape: {emb.shape}")
    return emb, model


def compute_entropy(embeddings, labels, k_neighbors=50):
    print("  Computing entropy scores...")
    nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(embeddings)-1),
                          metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    entropies = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        counts = Counter(labels[indices[i]])
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        entropies[i] = -np.sum(probs * np.log(probs + 1e-10))
    return entropies


def sample_triplets(embeddings, labels, entropies, n_triplets, seed=42):
    rng = np.random.RandomState(seed)
    high_idx = np.where(entropies >= np.percentile(entropies, 60))[0]
    c2idx = defaultdict(list)
    for i, l in enumerate(labels): c2idx[l].append(i)

    triplets = []
    for _ in range(n_triplets * 20):
        if len(triplets) >= n_triplets: break
        a = rng.choice(high_idx); ac = labels[a]
        if len(c2idx[ac]) < 2: continue
        p = rng.choice([x for x in c2idx[ac] if x != a])
        others = [c for c in c2idx if c != ac]
        if not others: continue
        n = rng.choice(c2idx[rng.choice(others)])
        triplets.append((int(a), int(p), int(n)))
    print(f"  {len(triplets)} triplets sampled")
    return triplets


def query_llm_triplets(prompts, triplets, api_key, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    results, total_tokens = [], 0
    print(f"  Querying {model} for {len(triplets)} triplets...")

    for ti, (ai, pi, ni) in enumerate(triplets):
        a, b, c = prompts[ai][:500], prompts[pi][:500], prompts[ni][:500]
        flip = random.random() > 0.5
        o1, o2 = (b, c) if flip else (c, b)
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=5, temperature=0,
                messages=[{"role": "user",
                           "content": f"Which option is more topically similar to the query?\n\n"
                                      f"Query: {a}\n\nOption 1: {o1}\n\nOption 2: {o2}\n\n"
                                      f"Reply '1' or '2' only."}])
            ans = resp.choices[0].message.content.strip()
            total_tokens += resp.usage.total_tokens
            if "1" in ans:
                w = pi if flip else ni; l = ni if flip else pi
            elif "2" in ans:
                w = ni if flip else pi; l = pi if flip else ni
            else: continue
            results.append((ai, w, l))
        except Exception as e:
            if "rate" in str(e).lower(): time.sleep(5)
        if (ti+1) % 200 == 0:
            print(f"    {ti+1}/{len(triplets)}, {len(results)} valid")

    print(f"  Done: {len(results)} valid, ~{total_tokens:,} tokens")
    return results, total_tokens


def finetune_embedder(model, prompts, triplet_results, epochs=3):
    from sentence_transformers import InputExample, losses
    from torch.utils.data import DataLoader
    print(f"  Fine-tuning ({epochs} epochs, {len(triplet_results)} triplets)...")
    examples = [InputExample(texts=[prompts[a], prompts[w], prompts[l]])
                for a, w, l in triplet_results]
    loader = DataLoader(examples, shuffle=True, batch_size=16)
    model.fit(train_objectives=[(loader, losses.TripletLoss(model))],
              epochs=epochs, warmup_steps=int(0.1*len(loader)),
              optimizer_params={'lr': 2e-5}, show_progress_bar=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='results/clusterllm')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--embed-model', default='all-MiniLM-L6-v2')
    parser.add_argument('--n-triplets', type=int, default=1024)
    parser.add_argument('--k', nargs='+', type=int, default=[5000, 10000, 15000])
    parser.add_argument('--finetune-epochs', type=int, default=3)
    parser.add_argument('--skip-finetune', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key: raise ValueError("Set --api-key or OPENAI_API_KEY")
    t0 = time.time()

    data, prompts = load_features(args.features_file)
    n = len(prompts)
    emb, model = compute_embeddings(prompts, args.embed_model)

    # Initial KMeans sweep
    print(f"\n── Initial SBERT + KMeans ──")
    best_bal, best_k, best_labels = 0, 0, None
    for k in args.k:
        labels = MiniBatchKMeans(n_clusters=k, random_state=args.seed,
                                 batch_size=4096, n_init=3).fit_predict(emb)
        met = compute_metrics(labels, n); gt = evaluate_gt(labels, args.gt_dir)
        bal = report(f"  k={k}", met, gt)
        if bal > best_bal: best_bal, best_k, best_labels = bal, k, labels

    # Triplet sampling + LLM queries
    entropies = compute_entropy(emb, best_labels)
    triplets = sample_triplets(emb, best_labels, entropies, args.n_triplets, args.seed)
    trip_results, tokens = query_llm_triplets(prompts, triplets, api_key, args.model)

    with open(os.path.join(args.output_dir, "triplets.json"), "w") as f:
        json.dump({"results": trip_results, "tokens": tokens}, f)

    if args.skip_finetune:
        results = {"method": "ClusterLLM (no-ft)", "bal": best_bal,
                   "best_k": best_k, "tokens": tokens}
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return

    # Fine-tune + re-cluster
    model = finetune_embedder(model, prompts, trip_results, args.finetune_epochs)
    new_emb = model.encode(prompts, batch_size=256, show_progress_bar=True,
                           normalize_embeddings=True)

    print(f"\n── ClusterLLM (fine-tuned) + KMeans ──")
    best_ft_bal, best_ft_k = 0, 0
    all_res = []
    for k in args.k:
        labels = MiniBatchKMeans(n_clusters=k, random_state=args.seed,
                                 batch_size=4096, n_init=3).fit_predict(new_emb)
        met = compute_metrics(labels, n); gt = evaluate_gt(labels, args.gt_dir)
        bal = report(f"  k={k}", met, gt)
        all_res.append({"k": k, **met, **gt, "bal": bal})
        if bal > best_ft_bal: best_ft_bal, best_ft_k = bal, k

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Initial:    bal={best_bal:.4f} (k={best_k})")
    print(f"  Fine-tuned: bal={best_ft_bal:.4f} (k={best_ft_k})")
    print(f"  Δ={best_ft_bal-best_bal:+.4f}, {tokens:,} tokens, {elapsed:.0f}s")
    print(f"{'='*70}")

    results = {"method": "ClusterLLM", "model": args.model,
               "initial": {"bal": best_bal, "k": best_k},
               "finetuned": {"bal": best_ft_bal, "k": best_ft_k},
               "all_k": all_res, "tokens": tokens, "elapsed_s": round(elapsed)}
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
