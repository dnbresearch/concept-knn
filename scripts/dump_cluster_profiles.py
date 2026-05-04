#!/usr/bin/env python3
"""
Cluster Profile Dumper — Run best Concept-kNN config and save cluster details.
==============================================================================

Run AFTER you know the best config. Outputs cluster_profiles.json with:
  - Per-cluster top concepts (by weight × prevalence)
  - Per-cluster example prompts (concept-richest items)
  - Per-cluster GT label distribution + purity
  - Summary statistics

USAGE:
  python dump_cluster_profiles.py \
    --features-file features_v4.json \
    --gt-dir ground_truth/ \
    --output cluster_profiles.json \
    --pmi-k 35 --alpha 0.8 --sigma 0.6 --resolution 50

This runs ONE config (the best) and saves detailed cluster profiles.
"""

import json, os, sys, time, gc, math, re, copy, argparse
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score

# ── Import all functions from concept_knn_v5 ──
# (paste the functions here, or do: from concept_knn_v5 import *)
from concept_knn_v3 import (
    refilter_v5_relaxed,
    propagate_conversation_concepts,
    pmi_densify,
    build_tfidf_vectors,
    build_concept_vectors,
    build_hybrid_vectors,
    community_detect,
    pass2_cascading,
    compute_metrics,
    evaluate_gt,
)


def dump_cluster_profiles(final_labels, data, concepts, c2i, gt_dir=None,
                          top_n_concepts=15, top_n_prompts=5,
                          output_path='cluster_profiles.json', verbose=True):
    """
    Profile each cluster found by Concept-kNN.
    
    For each cluster, computes:
      - Top concepts by (mean_weight × cluster_prevalence)
      - Example prompts (concept-richest items with actual text)
      - GT label distribution at coarse and fine granularity
      - Cluster purity
    """
    n = len(final_labels)
    
    # ── 1. Group items by cluster ──
    cluster_members = defaultdict(list)
    for i, label in enumerate(final_labels):
        cluster_members[int(label)].append(i)
    
    # ── 2. Load GT labels if available ──
    gt_labels = {}
    gt_coarse = {}
    if gt_dir and os.path.isdir(gt_dir):
        for fname in sorted(os.listdir(gt_dir)):
            if fname.endswith('.json') and '_prompt_labels' not in fname:
                gran = None
                if 'gt_fine_' in fname: gran = 'fine'
                elif 'gt_coarse_' in fname: gran = 'coarse'
                if gran is None: continue
                try:
                    with open(os.path.join(gt_dir, fname)) as f:
                        gt = json.load(f)
                    pl = gt.get('prompt_labels', {})
                    for qi, cat in pl.items():
                        key = int(qi) if isinstance(qi, str) and qi.isdigit() else qi
                        if gran == 'fine':
                            gt_labels[key] = cat
                        else:
                            gt_coarse[key] = cat
                except Exception as e:
                    print(f"  Warning: failed to load {fname}: {e}")
    
    if verbose:
        print(f"  GT loaded: {len(gt_labels)} fine, {len(gt_coarse)} coarse labels")
    
    # ── 3. Profile each cluster ──
    profiles = []
    
    for cid in sorted(cluster_members.keys(),
                      key=lambda c: len(cluster_members[c]), reverse=True):
        members = cluster_members[cid]
        size = len(members)
        
        # --- Top concepts by mean weight × prevalence ---
        concept_weight_sum = defaultdict(float)
        concept_count = defaultdict(int)
        
        for idx in members:
            weights = data[idx].get('features', {}).get('concept_weights', {})
            for c, w in weights.items():
                concept_weight_sum[c] += w
                concept_count[c] += 1
        
        concept_scores = {}
        for c in concept_weight_sum:
            mean_w = concept_weight_sum[c] / size
            prevalence = concept_count[c] / size
            concept_scores[c] = mean_w * prevalence
        
        top_concepts = sorted(concept_scores.items(),
                              key=lambda x: x[1], reverse=True)[:top_n_concepts]
        
        # --- Example prompts ---
        prompt_entries = []
        for idx in members:
            item = data[idx]
            prompt = item.get('prompt', '') or item.get('features', {}).get('raw_prompt', '')
            n_concepts = len(item.get('features', {}).get('concepts_all', []))
            if prompt and len(prompt.strip()) > 10:
                prompt_entries.append((idx, prompt, n_concepts))
        
        prompt_entries.sort(key=lambda x: x[2], reverse=True)
        example_prompts = []
        for idx, prompt, nc in prompt_entries[:top_n_prompts]:
            truncated = prompt[:300].strip()
            if len(prompt) > 300:
                truncated += '...'
            example_prompts.append({
                'idx': int(idx),
                'prompt': truncated,
                'n_concepts': nc,
                'concepts': data[idx].get('features', {}).get('concepts_all', [])[:10],
            })
        
        # --- GT distribution ---
        gt_dist_fine = Counter()
        gt_dist_coarse = Counter()
        for idx in members:
            if idx in gt_labels:
                gt_dist_fine[gt_labels[idx]] += 1
            if idx in gt_coarse:
                gt_dist_coarse[gt_coarse[idx]] += 1
        
        purity_fine = (max(gt_dist_fine.values()) / sum(gt_dist_fine.values())
                       if gt_dist_fine else None)
        purity_coarse = (max(gt_dist_coarse.values()) / sum(gt_dist_coarse.values())
                         if gt_dist_coarse else None)
        
        profile = {
            'cluster_id': int(cid),
            'size': size,
            'top_concepts': [{'concept': c, 'score': round(s, 6)} 
                             for c, s in top_concepts],
            'example_prompts': example_prompts,
            'gt_fine': {
                'dominant_label': gt_dist_fine.most_common(1)[0][0] if gt_dist_fine else None,
                'purity': round(purity_fine, 4) if purity_fine is not None else None,
                'top_labels': [{'label': l, 'count': c} 
                               for l, c in gt_dist_fine.most_common(5)],
            },
            'gt_coarse': {
                'dominant_label': gt_dist_coarse.most_common(1)[0][0] if gt_dist_coarse else None,
                'purity': round(purity_coarse, 4) if purity_coarse is not None else None,
                'top_labels': [{'label': l, 'count': c}
                               for l, c in gt_dist_coarse.most_common(5)],
            },
        }
        profiles.append(profile)
    
    # ── 4. Summary ──
    sizes = [p['size'] for p in profiles]
    purities_c = [p['gt_coarse']['purity'] for p in profiles 
                  if p['gt_coarse']['purity'] is not None]
    purities_f = [p['gt_fine']['purity'] for p in profiles 
                  if p['gt_fine']['purity'] is not None]
    
    output = {
        'summary': {
            'n_clusters': len(profiles),
            'n_items': n,
            'size_stats': {
                'mean': round(np.mean(sizes), 1),
                'median': round(np.median(sizes), 1),
                'min': int(min(sizes)),
                'max': int(max(sizes)),
            },
            'purity_coarse': {
                'mean': round(np.mean(purities_c), 4) if purities_c else None,
                'median': round(np.median(purities_c), 4) if purities_c else None,
            },
            'purity_fine': {
                'mean': round(np.mean(purities_f), 4) if purities_f else None,
                'median': round(np.median(purities_f), 4) if purities_f else None,
            },
        },
        'clusters': profiles,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"CLUSTER PROFILES — {len(profiles)} clusters, {n} items")
        print(f"{'='*70}")
        print(f"  Size: mean={output['summary']['size_stats']['mean']}, "
              f"median={output['summary']['size_stats']['median']}, "
              f"range=[{output['summary']['size_stats']['min']}, "
              f"{output['summary']['size_stats']['max']}]")
        if purities_c:
            print(f"  Coarse purity: mean={np.mean(purities_c):.3f}, "
                  f"median={np.median(purities_c):.3f}")
        
        print(f"\n  {'Size':>6} {'Pur_c':>6} {'GT coarse':>20}  Top Concepts")
        print(f"  {'─'*6} {'─'*6} {'─'*20}  {'─'*50}")
        for p in profiles[:30]:
            pur = f"{p['gt_coarse']['purity']:.2f}" if p['gt_coarse']['purity'] else '  -  '
            dom = (p['gt_coarse']['dominant_label'] or '-')[:20]
            tc = ', '.join(c['concept'] for c in p['top_concepts'][:5])
            print(f"  {p['size']:>6} {pur:>6} {dom:>20}  {tc}")
        if len(profiles) > 30:
            print(f"  ... and {len(profiles)-30} more")
        
        # Detailed examples for top 5 clusters
        print(f"\n{'─'*70}")
        print(f"DETAILED CLUSTER EXAMPLES (top 5 by size)")
        print(f"{'─'*70}")
        for p in profiles[:5]:
            print(f"\n  ▸ Cluster #{p['cluster_id']} — {p['size']} items "
                  f"(GT: {p['gt_coarse']['dominant_label']}, "
                  f"purity={p['gt_coarse']['purity']})")
            print(f"    Concepts: {', '.join(c['concept'] for c in p['top_concepts'][:10])}")
            for j, ex in enumerate(p['example_prompts'][:3]):
                print(f"    Prompt {j+1}: \"{ex['prompt'][:150]}\"")
                print(f"      → concepts: {ex['concepts'][:6]}")
        
        print(f"\n  ✓ Saved: {output_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description='Dump Concept-kNN cluster profiles')
    parser.add_argument('--features-file', required=True,
                        help='Path to features_v4.json')
    parser.add_argument('--gt-dir', default=None,
                        help='Path to ground truth directory')
    parser.add_argument('--output', default='cluster_profiles.json',
                        help='Output JSON path')
    parser.add_argument('--pmi-k', type=int, default=35,
                        help='PMI top-K associates (default: 35)')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Blend ratio (default: 0.8)')
    parser.add_argument('--sigma', type=float, default=0.6,
                        help='Similarity threshold (default: 0.6)')
    parser.add_argument('--resolution', type=float, default=50.0,
                        help='Leiden resolution (default: 50)')
    parser.add_argument('--top-concepts', type=int, default=15,
                        help='Top concepts per cluster (default: 15)')
    parser.add_argument('--top-prompts', type=int, default=5,
                        help='Example prompts per cluster (default: 5)')
    args = parser.parse_args()
    
    t0 = time.time()
    
    # ── Load data ──
    print(f"Loading {args.features_file}...")
    with open(args.features_file) as f:
        raw_data = json.load(f)
    n_total = len(raw_data)
    print(f"  {n_total:,} items")
    
    # ── Preprocess ──
    base_data, idf, high_idf = refilter_v5_relaxed(copy.deepcopy(raw_data))
    base_data = propagate_conversation_concepts(base_data, idf, high_idf)
    
    # Build base concept vocabulary
    all_c = set()
    for item in base_data:
        all_c.update(item['features'].get('concepts_all', []))
    concepts_base = sorted(all_c)
    c2i_base = {c: i for i, c in enumerate(concepts_base)}
    
    # ── PMI densification ──
    if args.pmi_k > 0:
        data_pmi = copy.deepcopy(base_data)
        data_pmi = pmi_densify(data_pmi, concepts_base, c2i_base,
                               top_k_associates=args.pmi_k)
    else:
        data_pmi = base_data
    
    # ── Build vectors ──
    QC, QC_norm, concepts, c2i = build_concept_vectors(data_pmi, verbose=True)
    
    if args.alpha < 1.0:
        tfidf_norm = build_tfidf_vectors(raw_data, max_features=5000)
        X_norm = build_hybrid_vectors(QC_norm, tfidf_norm, alpha=args.alpha)
    else:
        X_norm = QC_norm
    
    # ── kNN graph ──
    print(f"\n  Building kNN graph (k=30, σ={args.sigma})...")
    t_knn = time.time()
    nn = NearestNeighbors(n_neighbors=31, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(X_norm)
    knn_dist, knn_idx = nn.kneighbors(X_norm)
    
    rows, cols, vals = [], [], []
    for i in range(n_total):
        for j_pos in range(1, knn_idx.shape[1]):
            j = int(knn_idx[i, j_pos])
            sim = 1.0 - knn_dist[i, j_pos]
            if sim >= args.sigma and j != i:
                a, b = min(i, j), max(i, j)
                rows.append(a); cols.append(b); vals.append(sim)
    graph = csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n_total, n_total))
    print(f"  Graph: {graph.nnz:,} edges [{time.time()-t_knn:.1f}s]")
    
    # ── Leiden + cascade ──
    print(f"  Clustering (resolution={args.resolution})...")
    base_labels = community_detect(graph, n_total, resolution=args.resolution)
    final_labels = pass2_cascading(base_labels, X_norm, verbose=True)
    
    # ── Evaluate ──
    met = compute_metrics(final_labels, n_total)
    gt_results = evaluate_gt(final_labels, args.gt_dir)
    nf = gt_results.get('nmi_fine', 0)
    bal = round(nf * met['coverage'], 4)
    
    print(f"\n  Results: NMI_f={nf:.4f}, coverage={met['coverage']:.1%}, "
          f"bal={bal:.4f}, clusters={met['k_query']:,}")
    
    # ── Dump profiles ──
    dump_cluster_profiles(
        final_labels=final_labels,
        data=data_pmi,  # Use PMI-enriched data for concept info
        concepts=concepts,
        c2i=c2i,
        gt_dir=args.gt_dir,
        top_n_concepts=args.top_concepts,
        top_n_prompts=args.top_prompts,
        output_path=args.output,
        verbose=True,
    )
    
    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
