#!/usr/bin/env python3
"""
Ablation: Representation × Algorithm Matrix
=============================================

Fills in the missing cells to separate contributions:

             │  KMeans   │  Leiden+cascade
  ───────────┼───────────┼────────────────
  TF-IDF     │  0.469 ✓  │  ? (this script)
  Concepts   │  ? (this) │  0.627 ✓ (ours)
  SBERT      │  0.491 ✓  │  0.561 ✓

This lets us claim exactly how much the representation vs algorithm contributes.
"""

import sys
sys.modules['tensorflow'] = None
sys.modules['tensorflow.python'] = None

import json, os, time, warnings
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# EVALUATION (same as baselines.py)
# ═══════════════════════════════════════════════════════════════

def evaluate_gt(labels, gt_dir):
    results = {}
    if not gt_dir or not os.path.isdir(gt_dir):
        return results
    for gran in ['coarse', 'mid', 'fine']:
        for fname in sorted(os.listdir(gt_dir)):
            if f'gt_{gran}_' in fname and fname.endswith('.json') \
               and '_prompt_labels' not in fname:
                try:
                    with open(os.path.join(gt_dir, fname)) as f:
                        gt = json.load(f)
                    pl = gt['prompt_labels']
                    # Convert string keys to int if needed
                    if pl and isinstance(next(iter(pl.keys())), str):
                        pl = {int(k): v for k, v in pl.items()}
                    pred, true, enc = [], [], {}
                    for qi in range(len(labels)):
                        if qi in pl:
                            t = pl[qi]
                            if t not in enc: enc[t] = len(enc)
                            pred.append(int(labels[qi])); true.append(enc[t])
                    if len(pred) >= 10:
                        results[f'nmi_{gran}'] = round(
                            normalized_mutual_info_score(
                                np.array(true), np.array(pred)), 4)
                except Exception as e:
                    print(f"    GT err ({gran}): {e}")
                break
    return results


def report(name, labels, n_total, gt_dir, t_elapsed):
    sizes = Counter(labels)
    sa = np.array(sorted(sizes.values(), reverse=True))
    coverage = sum(s for s in sa if s > 1) / n_total
    singletons = sum(1 for s in sa if s == 1)
    gt = evaluate_gt(labels, gt_dir)
    nf = gt.get('nmi_fine', 0)
    bal = round(nf * coverage, 4)

    print(f"\n  {'─'*60}")
    print(f"  {name}")
    print(f"  {'─'*60}")
    print(f"    Clusters: {len(sizes):,} (real: {sum(1 for s in sa if s>1):,}, "
          f"singletons: {singletons:,})")
    print(f"    Coverage: {coverage:.1%}, largest: {sa[0]/n_total:.1%}")
    print(f"    NMI coarse: {gt.get('nmi_coarse', '?')}")
    print(f"    NMI mid:    {gt.get('nmi_mid', '?')}")
    print(f"    NMI fine:   {nf}")
    print(f"    BALANCED:   {bal}")
    print(f"    Time:       {t_elapsed:.1f}s")
    return {
        'method': name, 'time': round(t_elapsed, 1),
        'k_clusters': len(sizes), 'coverage': round(coverage, 4),
        'singletons': singletons, 'largest_pct': round(sa[0]/n_total, 4),
        **gt, 'bal': bal,
    }


# ═══════════════════════════════════════════════════════════════
# BUILD CONCEPT VECTORS (same as concept_knn_v5)
# ═══════════════════════════════════════════════════════════════

def build_concept_vectors(data, pmi_k=35):
    """Build sparse concept vectors with PMI densification."""
    from collections import defaultdict

    n = len(data)
    print(f"  Building concept vocabulary...")

    # Collect all concepts
    concept_counter = Counter()
    doc_concepts = []
    for item in data:
        feats = item.get('features', {})
        concepts = feats.get('concepts_all', [])
        concepts = [c.lower().strip() for c in concepts if len(c.strip()) > 1]
        doc_concepts.append(concepts)
        concept_counter.update(set(concepts))

    # IDF filter: keep concepts appearing in 3+ but <10% of docs
    min_df, max_df = 3, int(n * 0.1)
    vocab = {}
    for c, cnt in concept_counter.items():
        if min_df <= cnt <= max_df:
            vocab[c] = len(vocab)
    print(f"  Vocabulary: {len(vocab):,} concepts (from {len(concept_counter):,})")

    # Build sparse TF-IDF-style vectors
    idf = {}
    for c, idx in vocab.items():
        df = concept_counter[c]
        idf[c] = np.log(n / (df + 1))

    rows, cols, vals = [], [], []
    for i, concepts in enumerate(doc_concepts):
        tf = Counter(concepts)
        for c, count in tf.items():
            if c in vocab:
                rows.append(i)
                cols.append(vocab[c])
                vals.append(count * idf[c])

    X = csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
    X = normalize(X, norm='l2', axis=1)
    print(f"  Base vectors: {X.shape}, nnz={X.nnz:,}, "
          f"avg_nnz={X.nnz/n:.1f}")

    if pmi_k <= 0:
        return X

    # PMI densification
    print(f"  PMI densification (top-{pmi_k})...")
    concept_list = [''] * len(vocab)
    for c, idx in vocab.items():
        concept_list[idx] = c

    # Co-occurrence matrix
    cooc = defaultdict(lambda: defaultdict(int))
    concept_freq = defaultdict(int)
    for concepts in doc_concepts:
        unique = set(c for c in concepts if c in vocab)
        for c in unique:
            concept_freq[c] += 1
        unique_list = list(unique)
        for i_c in range(len(unique_list)):
            for j_c in range(i_c + 1, len(unique_list)):
                a, b = unique_list[i_c], unique_list[j_c]
                cooc[a][b] += 1
                cooc[b][a] += 1

    # For each concept, find top-K PMI associates
    pmi_assoc = {}  # concept_idx -> [(assoc_idx, pmi_score), ...]
    for c, idx in vocab.items():
        if c not in cooc:
            continue
        scores = []
        fc = concept_freq[c]
        for other, joint in cooc[c].items():
            if other not in vocab:
                continue
            fo = concept_freq[other]
            pmi = np.log2((joint * n) / (fc * fo + 1e-10) + 1e-10)
            if pmi > 0:
                scores.append((vocab[other], pmi))
        scores.sort(key=lambda x: -x[1])
        pmi_assoc[idx] = scores[:pmi_k]

    # Expand vectors
    print(f"  Expanding vectors with PMI associates...")
    X_dense = lil_matrix(X.shape, dtype=np.float32)
    X_csr = X.tocsr()
    for i in range(n):
        row = X_csr[i]
        _, orig_cols = row.nonzero()
        orig_vals = {c: row[0, c] for c in orig_cols}

        expanded = dict(orig_vals)
        for c_idx in orig_cols:
            if c_idx in pmi_assoc:
                orig_weight = orig_vals[c_idx]
                for assoc_idx, pmi_score in pmi_assoc[c_idx]:
                    add = orig_weight * 0.3 * (pmi_score / 10.0)
                    if assoc_idx in expanded:
                        expanded[assoc_idx] = max(expanded[assoc_idx], add)
                    else:
                        expanded[assoc_idx] = add

        for c_idx, val in expanded.items():
            X_dense[i, c_idx] = val

    X_out = normalize(X_dense.tocsr(), norm='l2', axis=1)
    print(f"  PMI vectors: nnz={X_out.nnz:,}, avg_nnz={X_out.nnz/n:.1f}")
    return X_out


# ═══════════════════════════════════════════════════════════════
# LEIDEN + CASCADE (same pipeline as our method)
# ═══════════════════════════════════════════════════════════════

def leiden_cascade(X, n_total, sigma=0.6, resolution=80.0, k_nn=30):
    """Run kNN → Leiden → cascade pass2. Works with any vector type."""

    t0 = time.time()

    # kNN
    print(f"    kNN (k={k_nn})...", end='', flush=True)
    nn = NearestNeighbors(n_neighbors=k_nn + 1, metric='cosine',
                           algorithm='brute', n_jobs=-1)
    nn.fit(X)
    knn_dist, knn_idx = nn.kneighbors(X)
    print(f" [{time.time()-t0:.1f}s]")

    # Build graph
    rows, cols, vals = [], [], []
    for i in range(n_total):
        for j_pos in range(1, knn_idx.shape[1]):
            j = int(knn_idx[i, j_pos])
            sim = 1.0 - knn_dist[i, j_pos]
            if sim >= sigma and j != i:
                a, b = min(i, j), max(i, j)
                rows.append(a); cols.append(b); vals.append(sim)
    graph = csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n_total, n_total))
    sym = graph + graph.T
    iso = int(np.sum(np.diff(sym.tocsr().indptr) == 0))
    print(f"    Graph: {graph.nnz:,} edges, iso={iso:,}")

    # Leiden
    print(f"    Leiden (res={resolution})...", end='', flush=True)
    try:
        import igraph as ig; import leidenalg
        g = ig.Graph.Weighted_Adjacency(sym, mode='undirected')
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=resolution,
            seed=42, n_iterations=3)
        base_labels = np.array(part.membership)
    except ImportError:
        import networkx as nx
        import networkx.algorithms.community as nx_comm
        G = nx.from_scipy_sparse_array(sym, edge_attribute='weight')
        comms = nx_comm.louvain_communities(G, weight='weight',
                                             resolution=resolution, seed=42)
        base_labels = np.zeros(n_total, dtype=int)
        for cid, members in enumerate(comms):
            for m in members: base_labels[m] = cid
    print(f" [{time.time()-t0:.1f}s]")

    # Cascade pass2
    # Need dense matrix for centroid computation
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)

    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    nl = base_labels.copy(); n = len(nl)
    for thr in thresholds:
        sizes = Counter(nl)
        sings = np.array([i for i in range(n) if sizes[nl[i]] == 1])
        non_s = [i for i in range(n) if sizes[nl[i]] > 1]
        if len(sings) == 0 or len(non_s) == 0:
            break
        rc = sorted(set(nl[i] for i in non_s))
        cl = []
        for c in rc:
            m = np.where(nl == c)[0]
            cl.append(X_dense[m].mean(axis=0))
        centroids = np.array(cl, dtype=np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1
        centroids = centroids / norms
        sv = X_dense[sings]
        sims = sv @ centroids.T
        for i, idx in enumerate(sings):
            bp = np.argmax(sims[i])
            if sims[i, bp] >= thr:
                nl[idx] = rc[bp]

    return nl, time.time() - t0


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='results_ablation')
    parser.add_argument('--pmi-k', type=int, default=35)
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Skip: concept_km tfidf_leiden concept_leiden tfidf_km')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data..."); t0 = time.time()
    with open(args.features_file) as f:
        data = json.load(f)
    n = len(data)
    print(f"  {n:,} items [{time.time()-t0:.1f}s]")

    texts = [item.get('prompt', '')[:2000] for item in data]

    all_results = []

    # ═══════════════════════════════════════════════════
    # ABLATION A: Concept vectors + KMeans
    # ═══════════════════════════════════════════════════
    if 'concept_km' not in args.skip:
        print(f"\n{'='*70}")
        print(f"ABLATION A: Concept Vectors (PMI={args.pmi_k}) + KMeans")
        print(f"{'='*70}")

        t0 = time.time()
        X_concept = build_concept_vectors(data, pmi_k=args.pmi_k)
        print(f"  Concept vectors built [{time.time()-t0:.1f}s]")

        for k in [1000, 3000, 5000, 10000, 15000]:
            km = MiniBatchKMeans(n_clusters=k, random_state=42,
                                  batch_size=5000, n_init=3, max_iter=300)
            labels = km.fit_predict(X_concept)
            r = report(f"Concept + KMeans (k={k})", labels, n,
                        args.gt_dir, time.time() - t0)
            r['k_param'] = k; r['ablation'] = 'concept_km'
            all_results.append(r)

    # ═══════════════════════════════════════════════════
    # ABLATION B: TF-IDF + Leiden+cascade
    # ═══════════════════════════════════════════════════
    if 'tfidf_leiden' not in args.skip:
        print(f"\n{'='*70}")
        print(f"ABLATION B: TF-IDF + Leiden+Cascade")
        print(f"{'='*70}")

        t0 = time.time()
        vectorizer = TfidfVectorizer(
            max_features=30000, min_df=3, max_df=0.1,
            ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
        X_tfidf = vectorizer.fit_transform(texts)
        X_tfidf = normalize(X_tfidf, norm='l2', axis=1)
        print(f"  TF-IDF: {X_tfidf.shape} [{time.time()-t0:.1f}s]")

        for sigma in [0.3, 0.4, 0.5]:
            for res in [30.0, 50.0, 80.0]:
                print(f"\n  Config: σ={sigma}, res={res}")
                labels, elapsed = leiden_cascade(
                    X_tfidf, n, sigma=sigma, resolution=res)
                r = report(f"TF-IDF + Leiden (σ={sigma}, res={res})",
                            labels, n, args.gt_dir, elapsed)
                r['sigma'] = sigma; r['resolution'] = res
                r['ablation'] = 'tfidf_leiden'
                all_results.append(r)

    # ═══════════════════════════════════════════════════
    # ABLATION C: Concept vectors + Leiden (our method, for reference)
    # ═══════════════════════════════════════════════════
    if 'concept_leiden' not in args.skip:
        print(f"\n{'='*70}")
        print(f"ABLATION C: Concept Vectors (PMI={args.pmi_k}) + Leiden+Cascade")
        print(f"{'='*70}")

        t0 = time.time()
        # Rebuild if not already done
        try:
            X_concept
        except NameError:
            X_concept = build_concept_vectors(data, pmi_k=args.pmi_k)

        for sigma in [0.6, 0.65, 0.7]:
            for res in [50.0, 80.0]:
                print(f"\n  Config: σ={sigma}, res={res}")
                labels, elapsed = leiden_cascade(
                    X_concept, n, sigma=sigma, resolution=res)
                r = report(f"Concept + Leiden (σ={sigma}, res={res})",
                            labels, n, args.gt_dir, elapsed)
                r['sigma'] = sigma; r['resolution'] = res
                r['ablation'] = 'concept_leiden'
                all_results.append(r)

    # ═══════════════════════════════════════════════════
    # ABLATION D: TF-IDF + KMeans (higher k, for reference)
    # ═══════════════════════════════════════════════════
    if 'tfidf_km' not in args.skip:
        print(f"\n{'='*70}")
        print(f"ABLATION D: TF-IDF + KMeans (higher k)")
        print(f"{'='*70}")

        t0 = time.time()
        try:
            X_tfidf
        except NameError:
            vectorizer = TfidfVectorizer(
                max_features=30000, min_df=3, max_df=0.1,
                ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
            X_tfidf = vectorizer.fit_transform(texts)

        for k in [15000, 20000]:
            km = MiniBatchKMeans(n_clusters=k, random_state=42,
                                  batch_size=5000, n_init=3, max_iter=300)
            labels = km.fit_predict(X_tfidf)
            r = report(f"TF-IDF + KMeans (k={k})", labels, n,
                        args.gt_dir, time.time() - t0)
            r['k_param'] = k; r['ablation'] = 'tfidf_km'
            all_results.append(r)

    # ═══════════════════════════════════════════════════
    # SUMMARY: 2×2 Matrix
    # ═══════════════════════════════════════════════════
    all_results.sort(key=lambda x: x['bal'], reverse=True)

    print(f"\n{'━'*100}")
    print(f"  ABLATION RESULTS")
    print(f"{'━'*100}")

    for r in all_results:
        print(f"  {r['method']:<45} NMI_f={r.get('nmi_fine',0):.4f} "
              f"cov={r['coverage']:.1%} bal={r['bal']:.4f}")

    # Find best per ablation
    best = {}
    for r in all_results:
        ab = r.get('ablation', '?')
        if ab not in best or r['bal'] > best[ab]['bal']:
            best[ab] = r

    print(f"\n{'━'*100}")
    print(f"  REPRESENTATION × ALGORITHM MATRIX")
    print(f"{'━'*100}")
    print(f"                    │  KMeans (best k)    │  Leiden+cascade     ")
    print(f"  ──────────────────┼─────────────────────┼─────────────────────")

    # TF-IDF row
    tfidf_km = best.get('tfidf_km', {})
    tfidf_lei = best.get('tfidf_leiden', {})
    print(f"  TF-IDF            │  bal={tfidf_km.get('bal','?'):<16} │  bal={tfidf_lei.get('bal','?'):<16}")

    # Concept row
    conc_km = best.get('concept_km', {})
    conc_lei = best.get('concept_leiden', {})
    print(f"  Concepts (PMI={args.pmi_k:>2}) │  bal={conc_km.get('bal','?'):<16} │  bal={conc_lei.get('bal','?'):<16}")

    # SBERT row (from previous runs)
    print(f"  SBERT (384d)      │  bal=0.4906 (k=10K) │  bal=0.5606 (σ=0.7)")

    print(f"\n  Representation effect (Leiden col): "
          f"Concept vs SBERT = {conc_lei.get('bal',0):.4f} vs 0.5606")
    print(f"  Algorithm effect (Concept row):     "
          f"Leiden vs KMeans = {conc_lei.get('bal',0):.4f} vs {conc_km.get('bal',0):.4f}")

    # Save
    out_path = os.path.join(args.output_dir, 'ablation.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Saved: {out_path}")


if __name__ == '__main__':
    main()