#!/usr/bin/env python3
"""
Clustering Baselines — Comprehensive Comparison
=================================================
"""

# CRITICAL: Block broken TensorFlow BEFORE any imports
import sys
sys.modules['tensorflow'] = None
sys.modules['tensorflow.python'] = None
sys.modules['tensorflow.python.platform'] = None

import json, os, time, gc, warnings
import numpy as np
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score

warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_gt(labels, gt_dir):
    results = {}
    if not gt_dir or not os.path.isdir(gt_dir):
        return results
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(gt_dir)))
        from llm_ground_truth import load_ground_truth
    except ImportError:
        def load_ground_truth(path):
            with open(path) as f:
                gt = json.load(f)
            if 'prompt_labels' not in gt:
                pl, cats = {}, set()
                for item in gt.get('items', gt.get('data', [])):
                    qi = item.get('query_idx', item.get('idx'))
                    cat = item.get('category', item.get('label'))
                    if qi is not None and cat is not None:
                        pl[qi] = cat; cats.add(cat)
                return {'prompt_labels': pl, 'n_categories': len(cats)}
            return gt

    for gran in ['coarse', 'mid', 'fine']:
        for fname in sorted(os.listdir(gt_dir)):
            if f'gt_{gran}_' in fname and fname.endswith('.json') \
               and '_prompt_labels' not in fname:
                try:
                    gt = load_ground_truth(os.path.join(gt_dir, fname))
                    pl = gt['prompt_labels']
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


def compute_metrics(labels, n_total):
    sizes = Counter(labels)
    sa = np.array(sorted(sizes.values(), reverse=True))
    coverage = sum(s for s in sa if s > 1) / n_total
    singletons = sum(1 for s in sa if s == 1)
    return {
        'k_clusters': len(sizes),
        'n_real': sum(1 for s in sa if s > 1),
        'coverage': round(coverage, 4),
        'singletons': singletons,
        'largest_pct': round(sa[0] / n_total, 4),
    }


def report_result(name, labels, n_total, gt_dir, t_elapsed):
    met = compute_metrics(labels, n_total)
    gt = evaluate_gt(labels, gt_dir)
    nf = gt.get('nmi_fine', 0)
    cv = met['coverage']
    bal = round(nf * cv, 4)
    print(f"\n  {'─'*60}")
    print(f"  {name}")
    print(f"  {'─'*60}")
    print(f"    Clusters: {met['k_clusters']:,} "
          f"(real: {met['n_real']:,}, singletons: {met['singletons']:,})")
    print(f"    Coverage: {cv:.1%}, largest: {met['largest_pct']:.1%}")
    print(f"    NMI coarse: {gt.get('nmi_coarse', '?')}")
    print(f"    NMI mid:    {gt.get('nmi_mid', '?')}")
    print(f"    NMI fine:   {nf}")
    print(f"    BALANCED:   {bal}")
    print(f"    Time:       {t_elapsed:.1f}s")
    return {
        'method': name, 'time': round(t_elapsed, 1),
        **met, **gt, 'bal': bal,
    }


# ═══════════════════════════════════════════════════════════════
# 1. TF-IDF + KMeans
# ═══════════════════════════════════════════════════════════════

def run_tfidf_kmeans(texts, n_total, gt_dir,
                      k_values=[500, 1000, 2000, 3000, 5000, 10000, 15000]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans

    print(f"\n{'='*70}")
    print(f"BASELINE 1: TF-IDF + KMeans")
    print(f"{'='*70}")

    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=20000, min_df=3, max_df=0.1,
        ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X = vectorizer.fit_transform(texts)
    print(f"  TF-IDF: {X.shape[0]:,} × {X.shape[1]:,} [{time.time()-t0:.1f}s]")

    results = []
    for k in k_values:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=5000,
                              n_init=3, max_iter=300)
        labels = km.fit_predict(X)
        r = report_result(f"TF-IDF + KMeans (k={k})", labels, n_total,
                           gt_dir, time.time() - t0)
        r['k_param'] = k
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 2. TF-IDF + HDBSCAN
# ═══════════════════════════════════════════════════════════════

def run_tfidf_hdbscan(texts, n_total, gt_dir,
                       min_cluster_sizes=[10, 25, 50]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import hdbscan

    print(f"\n{'='*70}")
    print(f"BASELINE 2: TF-IDF + HDBSCAN")
    print(f"{'='*70}")

    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=20000, min_df=3, max_df=0.1,
        ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_dense = svd.fit_transform(X)
    print(f"  TF-IDF→SVD(50): [{time.time()-t0:.1f}s]")

    results = []
    for mcs in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=5,
            metric='euclidean', core_dist_n_jobs=-1)
        labels = clusterer.fit_predict(X_dense)
        max_label = labels.max() + 1
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = max_label; max_label += 1
        r = report_result(f"TF-IDF + HDBSCAN (min_cs={mcs})", labels,
                           n_total, gt_dir, time.time() - t0)
        r['min_cluster_size'] = mcs
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 3. LDA
# ═══════════════════════════════════════════════════════════════

def run_lda(texts, n_total, gt_dir, n_topics_list=[100]):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    print(f"\n{'='*70}")
    print(f"BASELINE 3: LDA")
    print(f"{'='*70}")

    t0 = time.time()
    vectorizer = CountVectorizer(
        max_features=20000, min_df=3, max_df=0.1,
        ngram_range=(1, 1), stop_words='english')
    X = vectorizer.fit_transform(texts)
    print(f"  BoW: {X.shape[0]:,} × {X.shape[1]:,} [{time.time()-t0:.1f}s]")

    results = []
    for n_topics in n_topics_list:
        t1 = time.time()
        print(f"  LDA (n_topics={n_topics})...", end='', flush=True)
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42,
            max_iter=20, learning_method='online',
            batch_size=2000, n_jobs=-1)
        doc_topics = lda.fit_transform(X)
        labels = np.argmax(doc_topics, axis=1)
        print(f" [{time.time()-t1:.1f}s]")
        r = report_result(f"LDA (n_topics={n_topics})", labels, n_total,
                           gt_dir, time.time() - t0)
        r['n_topics'] = n_topics
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 4. SBERT + KMeans (uses precomputed embeddings)
# ═══════════════════════════════════════════════════════════════

def run_sbert_kmeans(embeddings, n_total, gt_dir,
                      k_values=[500, 1000, 2000, 3000, 5000, 10000, 15000]):
    from sklearn.cluster import MiniBatchKMeans

    print(f"\n{'='*70}")
    print(f"BASELINE 4: SBERT + KMeans")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"{'='*70}")

    t0 = time.time()
    results = []
    for k in k_values:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=5000,
                              n_init=3, max_iter=300)
        labels = km.fit_predict(embeddings)
        r = report_result(f"SBERT + KMeans (k={k})", labels, n_total,
                           gt_dir, time.time() - t0)
        r['k_param'] = k
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 5. SBERT + HDBSCAN
# ═══════════════════════════════════════════════════════════════

def run_sbert_hdbscan(embeddings, n_total, gt_dir,
                       min_cluster_sizes=[10, 25, 50, 100]):
    import hdbscan

    print(f"\n{'='*70}")
    print(f"BASELINE 5: SBERT + HDBSCAN")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        import umap
        print(f"  UMAP reducing {embeddings.shape[1]}→50...", end='', flush=True)
        reducer = umap.UMAP(n_components=50, metric='cosine',
                             n_neighbors=30, min_dist=0.0, random_state=42)
        X_reduced = reducer.fit_transform(embeddings)
        print(f" [{time.time()-t0:.1f}s]")
    except ImportError:
        print(f"  No UMAP, using raw embeddings")
        X_reduced = embeddings

    results = []
    for mcs in min_cluster_sizes:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs, min_samples=5,
            metric='euclidean', core_dist_n_jobs=-1)
        labels = clusterer.fit_predict(X_reduced)
        max_label = labels.max() + 1
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = max_label; max_label += 1
        r = report_result(f"SBERT + HDBSCAN (min_cs={mcs})", labels,
                           n_total, gt_dir, time.time() - t0)
        r['min_cluster_size'] = mcs
        results.append(r)
    return results, X_reduced  # Return UMAP-reduced for reuse


# ═══════════════════════════════════════════════════════════════
# 6. BERTopic
# ═══════════════════════════════════════════════════════════════

def run_bertopic(texts, embeddings, n_total, gt_dir,
                  min_topic_sizes=[10, 25, 50]):
    from bertopic import BERTopic
    import umap

    print(f"\n{'='*70}")
    print(f"BASELINE 6: BERTopic")
    print(f"{'='*70}")

    results = []
    for mts in min_topic_sizes:
        t0 = time.time()
        print(f"\n  BERTopic (min_topic_size={mts})...")

        umap_model = umap.UMAP(
            n_components=50, metric='cosine',
            n_neighbors=30, min_dist=0.0, random_state=42)

        topic_model = BERTopic(
            umap_model=umap_model,
            min_topic_size=mts,
            nr_topics='auto',
            verbose=True)

        topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
        labels = np.array(topics)

        max_label = labels.max() + 1
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = max_label; max_label += 1

        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = sum(1 for t in topics if t == -1)
        print(f"    Topics: {n_topics}, outliers: {n_outliers:,}")

        r = report_result(f"BERTopic (min_ts={mts})", labels, n_total,
                           gt_dir, time.time() - t0)
        r['min_topic_size'] = mts
        r['n_topics_found'] = n_topics
        r['n_outliers'] = n_outliers
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 7. SBERT + Leiden (same algorithm as ours, different vectors)
# ═══════════════════════════════════════════════════════════════

def run_sbert_leiden(embeddings, n_total, gt_dir,
                      sigma_values=[0.5, 0.6, 0.7, 0.8],
                      resolutions=[10.0, 30.0, 50.0]):
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    print(f"\n{'='*70}")
    print(f"BASELINE 7: SBERT + Leiden")
    print(f"{'='*70}")

    t0 = time.time()
    print(f"  kNN (k=30)...", end='', flush=True)
    nn = NearestNeighbors(n_neighbors=31, metric='cosine',
                           algorithm='brute', n_jobs=-1)
    nn.fit(embeddings)
    knn_dist, knn_idx = nn.kneighbors(embeddings)
    print(f" [{time.time()-t0:.1f}s]")

    results = []
    for sigma in sigma_values:
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
        iso = int(np.sum(np.diff((graph + graph.T).tocsr().indptr) == 0))
        print(f"\n  σ={sigma}: {graph.nnz:,} edges, iso={iso:,}")

        for res in resolutions:
            sym = graph + graph.T
            try:
                import igraph as ig; import leidenalg
                g = ig.Graph.Weighted_Adjacency(sym, mode='undirected')
                part = leidenalg.find_partition(
                    g, leidenalg.RBConfigurationVertexPartition,
                    weights='weight', resolution_parameter=res,
                    seed=42, n_iterations=3)
                base_labels = np.array(part.membership)
            except ImportError:
                import networkx as nx
                import networkx.algorithms.community as nx_comm
                G = nx.from_scipy_sparse_array(sym, edge_attribute='weight')
                comms = nx_comm.louvain_communities(G, weight='weight',
                                                     resolution=res, seed=42)
                base_labels = np.zeros(n_total, dtype=int)
                for cid, members in enumerate(comms):
                    for m in members: base_labels[m] = cid

            # Cascade pass2
            thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
            nl = base_labels.copy(); n = len(nl)
            for thr in thresholds:
                sizes = Counter(nl)
                sings = np.array([i for i in range(n) if sizes[nl[i]] == 1])
                non_s = [i for i in range(n) if sizes[nl[i]] > 1]
                if len(sings) == 0 or len(non_s) == 0: break
                rc = sorted(set(nl[i] for i in non_s))
                cl = []
                for c in rc:
                    m = np.where(nl == c)[0]
                    cl.append(embeddings[m].mean(axis=0))
                centroids = np.array(cl, dtype=np.float32)
                norms = np.linalg.norm(centroids, axis=1, keepdims=True)
                norms[norms == 0] = 1
                centroids = centroids / norms
                sv = embeddings[sings]
                sims = sv @ centroids.T
                for i, idx in enumerate(sings):
                    bp = np.argmax(sims[i])
                    if sims[i, bp] >= thr: nl[idx] = rc[bp]

            r = report_result(f"SBERT + Leiden (σ={sigma}, res={res})",
                               nl, n_total, gt_dir, time.time() - t0)
            r['sigma'] = sigma; r['resolution'] = res
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 8. SBERT + Spectral Clustering (Ng et al. 2002)
#    Classic graph-based method, highly cited
# ═══════════════════════════════════════════════════════════════

def run_sbert_spectral(embeddings, n_total, gt_dir,
                        k_values=[500, 1000, 2000, 3000]):
    from sklearn.cluster import SpectralClustering

    print(f"\n{'='*70}")
    print(f"BASELINE 8: SBERT + Spectral Clustering")
    print(f"{'='*70}")

    # Spectral needs affinity matrix — use kNN precomputed
    from sklearn.neighbors import NearestNeighbors
    t0 = time.time()
    print(f"  Building affinity...", end='', flush=True)
    nn = NearestNeighbors(n_neighbors=31, metric='cosine',
                           algorithm='brute', n_jobs=-1)
    nn.fit(embeddings)
    affinity = nn.kneighbors_graph(embeddings, mode='connectivity')
    affinity = 0.5 * (affinity + affinity.T)  # Symmetrize
    print(f" [{time.time()-t0:.1f}s]")

    results = []
    for k in k_values:
        t1 = time.time()
        print(f"  Spectral (k={k})...", end='', flush=True)
        sc = SpectralClustering(
            n_clusters=k, affinity='precomputed',
            random_state=42, n_init=3, assign_labels='kmeans')
        labels = sc.fit_predict(affinity)
        print(f" [{time.time()-t1:.1f}s]")
        r = report_result(f"SBERT + Spectral (k={k})", labels, n_total,
                           gt_dir, time.time() - t0)
        r['k_param'] = k
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 9. SBERT + UMAP + KMeans
#    (What BERTopic does but with KMeans instead of HDBSCAN)
# ═══════════════════════════════════════════════════════════════

def run_sbert_umap_kmeans(embeddings, n_total, gt_dir,
                           umap_reduced=None,
                           k_values=[1000, 3000, 5000, 10000, 15000]):
    from sklearn.cluster import MiniBatchKMeans

    print(f"\n{'='*70}")
    print(f"BASELINE 9: SBERT + UMAP + KMeans")
    print(f"{'='*70}")

    t0 = time.time()
    if umap_reduced is None:
        try:
            import umap
            print(f"  UMAP reducing {embeddings.shape[1]}→50...", end='', flush=True)
            reducer = umap.UMAP(n_components=50, metric='cosine',
                                 n_neighbors=30, min_dist=0.0, random_state=42)
            umap_reduced = reducer.fit_transform(embeddings)
            print(f" [{time.time()-t0:.1f}s]")
        except ImportError:
            print(f"  No UMAP available, skipping")
            return []
    else:
        print(f"  Using pre-computed UMAP ({umap_reduced.shape})")

    results = []
    for k in k_values:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=5000,
                              n_init=3, max_iter=300)
        labels = km.fit_predict(umap_reduced)
        r = report_result(f"SBERT + UMAP + KMeans (k={k})", labels,
                           n_total, gt_dir, time.time() - t0)
        r['k_param'] = k
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 10. Spherical KMeans on SBERT
#     (better for cosine similarity — normalized embeddings)
# ═══════════════════════════════════════════════════════════════

def run_spherical_kmeans(embeddings, n_total, gt_dir,
                          k_values=[1000, 3000, 5000, 10000, 15000]):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.preprocessing import normalize

    print(f"\n{'='*70}")
    print(f"BASELINE 10: Spherical KMeans on SBERT")
    print(f"{'='*70}")

    t0 = time.time()
    # Spherical KMeans = normalize embeddings, run KMeans, repeat
    # We approximate with iterative normalize → KMeans
    X = normalize(embeddings, norm='l2')

    results = []
    for k in k_values:
        # Run 3 rounds of: KMeans → normalize centroids → reassign
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=5000,
                              n_init=3, max_iter=100)
        km.fit(X)
        # Normalize centroids and reassign
        for _ in range(3):
            centroids = normalize(km.cluster_centers_, norm='l2')
            # Cosine similarity = dot product for normalized vectors
            sims = X @ centroids.T
            labels = np.argmax(sims, axis=1)
            # Recompute centroids
            for c in range(k):
                mask = labels == c
                if mask.sum() > 0:
                    centroids[c] = normalize(X[mask].mean(axis=0).reshape(1, -1))[0]
            km.cluster_centers_ = centroids

        labels = np.argmax(X @ normalize(km.cluster_centers_, norm='l2').T, axis=1)
        r = report_result(f"Spherical KMeans (k={k})", labels, n_total,
                           gt_dir, time.time() - t0)
        r['k_param'] = k
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# 11. NMF Topic Model (Xu et al. 2003)
#     Non-negative Matrix Factorization — faster than LDA
# ═══════════════════════════════════════════════════════════════

def run_nmf(texts, n_total, gt_dir,
             n_topics_list=[100, 500, 1000, 2000, 3000]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    print(f"\n{'='*70}")
    print(f"BASELINE 11: NMF Topic Model")
    print(f"{'='*70}")

    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=20000, min_df=3, max_df=0.1,
        ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X = vectorizer.fit_transform(texts)
    print(f"  TF-IDF: {X.shape[0]:,} × {X.shape[1]:,} [{time.time()-t0:.1f}s]")

    results = []
    for n_topics in n_topics_list:
        t1 = time.time()
        print(f"  NMF (n_topics={n_topics})...", end='', flush=True)
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=200,
                   init='nndsvda')
        doc_topics = nmf.fit_transform(X)
        labels = np.argmax(doc_topics, axis=1)
        print(f" [{time.time()-t1:.1f}s]")
        r = report_result(f"NMF (n_topics={n_topics})", labels, n_total,
                           gt_dir, time.time() - t0)
        r['n_topics'] = n_topics
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Clustering Baselines')
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='results_baselines')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Skip: tfidf_km tfidf_hdb lda sbert_km '
                             'sbert_hdb bertopic sbert_leiden '
                             'spectral umap_km spherical nmf')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load data ───
    print("Loading data..."); t0 = time.time()
    with open(args.features_file) as f:
        data = json.load(f)
    n_total = len(data)
    print(f"  {n_total:,} items [{time.time()-t0:.1f}s]")

    texts = []
    for item in data:
        prompt = item.get('prompt', '')
        if not prompt:
            prompt = ' '.join(item.get('features', {}).get('concepts_all', []))
        texts.append(prompt[:2000])

    n_empty = sum(1 for t in texts if len(t.strip()) < 5)
    print(f"  Texts: {n_total:,} (empty/short: {n_empty:,})")

    # ─── Load precomputed SBERT embeddings ───
    embeddings = None
    emb_path = os.path.join(args.output_dir, 'sbert_embeddings.npy')
    if os.path.exists(emb_path):
        print(f"  Loading saved embeddings from {emb_path}")
        embeddings = np.load(emb_path)
        print(f"  Embeddings: {embeddings.shape}")
    else:
        print(f"  ⚠ No embeddings found at {emb_path}")
        print(f"    Run sbert_encode.py first!")

    all_results = []
    umap_reduced = None  # Cache UMAP result

    # ─── 1. TF-IDF + KMeans ───
    if 'tfidf_km' not in args.skip:
        try:
            r = run_tfidf_kmeans(texts, n_total, args.gt_dir)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ─── 2. TF-IDF + HDBSCAN ───
    if 'tfidf_hdb' not in args.skip:
        try:
            r = run_tfidf_hdbscan(texts, n_total, args.gt_dir)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ─── 3. LDA ───
    if 'lda' not in args.skip:
        try:
            r = run_lda(texts, n_total, args.gt_dir, n_topics_list=[100])
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    # ─── 4. SBERT + KMeans ───
    if 'sbert_km' not in args.skip and embeddings is not None:
        try:
            r = run_sbert_kmeans(embeddings, n_total, args.gt_dir)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}"); import traceback; traceback.print_exc()

    # ─── 5. SBERT + HDBSCAN ───
    if 'sbert_hdb' not in args.skip and embeddings is not None:
        try:
            r, umap_reduced = run_sbert_hdbscan(embeddings, n_total, args.gt_dir)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}"); import traceback; traceback.print_exc()

    # ─── 6. BERTopic ───
    if 'bertopic' not in args.skip and embeddings is not None:
        try:
            r = run_bertopic(texts, embeddings, n_total, args.gt_dir)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}"); import traceback; traceback.print_exc()

    # ─── 7. SBERT + Leiden ───
    if 'sbert_leiden' not in args.skip and embeddings is not None:
        try:
            r = run_sbert_leiden(embeddings, n_total, args.gt_dir)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}"); import traceback; traceback.print_exc()

    # ─── 8. SBERT + Spectral ───
    if 'spectral' not in args.skip and embeddings is not None:
        try:
            r = run_sbert_spectral(embeddings, n_total, args.gt_dir,
                                    k_values=[500, 1000, 2000])
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR Spectral: {e}"); import traceback; traceback.print_exc()

    # ─── 9. SBERT + UMAP + KMeans ───
    if 'umap_km' not in args.skip and embeddings is not None:
        try:
            r = run_sbert_umap_kmeans(embeddings, n_total, args.gt_dir,
                                       umap_reduced=umap_reduced)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}"); import traceback; traceback.print_exc()

    # ─── 10. Spherical KMeans ───
    if 'spherical' not in args.skip and embeddings is not None:
        try:
            r = run_spherical_kmeans(embeddings, n_total, args.gt_dir)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}"); import traceback; traceback.print_exc()

    # ─── 11. NMF ───
    if 'nmf' not in args.skip:
        try:
            r = run_nmf(texts, n_total, args.gt_dir)
            all_results.extend(r)
        except Exception as e:
            print(f"  ERROR: {e}"); import traceback; traceback.print_exc()

    # ═══════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════
    all_results.sort(key=lambda x: x['bal'], reverse=True)

    print(f"\n{'━'*120}")
    print(f"  COMPREHENSIVE BASELINE COMPARISON — {len(all_results)} configs")
    print(f"  Our best (Concept-kNN v5): bal=0.6239")
    print(f"{'━'*120}")
    print(f"  {'Method':<45} │ {'k':>7} {'Cov%':>6} │ "
          f"{'NMI_c':>6} {'NMI_m':>6} {'NMI_f':>6} │ {'Bal':>7}")
    print(f"  {'─'*45} │ {'─'*7} {'─'*6} │ "
          f"{'─'*6} {'─'*6} {'─'*6} │ {'─'*7}")

    for r in all_results:
        beat = ' ★' if r['bal'] >= 0.6239 else ''
        print(f"  {r['method']:<45} │ {r['k_clusters']:>7,} "
              f"{r['coverage']*100:>5.1f}% │ "
              f"{r.get('nmi_coarse',0):>6.4f} "
              f"{r.get('nmi_mid',0):>6.4f} "
              f"{r.get('nmi_fine',0):>6.4f} │ "
              f"{r['bal']:>7.4f}{beat}")

    # Best per family
    print(f"\n{'━'*120}")
    print(f"  BEST PER METHOD FAMILY")
    print(f"{'━'*120}")
    families = {}
    for r in all_results:
        name = r['method']
        for prefix, fam in [
            ('TF-IDF + KMeans', 'TF-IDF + KMeans'),
            ('TF-IDF + HDBSCAN', 'TF-IDF + HDBSCAN'),
            ('LDA', 'LDA'),
            ('NMF', 'NMF'),
            ('SBERT + KMeans', 'SBERT + KMeans'),
            ('SBERT + HDBSCAN', 'SBERT + HDBSCAN'),
            ('SBERT + UMAP + KMeans', 'SBERT + UMAP + KMeans'),
            ('BERTopic', 'BERTopic'),
            ('SBERT + Leiden', 'SBERT + Leiden'),
            ('SBERT + Spectral', 'SBERT + Spectral'),
            ('Spherical KMeans', 'Spherical KMeans'),
        ]:
            if name.startswith(prefix):
                if fam not in families or r['bal'] > families[fam]['bal']:
                    families[fam] = r
                break

    order = ['LDA', 'NMF', 'TF-IDF + KMeans', 'TF-IDF + HDBSCAN',
             'SBERT + KMeans', 'Spherical KMeans', 'SBERT + UMAP + KMeans',
             'SBERT + HDBSCAN', 'SBERT + Spectral', 'BERTopic',
             'SBERT + Leiden']
    for fam in order:
        if fam in families:
            r = families[fam]
            bar = '█' * int(r['bal'] * 50)
            delta = ((r['bal'] - 0.6239) / 0.6239) * 100
            print(f"  {fam:<25} bal={r['bal']:.4f} NMI_f={r.get('nmi_fine',0):.4f} "
                  f"cov={r['coverage']:.1%} ({delta:>+.1f}%) {bar}")

    print(f"  {'Concept-kNN v5 (OURS)':<25} bal=0.6239 NMI_f=0.6239 "
          f"cov=100.0% (  ref) {'█' * int(0.6239 * 50)}")

    # Save
    out_path = os.path.join(args.output_dir, 'baselines.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✓ Saved: {out_path}")


if __name__ == '__main__':
    main()