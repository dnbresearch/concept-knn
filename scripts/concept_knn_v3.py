#!/usr/bin/env python3
"""
Concept kNN v5 — Hybrid Vectors + Ensemble Clustering
======================================================
v4 best: NMI=0.619 × cov=100% = 0.619 (PMI=20, σ=0.7, res=30, cascade5)

THREE NEW IDEAS:

1. HYBRID CONCEPT + TF-IDF VECTORS
   Concept vectors capture semantic topics but are sparse (8-25 dims).
   TF-IDF vectors capture lexical signal from raw prompt text.
   Blend: hybrid = [α·concept, (1-α)·tfidf] then L2-normalize.
   Different error modes → ensemble gain.

2. HIGHER PMI (25, 30)
   PMI=20 was the highest tested and won. Trend was still rising.
   May be more densification to extract.

3. ENSEMBLE CLUSTERING (co-association on kNN)
   Run M configs → M label sets.
   For each kNN pair (i,j): agreement = #(same cluster) / M
   Cluster the agreement-weighted graph.
   Averages out noise from any single config.

USAGE:
  python concept_knn_v5.py \\
    --features-file features_v4.json \\
    --gt-dir ground_truth/ \\
    --output-dir results_v5/
"""

import json, os, sys, time, gc, math, re, copy
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score


# ═══════════════════════════════════════════════════════════════
# FILTERING + PROPAGATION (from v4, unchanged)
# ═══════════════════════════════════════════════════════════════

COMMON_FIRST_NAMES = {
    'james','john','robert','michael','david','william','richard',
    'joseph','thomas','charles','christopher','daniel','matthew',
    'anthony','mark','donald','steven','paul','andrew','joshua',
    'kenneth','kevin','brian','george','timothy','ronald','edward',
    'jason','jeffrey','ryan','jacob','gary','nicholas','eric',
    'jonathan','stephen','larry','justin','scott','brandon','benjamin',
    'samuel','raymond','gregory','frank','alexander','patrick','jack',
    'dennis','jerry','tyler','aaron','jose','adam','nathan','henry',
    'peter','zachary','douglas','harold','kyle','noah','carl',
    'arthur','gerald','roger','keith','jeremy','terry','lawrence',
    'sean','christian','albert','jesse','ralph','roy','eugene',
    'randy','philip','harry','vincent','bobby','dylan','billy',
    'bruce','willie','jordan','dave','mike','bob','tom','alex',
    'max','luke','jake','ethan','logan','mason','liam','owen',
    'leo','eli','oscar','sam','ben','charlie','oliver','finn',
    'marcus','victor','felix','kai','cole','blake','derek',
    'mary','patricia','jennifer','linda','barbara','elizabeth',
    'susan','jessica','sarah','karen','lisa','nancy','betty',
    'margaret','sandra','ashley','dorothy','kimberly','emily',
    'donna','michelle','carol','amanda','melissa','deborah',
    'stephanie','rebecca','sharon','laura','cynthia','kathleen',
    'amy','angela','shirley','anna','brenda','pamela','emma',
    'nicole','helen','samantha','katherine','christine','debra',
    'rachel','carolyn','janet','catherine','maria','heather',
    'diane','ruth','julie','olivia','joyce','virginia','victoria',
    'kelly','lauren','christina','joan','evelyn','judith',
    'megan','andrea','cheryl','hannah','jacqueline','martha',
    'gloria','teresa','ann','sara','madison','frances','kathryn',
    'janice','jean','abigail','alice','judy','sophia','grace',
    'denise','amber','doris','marilyn','danielle','beverly',
    'isabella','theresa','diana','natalie','brittany','charlotte',
    'marie','kayla','alexis','lori','lily','claire','elena',
    'maya','rosa','ivy','luna','aria','mia','zoe','chloe',
    'ruby','stella','hazel','nora','violet',
}
LLM_FILLER = {
    'good','great','nice','fine','excellent','wonderful','fantastic',
    'amazing','awesome','perfect','interesting','important',
    'consider','consider following','consider using',
    'further questions','further assistance','feel free',
    'detailed explanation','detailed answer','comprehensive guide',
    'correct answer','right answer','best answer',
    'happy help','glad help','hope helps','hope helpful',
    'let know','let me know','don hesitate',
    'first column','second column','third column',
    'first step','second step','third step','next step',
    'key point','key points','main point','main points',
    'key takeaway','key takeaways','main takeaway',
    'important note','important thing','keep mind',
    'real world','real life','day life',
    'wide range','broad range','large number',
    'various types','different types','many types',
    'long run','short term','long term',
    'best practice','best practices','common practice',
    'well known','widely used','commonly used',
    'high quality','low cost','high performance',
    'make sure','take look','take care',
    'play role','play important role','crucial role',
    'provide information','gain insight',
    'read','agree','finally','update','guide',
    'pro con','pros cons',
}
LLM_FILLER_PATTERNS = [
    r'^(here|there)\s+(is|are|was|were)\b',
    r'^(you|we|i)\s+(can|could|should|may|might)\b',
    r'\b(overall|basically|essentially|generally|typically)\b',
    r'^(please|kindly|simply)\s+',
    r'\b(straightforward|comprehensive|robust|scalable)\b',
]
SINGLE_WORD_NOISE = {
    'set','get','run','use','add','put','let','try',
    'new','old','big','top','end','way','day','lot',
    'bit','part','case','base','line','time','place',
    'work','call','need','help','show','make','take',
    'give','find','keep','tell','come','look','know',
    'turn','move','play','live','feel','hold','bring',
    'happen','start','begin','change','follow','include',
    'continue','provide','require','ensure',
    'value','number','level','type','form','kind',
    'area','point','fact','hand','side','head',
}
GEO_NOISE = {
    'united','kingdom','states','america','european',
    'american','british','english','french','german',
    'chinese','japanese','indian','african','asian',
    'western','eastern','northern','southern',
    'york','london','california','texas',
    'india','china','japan','canada','australia',
    'europe','brazil','russia','mexico','italy',
    'spain','germany','france','korea','africa',
    'global','worldwide','international','national',
    'city','country','region','continent',
}

def filter_concept(concept, is_prompt=True, min_len=3):
    c = concept.lower().strip()
    if len(c) < min_len or len(c.replace(' ', '')) < 2: return 'short'
    if c.replace(' ','').replace('.','').replace('-','').isdigit(): return 'number'
    words = c.split()
    if len(words) == 1 and words[0] in COMMON_FIRST_NAMES: return 'person_name'
    if len(words) == 2 and words[0] in COMMON_FIRST_NAMES:
        if words[1] not in COMMON_FIRST_NAMES and len(words[1]) <= 10:
            if not any(words[1].endswith(s) for s in ('ing','tion','ment','ness','ity','ics','ism')):
                return 'person_name'
    if c in LLM_FILLER: return 'llm_filler'
    for pat in LLM_FILLER_PATTERNS:
        if re.search(pat, c): return 'llm_filler'
    if c in SINGLE_WORD_NOISE: return 'noise'
    if not is_prompt and len(words) == 1 and words[0] in GEO_NOISE: return 'geo_noise'
    return None

def refilter_v5_relaxed(data, max_df_pct=0.08, min_df=3, min_idf=0.5,
                         max_concepts=30, prompt_boost=3.0, verbose=True):
    t0 = time.time(); n = len(data)
    if verbose: print(f"\n{'='*70}\nV5-RELAXED FILTERING ({n:,} items)\n{'='*70}")
    sample = data[0].get('features', {}); has_split = 'concepts_prompt' in sample
    concept_df = Counter(); filtered = []
    for item in data:
        feat = item.get('features', {})
        if has_split:
            prompt_c = set(feat.get('concepts_prompt', [])); response_c = set(feat.get('concepts_response', []))
        else:
            prompt_c = set(feat.get('concepts_all', [])); response_c = set()
        kept_p, kept_r = set(), set()
        for c in prompt_c:
            if not filter_concept(c, True): kept_p.add(c)
        for c in response_c:
            if not filter_concept(c, False): kept_r.add(c)
        kept_all = kept_p | kept_r
        for c in kept_all: concept_df[c] += 1
        filtered.append((kept_p, kept_r))
    max_df = int(n * max_df_pct)
    surviving = {c for c, df in concept_df.items() if min_df <= df <= max_df}
    idf = {c: math.log(n / concept_df[c]) for c in surviving}
    high_idf = {c for c, s in idf.items() if s >= min_idf}
    for idx, item in enumerate(data):
        kept_p, kept_r = filtered[idx]
        prompt_c = kept_p & high_idf; response_c = kept_r & high_idf; all_c = prompt_c | response_c
        scores = {}
        for c in all_c:
            scores[c] = idf.get(c, 0) * (prompt_boost if c in prompt_c else 1.0)
        if len(scores) > max_concepts:
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_concepts]
            scores = dict(top); all_c = set(scores.keys())
        total = sum(scores.values())
        weights = {c: round(s/total, 6) for c, s in scores.items()} if total > 0 else {}
        item['features']['concepts_all'] = sorted(all_c)
        item['features']['concept_weights'] = weights
    cpc = [len(item['features']['concepts_all']) for item in data]
    if verbose:
        print(f"  Concepts: {len(high_idf):,}, per query: mean={np.mean(cpc):.1f}")
        print(f"  [{time.time()-t0:.1f}s]")
    return data, idf, high_idf

def propagate_conversation_concepts(data, idf, high_idf, inherit_weight=0.5,
                                     min_concepts_to_receive=3, max_concepts=30, verbose=True):
    t0 = time.time()
    conv_map = defaultdict(list)
    for idx, item in enumerate(data):
        cid = item.get('conv_id')
        if cid: conv_map[cid].append(idx)
    n_prop = 0
    for cid, indices in conv_map.items():
        if len(indices) < 2: continue
        ri = max(indices, key=lambda i: len(data[i]['features'].get('concepts_all', [])))
        dc = set(data[ri]['features'].get('concepts_all', []))
        dw = data[ri]['features'].get('concept_weights', {})
        if len(dc) < 3: continue
        for idx in indices:
            if idx == ri: continue
            cur = set(data[idx]['features'].get('concepts_all', []))
            cw = dict(data[idx]['features'].get('concept_weights', {}))
            if len(cur) >= min_concepts_to_receive: continue
            nc = dc - cur
            if not nc: continue
            for c in nc: cw[c] = round(dw.get(c, 0.05) * inherit_weight, 6)
            cur = cur | nc
            if len(cur) > max_concepts:
                top = sorted(cw.items(), key=lambda x: x[1], reverse=True)[:max_concepts]
                cw = dict(top); cur = set(cw.keys())
            total = sum(cw.values())
            if total > 0: cw = {c: round(w/total, 6) for c, w in cw.items()}
            data[idx]['features']['concepts_all'] = sorted(cur)
            data[idx]['features']['concept_weights'] = cw
            n_prop += 1
    if verbose: print(f"  Conv propagation: {n_prop:,} [{time.time()-t0:.1f}s]")
    return data


# ═══════════════════════════════════════════════════════════════
# PMI DENSIFICATION (from v4)
# ═══════════════════════════════════════════════════════════════

def pmi_densify(data, concepts, c2i, top_k_associates=20,
                pmi_weight_decay=0.3, min_pmi=1.0, verbose=True):
    t0 = time.time(); n_q = len(data); n_c = len(concepts)
    if verbose: print(f"\n  PMI densification (top_k={top_k_associates})...", end='', flush=True)
    rows, cols = [], []
    for qi, item in enumerate(data):
        for c in item['features'].get('concepts_all', []):
            ci = c2i.get(c)
            if ci is not None: rows.append(qi); cols.append(ci)
    QC_bin = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_q, n_c))
    freq = np.array(QC_bin.sum(axis=0)).flatten().astype(np.float64)
    cooc = (QC_bin.T @ QC_bin).toarray().astype(np.float64)
    np.fill_diagonal(cooc, 0)
    freq[freq == 0] = 1
    expected = np.outer(freq, freq) / n_q; expected[expected == 0] = 1
    pmi_mat = np.log(cooc / expected + 1e-10)
    pmi_mat[cooc == 0] = -100
    associates = {}
    for ci in range(n_c):
        if freq[ci] < 3: continue
        candidates = np.where(pmi_mat[ci] >= min_pmi)[0]
        if len(candidates) == 0: continue
        scores = pmi_mat[ci, candidates]
        if len(candidates) > top_k_associates:
            top_idx = np.argpartition(scores, -top_k_associates)[-top_k_associates:]
            candidates = candidates[top_idx]; scores = scores[top_idx]
        associates[ci] = list(zip(candidates.tolist(), scores.tolist()))
    del cooc, pmi_mat, expected; gc.collect()

    for qi, item in enumerate(data):
        current = set(item['features'].get('concepts_all', []))
        cw = dict(item['features'].get('concept_weights', {}))
        if not current: continue
        new_c = {}
        for c in current:
            ci = c2i.get(c)
            if ci is None or ci not in associates: continue
            c_w = cw.get(c, 0.1)
            for a_ci, pmi_s in associates[ci]:
                a_name = concepts[a_ci]
                if a_name in current: continue
                nw = c_w * pmi_weight_decay * min(pmi_s / 3.0, 1.0)
                new_c[a_name] = max(new_c.get(a_name, 0), nw)
        if new_c:
            if len(new_c) > 20:
                top = sorted(new_c.items(), key=lambda x: x[1], reverse=True)[:20]
                new_c = dict(top)
            for c, w in new_c.items(): cw[c] = round(w, 6)
            current = current | set(new_c.keys())
        total = sum(cw.values())
        if total > 0: cw = {c: round(w/total, 6) for c, w in cw.items()}
        item['features']['concepts_all'] = sorted(current)
        item['features']['concept_weights'] = cw
    if verbose: print(f" done [{time.time()-t0:.1f}s]")
    return data


# ═══════════════════════════════════════════════════════════════
# V5 NEW #1: HYBRID CONCEPT + TF-IDF VECTORS
# ═══════════════════════════════════════════════════════════════

def build_tfidf_vectors(data, max_features=5000, verbose=True):
    """Build TF-IDF vectors from raw prompt text."""
    t0 = time.time()
    if verbose: print(f"\n  Building TF-IDF vectors (max_features={max_features})...", end='', flush=True)

    # Extract prompt text
    texts = []
    for item in data:
        prompt = item.get('prompt', '')
        if not prompt:
            prompt = ' '.join(item['features'].get('concepts_all', []))
        texts.append(prompt[:2000])  # Cap length

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=3,
        max_df=0.08,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english',
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_norm = sk_normalize(tfidf_matrix, norm='l2', axis=1)

    if verbose:
        nnz = np.diff(tfidf_matrix.indptr)
        print(f" {tfidf_norm.shape[1]:,} features, mean={nnz.mean():.1f}/q [{time.time()-t0:.1f}s]")
    return tfidf_norm


def build_concept_vectors(data, verbose=True):
    n_q = len(data); all_concepts = set()
    for item in data:
        if 'features' in item: all_concepts.update(item['features'].get('concepts_all', []))
    concepts = sorted(all_concepts); c2i = {c: i for i, c in enumerate(concepts)}; n_c = len(concepts)
    rows, cols, vals = [], [], []
    for qi, item in enumerate(data):
        if 'features' not in item: continue
        weights = item['features'].get('concept_weights', {})
        for c in item['features'].get('concepts_all', []):
            ci = c2i.get(c)
            if ci is not None: rows.append(qi); cols.append(ci); vals.append(weights.get(c, 0.1))
    QC = csr_matrix((np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))), shape=(n_q, n_c))
    QC_norm = sk_normalize(QC, norm='l2', axis=1)
    if verbose:
        nnz = np.diff(QC.indptr)
        print(f"  Concept vectors: {n_q:,}×{n_c:,}, mean={nnz.mean():.1f}/q")
    return QC, QC_norm, concepts, c2i


def build_hybrid_vectors(concept_norm, tfidf_norm, alpha=0.8, verbose=True):
    """
    Blend concept and TF-IDF vectors.
    alpha=1.0 → pure concept, alpha=0.0 → pure TF-IDF
    """
    t0 = time.time()
    hybrid = hstack([concept_norm * alpha, tfidf_norm * (1.0 - alpha)])
    hybrid_norm = sk_normalize(hybrid, norm='l2', axis=1)
    if verbose:
        print(f"  Hybrid (α={alpha}): {hybrid_norm.shape[1]:,} dims [{time.time()-t0:.1f}s]")
    return hybrid_norm


# ═══════════════════════════════════════════════════════════════
# GRAPH + CLUSTERING + PASS2
# ═══════════════════════════════════════════════════════════════

def community_detect(sim_upper, n, resolution=1.0):
    if sim_upper.nnz == 0: return np.arange(n, dtype=int)
    sym = sim_upper + sim_upper.T
    try:
        import igraph as ig; import leidenalg
        g = ig.Graph.Weighted_Adjacency(sym, mode='undirected')
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                         weights='weight', resolution_parameter=resolution, seed=42, n_iterations=3)
        return np.array(part.membership)
    except ImportError: pass
    import networkx as nx; import networkx.algorithms.community as nx_comm
    G = nx.from_scipy_sparse_array(sym, edge_attribute='weight')
    comms = nx_comm.louvain_communities(G, weight='weight', resolution=resolution, seed=42)
    labels = np.zeros(n, dtype=int)
    for cid, members in enumerate(comms):
        for m in members: labels[m] = cid
    return labels


def _compute_centroids(labels, X_norm, cluster_ids):
    cl = []
    for c in cluster_ids:
        m = np.where(labels == c)[0]
        if len(m) == 0: cl.append(np.zeros(X_norm.shape[1], dtype=np.float32))
        else: cl.append(np.asarray(X_norm[m].mean(axis=0)).flatten())
    centroids = np.array(cl, dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True); norms[norms==0] = 1
    return centroids / norms

def pass2_cascading(labels, X_norm, thresholds=[0.5,0.4,0.3,0.2,0.1,0.0], verbose=True):
    t0 = time.time(); n = len(labels); nl = labels.copy(); total = 0
    for thr in thresholds:
        sizes = Counter(nl)
        sings = np.array([i for i in range(n) if sizes[nl[i]] == 1])
        non_s = [i for i in range(n) if sizes[nl[i]] > 1]
        if len(sings) == 0 or len(non_s) == 0: break
        rc = sorted(set(nl[i] for i in non_s))
        centroids = _compute_centroids(nl, X_norm, rc)
        sv = X_norm[sings]
        if hasattr(sv, 'toarray'): sv = sv.toarray()
        sims = sv @ centroids.T
        nr = 0
        for i, idx in enumerate(sings):
            bp = np.argmax(sims[i])
            if sims[i, bp] >= thr: nl[idx] = rc[bp]; nr += 1
        total += nr
    if verbose: print(f"      cascade: {total:,} assigned [{time.time()-t0:.1f}s]")
    return nl


# ═══════════════════════════════════════════════════════════════
# V5 NEW #3: ENSEMBLE CLUSTERING
# ═══════════════════════════════════════════════════════════════

def ensemble_cluster(all_label_sets, knn_idx, knn_dist, n_total,
                     resolution=30.0, verbose=True):
    """
    Ensemble clustering via co-association on kNN graph.

    For each kNN pair (i,j), compute agreement = fraction of runs
    where labels[i]==labels[j]. Use as edge weight for final clustering.
    """
    t0 = time.time()
    M = len(all_label_sets)
    if verbose:
        print(f"\n  Ensemble clustering from {M} label sets...")

    # Build co-association weights for kNN edges
    rows, cols, vals = [], [], []
    for i in range(n_total):
        for j_pos in range(1, knn_idx.shape[1]):
            j = int(knn_idx[i, j_pos])
            if j <= i:
                continue  # Upper triangle only
            # Count agreement across all label sets
            agree = sum(1 for labels in all_label_sets if labels[i] == labels[j])
            if agree > 0:
                weight = agree / M
                rows.append(i); cols.append(j); vals.append(weight)

    coassoc = csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n_total, n_total))

    if verbose:
        print(f"    Co-association: {coassoc.nnz:,} edges, "
              f"mean weight={np.mean(vals):.3f}")

    # Cluster the co-association graph
    labels = community_detect(coassoc, n_total, resolution=resolution)
    if verbose:
        sizes = Counter(labels)
        n_sing = sum(1 for s in sizes.values() if s == 1)
        print(f"    Ensemble clusters: {len(sizes):,}, singletons={n_sing:,}")
        print(f"    [{time.time()-t0:.1f}s]")
    return labels


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_gt(labels, gt_dir):
    results = {}
    if not gt_dir or not os.path.isdir(gt_dir): return results
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(gt_dir)))
        from llm_ground_truth import load_ground_truth
    except ImportError:
        def load_ground_truth(path):
            with open(path) as f: gt = json.load(f)
            if 'prompt_labels' not in gt:
                pl, cats = {}, set()
                for item in gt.get('items', gt.get('data', [])):
                    qi = item.get('query_idx', item.get('idx'))
                    cat = item.get('category', item.get('label'))
                    if qi is not None and cat is not None: pl[qi] = cat; cats.add(cat)
                return {'prompt_labels': pl, 'n_categories': len(cats)}
            return gt
    for gran in ['coarse', 'mid', 'fine']:
        for fname in sorted(os.listdir(gt_dir)):
            if f'gt_{gran}_' in fname and fname.endswith('.json') and '_prompt_labels' not in fname:
                try:
                    gt = load_ground_truth(os.path.join(gt_dir, fname)); pl = gt['prompt_labels']
                    pred, true, enc = [], [], {}
                    for qi in range(len(labels)):
                        if qi in pl:
                            t = pl[qi]
                            if t not in enc: enc[t] = len(enc)
                            pred.append(int(labels[qi])); true.append(enc[t])
                    if len(pred) >= 10:
                        results[f'nmi_{gran}'] = round(normalized_mutual_info_score(
                            np.array(true), np.array(pred)), 4)
                except Exception as e: print(f"    GT err ({gran}): {e}")
                break
    return results

def compute_metrics(labels, n_total):
    sizes = Counter(labels); sa = np.array(sorted(sizes.values(), reverse=True))
    coverage = sum(s for s in sa if s > 1) / n_total
    return {
        'k_query': len(sizes), 'n_real': sum(1 for s in sa if s > 1),
        'coverage': round(coverage, 4), 'singletons': sum(1 for s in sa if s == 1),
    }


# ═══════════════════════════════════════════════════════════════
# PIPELINE: single config run
# ═══════════════════════════════════════════════════════════════

def run_single_config(X_norm, n_total, sigma, resolution, gt_dir,
                       label, verbose=True):
    """Build graph, cluster, cascade pass2. Return labels + metrics."""
    # Build graph from precomputed kNN
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=31, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(X_norm)
    knn_dist, knn_idx = nn.kneighbors(X_norm)

    rows, cols, vals = [], [], []
    for i in range(n_total):
        for j_pos in range(1, knn_idx.shape[1]):
            j = int(knn_idx[i, j_pos])
            sim = 1.0 - knn_dist[i, j_pos]
            if sim >= sigma and j != i:
                a, b = min(i, j), max(i, j)
                rows.append(a); cols.append(b); vals.append(sim)
    graph = csr_matrix((np.array(vals, dtype=np.float32),
                        (np.array(rows, dtype=np.int32),
                         np.array(cols, dtype=np.int32))),
                       shape=(n_total, n_total))

    base_labels = community_detect(graph, n_total, resolution=resolution)
    final_labels = pass2_cascading(base_labels, X_norm, verbose=verbose)

    met = compute_metrics(final_labels, n_total)
    gt = evaluate_gt(final_labels, gt_dir)
    nf = gt.get('nmi_fine', 0); cv = met['coverage']
    bal = round(nf * cv, 4)

    if verbose:
        print(f"    {label}: NMI_f={nf:.4f} cov={cv:.1%} bal={bal:.4f} [{time.time()-t0:.1f}s]")

    return final_labels, knn_dist, knn_idx, {**met, **gt, 'bal': bal}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='results_v5')
    parser.add_argument('--sigma-q', nargs='+', type=float, default=[0.7])
    parser.add_argument('--resolutions', nargs='+', type=float, default=[30.0])
    parser.add_argument('--pmi-top-k', nargs='+', type=int, default=[0, 20, 25, 30])
    parser.add_argument('--alphas', nargs='+', type=float, default=[1.0, 0.9, 0.8, 0.7])
    parser.add_argument('--ensemble-res', nargs='+', type=float, default=[10.0, 20.0, 30.0, 50.0])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load & preprocess ───
    print("Loading..."); t0_all = time.time()
    with open(args.features_file) as f: raw_data = json.load(f)
    n_total = len(raw_data)
    print(f"  {n_total:,} items")

    # Base processing
    base_data, idf, high_idf = refilter_v5_relaxed(copy.deepcopy(raw_data))
    base_data = propagate_conversation_concepts(base_data, idf, high_idf)

    # Build base concept vocabulary
    all_c = set()
    for item in base_data: all_c.update(item['features'].get('concepts_all', []))
    concepts_base = sorted(all_c); c2i_base = {c: i for i, c in enumerate(concepts_base)}

    # Build TF-IDF vectors ONCE (doesn't depend on PMI)
    tfidf_norm = build_tfidf_vectors(raw_data, max_features=5000)

    results = []
    run_num = 0
    ensemble_candidates = []  # Collect (labels, knn_idx, knn_dist) for ensemble

    # ═══════════════════════════════════════════════════
    # PART 1: Individual config sweep
    # ═══════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print(f"PART 1: INDIVIDUAL CONFIG SWEEP")
    n_cfg = len(args.pmi_top_k) * len(args.alphas) * len(args.sigma_q) * len(args.resolutions)
    print(f"  {n_cfg} configs: PMI={args.pmi_top_k} × α={args.alphas} × σ={args.sigma_q} × res={args.resolutions}")
    print(f"{'#'*70}")

    best_knn_dist = None
    best_knn_idx = None

    for pmi_k in args.pmi_top_k:
        pmi_label = f"pmi{pmi_k}" if pmi_k > 0 else "nopmi"
        print(f"\n  ── {pmi_label} ──")

        if pmi_k > 0:
            data_pmi = copy.deepcopy(base_data)
            data_pmi = pmi_densify(data_pmi, concepts_base, c2i_base,
                                    top_k_associates=pmi_k)
        else:
            data_pmi = base_data

        # Build concept vectors for this PMI level
        QC, QC_norm, concepts, c2i = build_concept_vectors(data_pmi, verbose=True)

        for alpha in args.alphas:
            # Build hybrid or pure concept vectors
            if alpha < 1.0:
                X_norm = build_hybrid_vectors(QC_norm, tfidf_norm, alpha=alpha)
                vec_label = f"hybrid_a{alpha}"
            else:
                X_norm = QC_norm
                vec_label = "concept"

            # Precompute kNN once for this vector variant
            print(f"\n    kNN for {pmi_label}/{vec_label}...", end='', flush=True)
            t0 = time.time()
            nn = NearestNeighbors(n_neighbors=31, metric='cosine',
                                   algorithm='brute', n_jobs=-1)
            nn.fit(X_norm)
            knn_dist, knn_idx = nn.kneighbors(X_norm)
            print(f" [{time.time()-t0:.1f}s]")

            for sigma in args.sigma_q:
                # Build graph from precomputed kNN
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
                iso = int(np.sum(np.diff((graph+graph.T).tocsr().indptr)==0))

                for res in args.resolutions:
                    run_num += 1
                    t_run = time.time()

                    base_labels = community_detect(graph, n_total, resolution=res)
                    final_labels = pass2_cascading(base_labels, X_norm, verbose=False)

                    met = compute_metrics(final_labels, n_total)
                    gt = evaluate_gt(final_labels, args.gt_dir)
                    nf = gt.get('nmi_fine', 0); cv = met['coverage']
                    bal = round(nf * cv, 4)
                    mark = ' ▲' if bal > 0.6189 else ''

                    print(f"    #{run_num:<3} {pmi_label:>6}/{vec_label:<12} "
                          f"σ={sigma} res={res:>4.0f} iso={iso:>6,} │ "
                          f"NMI_f={nf:.4f} cov={cv:.1%} bal={bal:.4f}{mark}")

                    results.append({
                        'run': run_num, 'pmi_k': pmi_k, 'alpha': alpha,
                        'sigma_q': sigma, 'resolution': res,
                        'pass2': 'cascade5', 'type': 'single',
                        **met, **gt, 'bal': bal,
                    })

                    # Collect for ensemble
                    ensemble_candidates.append({
                        'labels': final_labels.copy(),
                        'bal': bal,
                        'config': f"{pmi_label}/{vec_label}/σ={sigma}/res={res}",
                    })

                    # Track best kNN for ensemble graph
                    if best_knn_dist is None or bal > max(r['bal'] for r in results[:-1] if r.get('type')=='single'):
                        best_knn_dist = knn_dist.copy()
                        best_knn_idx = knn_idx.copy()

    # ═══════════════════════════════════════════════════
    # PART 2: ENSEMBLE CLUSTERING
    # ═══════════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print(f"PART 2: ENSEMBLE CLUSTERING")
    print(f"{'#'*70}")

    # Sort candidates by bal, take top-N for ensemble
    ensemble_candidates.sort(key=lambda x: x['bal'], reverse=True)

    for n_top in [3, 5, 8, len(ensemble_candidates)]:
        top_labels = [c['labels'] for c in ensemble_candidates[:n_top]]
        print(f"\n  Ensemble from top-{n_top} configs:")
        for c in ensemble_candidates[:min(n_top, 5)]:
            print(f"    {c['config']}: bal={c['bal']:.4f}")
        if n_top > 5:
            print(f"    ... and {n_top-5} more")

        ens_labels = ensemble_cluster(top_labels, best_knn_idx, best_knn_dist,
                                       n_total, resolution=30.0)

        # Apply cascade pass2 to ensemble result
        # Use concept vectors from best single config for pass2
        best_single = max(results, key=lambda r: r['bal'] if r.get('type')=='single' else 0)
        best_pmi = best_single['pmi_k']
        best_alpha = best_single['alpha']

        # Rebuild vectors for pass2 (use cached if possible)
        if best_pmi > 0:
            data_pass2 = copy.deepcopy(base_data)
            data_pass2 = pmi_densify(data_pass2, concepts_base, c2i_base,
                                      top_k_associates=best_pmi, verbose=False)
        else:
            data_pass2 = base_data
        _, QC_pass2, _, _ = build_concept_vectors(data_pass2, verbose=False)
        if best_alpha < 1.0:
            X_pass2 = build_hybrid_vectors(QC_pass2, tfidf_norm, alpha=best_alpha, verbose=False)
        else:
            X_pass2 = QC_pass2

        # Try different resolutions for ensemble
        for ens_res in args.ensemble_res:
            run_num += 1

            ens_labels_r = ensemble_cluster(top_labels, best_knn_idx, best_knn_dist,
                                             n_total, resolution=ens_res, verbose=False)
            ens_final = pass2_cascading(ens_labels_r, X_pass2, verbose=False)

            met = compute_metrics(ens_final, n_total)
            gt = evaluate_gt(ens_final, args.gt_dir)
            nf = gt.get('nmi_fine', 0); cv = met['coverage']
            bal = round(nf * cv, 4)
            mark = ' ▲' if bal > 0.6189 else ''

            print(f"    #{run_num:<3} ens-top{n_top}/res={ens_res:>4.0f} │ "
                  f"NMI_f={nf:.4f} cov={cv:.1%} sing={met['singletons']:>6,} "
                  f"bal={bal:.4f}{mark}")

            results.append({
                'run': run_num, 'type': 'ensemble',
                'ensemble_n': n_top, 'resolution': ens_res,
                'pass2': 'cascade5',
                **met, **gt, 'bal': bal,
            })

    # ═══════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════
    results.sort(key=lambda x: x['bal'], reverse=True)
    best = results[0]

    print(f"\n{'━'*120}")
    print(f"  V5 RESULTS — {len(results)} configs (v4 best: 0.6189)")
    print(f"{'━'*120}")
    print(f"  {'#':>3} {'Type':<12} {'PMI':>4} {'α':>4} {'σ':>4} {'Res':>4} │ "
          f"{'Cov%':>6} {'NMI_f':>6} │ {'Bal':>7}")
    print(f"  {'─'*3} {'─'*12} {'─'*4} {'─'*4} {'─'*4} {'─'*4} │ "
          f"{'─'*6} {'─'*6} │ {'─'*7}")
    for r in results[:30]:
        star = ' ★' if abs(r['bal'] - best['bal']) < 0.0001 else ''
        beat = ' ▲' if r['bal'] > 0.6189 else ''
        tp = r.get('type', '?')
        pmi = r.get('pmi_k', '-')
        alp = r.get('alpha', '-')
        sig = r.get('sigma_q', '-')
        print(f"  {r['run']:>3} {tp:<12} {pmi:>4} {alp:>4} {sig:>4} "
              f"{r['resolution']:>4.0f} │ {r['coverage']*100:>5.1f}% "
              f"{r.get('nmi_fine',0):>6.4f} │ {r['bal']:>7.4f}{star}{beat}")

    # Effect analysis
    print(f"\n  ── HYBRID α EFFECT (PMI=20, σ=0.7, res=30) ──")
    for alpha in args.alphas:
        m = [r for r in results if r.get('pmi_k')==20 and r.get('alpha')==alpha
             and r.get('type')=='single' and abs(r.get('sigma_q',0)-0.7)<0.01]
        if m:
            r = m[0]
            print(f"    α={alpha}: NMI_f={r.get('nmi_fine',0):.4f} bal={r['bal']:.4f}")

    print(f"\n  ── PMI EFFECT (α=1.0, σ=0.7, res=30) ──")
    for pmi in args.pmi_top_k:
        m = [r for r in results if r.get('pmi_k')==pmi and r.get('alpha')==1.0
             and r.get('type')=='single' and abs(r.get('sigma_q',0)-0.7)<0.01]
        if m:
            r = m[0]
            print(f"    PMI={pmi:>2}: NMI_f={r.get('nmi_fine',0):.4f} bal={r['bal']:.4f}")

    if best:
        print(f"\n{'═'*120}")
        print(f"  BEST: {best}")
        delta = (best['bal'] - 0.6189) / 0.6189 * 100
        print(f"  vs v4 (0.6189): {'+' if delta>0 else ''}{delta:.1f}%")

    out = os.path.join(args.output_dir, 'sweep_v5.json')
    with open(out, 'w') as f: json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved: {out}")
    print(f"  Total time: {time.time()-t0_all:.0f}s")


if __name__ == '__main__':
    main()