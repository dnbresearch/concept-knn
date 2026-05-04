#!/usr/bin/env python3
"""
Cascade Analysis — Track bal at each cascade round
===================================================
Runs Concept-kNN with best config and evaluates after:
  - Leiden (before any cascade)
  - Each cascade round (τ = 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)

Usage (ShareGPT):
  python cascade_analysis.py \
    --features-file features_v4.json \
    --gt-dir ground_truth/ \
    --output cascade_sharegpt.json \
    --resolution 80

Usage (LMSYS):
  python cascade_analysis.py \
    --features-file features_v4.json \
    --gt-dir ground_truth/ \
    --output cascade_lmsys.json \
    --resolution 50
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
# FILTERING (from concept_knn_v5.py — unchanged)
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
    if verbose: print(f"  Filtering ({n:,} items)...", end='', flush=True)
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
    if verbose: print(f" done [{time.time()-t0:.1f}s]")
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

def pmi_densify(data, concepts, c2i, top_k_associates=20,
                pmi_weight_decay=0.3, min_pmi=1.0, verbose=True):
    t0 = time.time(); n_q = len(data); n_c = len(concepts)
    if verbose: print(f"  PMI densification (top_k={top_k_associates})...", end='', flush=True)
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
# VECTOR BUILDING
# ═══════════════════════════════════════════════════════════════

def build_tfidf_vectors(data, max_features=5000, verbose=True):
    t0 = time.time()
    if verbose: print(f"  Building TF-IDF vectors...", end='', flush=True)
    texts = []
    for item in data:
        prompt = item.get('prompt', '')
        if not prompt: prompt = ' '.join(item['features'].get('concepts_all', []))
        texts.append(prompt[:2000])
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=3, max_df=0.08,
                                  ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    tfidf_norm = sk_normalize(tfidf_matrix, norm='l2', axis=1)
    if verbose: print(f" {tfidf_norm.shape[1]} features [{time.time()-t0:.1f}s]")
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
    if verbose: print(f"  Concept vectors: {n_q:,}×{n_c:,}")
    return QC, QC_norm, concepts, c2i

def build_hybrid_vectors(concept_norm, tfidf_norm, alpha=0.8, verbose=True):
    hybrid = hstack([concept_norm * alpha, tfidf_norm * (1.0 - alpha)])
    hybrid_norm = sk_normalize(hybrid, norm='l2', axis=1)
    if verbose: print(f"  Hybrid (α={alpha}): {hybrid_norm.shape[1]:,} dims")
    return hybrid_norm


# ═══════════════════════════════════════════════════════════════
# COMMUNITY DETECTION
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
    sizes = Counter(labels)
    coverage = sum(s for s in sizes.values() if s > 1) / n_total
    return {
        'n_clusters': len(sizes),
        'singletons': sum(1 for s in sizes.values() if s == 1),
        'coverage': round(coverage, 4),
    }


# ═══════════════════════════════════════════════════════════════
# CASCADE ROUND-BY-ROUND
# ═══════════════════════════════════════════════════════════════

def _compute_centroids(labels, X_norm, cluster_ids):
    cl = []
    for c in cluster_ids:
        m = np.where(labels == c)[0]
        if len(m) == 0: cl.append(np.zeros(X_norm.shape[1], dtype=np.float32))
        else: cl.append(np.asarray(X_norm[m].mean(axis=0)).flatten())
    centroids = np.array(cl, dtype=np.float32)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True); norms[norms==0] = 1
    return centroids / norms


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output', default='cascade_rounds.json')
    parser.add_argument('--pmi-top-k', type=int, default=35)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--sigma', type=float, default=0.6)
    parser.add_argument('--resolution', type=float, default=80.0)
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"Cascade Round-by-Round Analysis")
    print(f"{'='*70}")
    print(f"  Config: PMI={args.pmi_top_k}, α={args.alpha}, σ={args.sigma}, γ={args.resolution}")
    print(f"  Features: {args.features_file}")
    print(f"  GT: {args.gt_dir}")
    t0_all = time.time()

    # ── Load ──
    with open(args.features_file) as f: raw_data = json.load(f)
    n_total = len(raw_data)
    print(f"  {n_total:,} items\n")

    # ── Preprocess ──
    data, idf, high_idf = refilter_v5_relaxed(copy.deepcopy(raw_data))
    data = propagate_conversation_concepts(data, idf, high_idf)

    all_c = set()
    for item in data: all_c.update(item['features'].get('concepts_all', []))
    concepts = sorted(all_c); c2i = {c: i for i, c in enumerate(concepts)}

    if args.pmi_top_k > 0:
        data = pmi_densify(data, concepts, c2i, top_k_associates=args.pmi_top_k)

    # ── Build vectors ──
    QC, QC_norm, concepts, c2i = build_concept_vectors(data)
    tfidf_norm = build_tfidf_vectors(raw_data)

    if args.alpha < 1.0:
        X_norm = build_hybrid_vectors(QC_norm, tfidf_norm, alpha=args.alpha)
    else:
        X_norm = QC_norm

    # ── kNN graph ──
    print(f"\n  Building kNN graph...", end='', flush=True)
    t0 = time.time()
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
    graph = csr_matrix((np.array(vals, dtype=np.float32),
                        (np.array(rows, dtype=np.int32),
                         np.array(cols, dtype=np.int32))),
                       shape=(n_total, n_total))
    print(f" {graph.nnz:,} edges [{time.time()-t0:.1f}s]")

    # ── Leiden ──
    print(f"  Leiden (γ={args.resolution})...", end='', flush=True)
    t0 = time.time()
    base_labels = community_detect(graph, n_total, resolution=args.resolution)
    print(f" [{time.time()-t0:.1f}s]")

    # ══════════════════════════════════════════════════════════
    # Evaluate after Leiden (before cascade)
    # ══════════════════════════════════════════════════════════
    met = compute_metrics(base_labels, n_total)
    gt = evaluate_gt(base_labels, args.gt_dir)
    nf = gt.get('nmi_fine', 0); cv = met['coverage']
    bal = round(nf * cv, 4)

    rounds = []
    rounds.append({
        'stage': 'leiden',
        'threshold': None,
        'assigned_this_round': 0,
        'cumulative_assigned': 0,
        'singletons_remaining': met['singletons'],
        'n_clusters': met['n_clusters'],
        'coverage': met['coverage'],
        'nmi_coarse': gt.get('nmi_coarse', 0),
        'nmi_mid': gt.get('nmi_mid', 0),
        'nmi_fine': nf,
        'bal': bal,
    })

    hdr = (f"  {'Stage':<16} {'τ':>5} {'Assigned':>10} {'Cumul':>8} "
           f"{'Singletons':>12} {'Cov%':>7} {'NMI_c':>7} {'NMI_m':>7} "
           f"{'NMI_f':>7} {'Bal':>7}")
    sep = f"  {'─'*16} {'─'*5} {'─'*10} {'─'*8} {'─'*12} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}"
    print(f"\n{hdr}\n{sep}")
    print(f"  {'leiden':<16} {'—':>5} {'—':>10} {'—':>8} {met['singletons']:>12,} "
          f"{met['coverage']*100:>6.1f}% {gt.get('nmi_coarse',0):>7.4f} "
          f"{gt.get('nmi_mid',0):>7.4f} {nf:>7.4f} {bal:>7.4f}")

    # ══════════════════════════════════════════════════════════
    # Cascade rounds — evaluate after EACH
    # ══════════════════════════════════════════════════════════
    thresholds = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    nl = base_labels.copy()
    cumulative = 0

    for thr in thresholds:
        sizes = Counter(nl)
        sings = np.array([i for i in range(n_total) if sizes[nl[i]] == 1])
        non_s = [i for i in range(n_total) if sizes[nl[i]] > 1]

        if len(sings) == 0 or len(non_s) == 0:
            print(f"  {'cascade':<16} {thr:>5.1f} {'(no singletons — skip)':>30}")
            rounds.append({
                'stage': f'cascade', 'threshold': thr,
                'assigned_this_round': 0, 'cumulative_assigned': cumulative,
                'singletons_remaining': 0, 'n_clusters': len(sizes),
                'coverage': rounds[-1]['coverage'],
                'nmi_coarse': rounds[-1]['nmi_coarse'],
                'nmi_mid': rounds[-1]['nmi_mid'],
                'nmi_fine': rounds[-1]['nmi_fine'],
                'bal': rounds[-1]['bal'],
            })
            continue

        # Recompute centroids from CURRENT labels (key design choice)
        rc = sorted(set(nl[i] for i in non_s))
        centroids = _compute_centroids(nl, X_norm, rc)

        sv = X_norm[sings]
        if hasattr(sv, 'toarray'): sv = sv.toarray()
        sims = sv @ centroids.T

        nr = 0
        for i, idx in enumerate(sings):
            bp = np.argmax(sims[i])
            if sims[i, bp] >= thr:
                nl[idx] = rc[bp]; nr += 1

        cumulative += nr

        # Evaluate after this round
        met = compute_metrics(nl, n_total)
        gt = evaluate_gt(nl, args.gt_dir)
        nf = gt.get('nmi_fine', 0); cv = met['coverage']
        bal = round(nf * cv, 4)

        rounds.append({
            'stage': 'cascade',
            'threshold': thr,
            'assigned_this_round': nr,
            'cumulative_assigned': cumulative,
            'singletons_remaining': met['singletons'],
            'n_clusters': met['n_clusters'],
            'coverage': met['coverage'],
            'nmi_coarse': gt.get('nmi_coarse', 0),
            'nmi_mid': gt.get('nmi_mid', 0),
            'nmi_fine': nf,
            'bal': bal,
        })

        delta_bal = bal - rounds[-2]['bal']
        print(f"  {'cascade':<16} {thr:>5.1f} {nr:>10,} {cumulative:>8,} "
              f"{met['singletons']:>12,} {met['coverage']*100:>6.1f}% "
              f"{gt.get('nmi_coarse',0):>7.4f} {gt.get('nmi_mid',0):>7.4f} "
              f"{nf:>7.4f} {bal:>7.4f}  ({delta_bal:+.4f})")

    # ── Summary ──
    print(f"\n{'─'*80}")
    leiden_bal = rounds[0]['bal']
    final_bal = rounds[-1]['bal']
    print(f"  Leiden → Final:  {leiden_bal:.4f} → {final_bal:.4f}  "
          f"(Δ = {final_bal - leiden_bal:+.4f})")
    print(f"  Total singletons reassigned: {cumulative:,}")
    print(f"  Total time: {time.time()-t0_all:.0f}s")

    # ── Save ──
    output = {
        'config': {
            'features_file': args.features_file,
            'pmi_top_k': args.pmi_top_k,
            'alpha': args.alpha,
            'sigma': args.sigma,
            'resolution': args.resolution,
            'n_items': n_total,
        },
        'rounds': rounds,
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {args.output}")


if __name__ == '__main__':
    main()
