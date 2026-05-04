#!/usr/bin/env python3
"""
Concept kNN v2 — All Improvements Combined
=============================================

IMPROVEMENTS OVER v1 (best was NMI=0.526, cov=70.7%, bal=0.372):

  A. CONVERSATION PROPAGATION
     Follow-up prompts ("continue", "more details") have 0-1 concepts.
     Use conv_id to inherit concepts from richest prompt in conversation.
     Expected: +15-20% coverage

  B. SIGMA GAP FILL
     Big jump from σ=0.3 (NMI=0.31) to σ=0.5 (NMI=0.53).
     Fill with σ=0.35, 0.4, 0.45

  C. RESOLUTION SWEEP
     Louvain resolution=1.0 was never tuned.
     Higher resolution → more smaller clusters → better fine-grained NMI.

  D. TWO-PASS CLUSTERING
     Pass 1: Strict clustering (good NMI)
     Pass 2: Assign singletons to nearest cluster centroid (boosts coverage)

USAGE:
  python concept_knn_v2.py \\
    --features-file features_v4.json \\
    --gt-dir ground_truth/ \\
    --output-dir results_v2/
"""

import json, os, sys, time, gc, math, re
import numpy as np
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


# ═════════════════════════════════════════════════════════════════════
# V5-RELAXED CONCEPT FILTERING (same as concept_knn.py)
# ═════════════════════════════════════════════════════════════════════

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
    if len(c) < min_len or len(c.replace(' ', '')) < 2:
        return 'short'
    if c.replace(' ','').replace('.','').replace('-','').isdigit():
        return 'number'
    words = c.split()
    if len(words) == 1 and words[0] in COMMON_FIRST_NAMES:
        return 'person_name'
    if len(words) == 2 and words[0] in COMMON_FIRST_NAMES:
        if words[1] not in COMMON_FIRST_NAMES and len(words[1]) <= 10:
            if not any(words[1].endswith(s) for s in
                      ('ing','tion','ment','ness','ity','ics','ism')):
                return 'person_name'
    if c in LLM_FILLER:
        return 'llm_filler'
    for pat in LLM_FILLER_PATTERNS:
        if re.search(pat, c):
            return 'llm_filler'
    if c in SINGLE_WORD_NOISE:
        return 'noise'
    if not is_prompt and len(words) == 1 and words[0] in GEO_NOISE:
        return 'geo_noise'
    return None


def refilter_v5_relaxed(data, max_df_pct=0.08, min_df=3,
                         min_idf=0.5, max_concepts=30,
                         prompt_boost=3.0, verbose=True):
    t0 = time.time()
    n = len(data)
    if verbose:
        print(f"\n{'='*70}")
        print(f"V5-RELAXED FILTERING ({n:,} items)")
        print(f"{'='*70}")

    sample = data[0].get('features', {})
    has_split = 'concepts_prompt' in sample

    stats = Counter()
    concept_df = Counter()
    filtered = []

    for item in data:
        feat = item.get('features', {})
        if has_split:
            prompt_c = set(feat.get('concepts_prompt', []))
            response_c = set(feat.get('concepts_response', []))
        else:
            prompt_c = set(feat.get('concepts_all', []))
            response_c = set()

        kept_p, kept_r = set(), set()
        for c in prompt_c:
            r = filter_concept(c, is_prompt=True)
            if r: stats[r] += 1
            else: kept_p.add(c)
        for c in response_c:
            r = filter_concept(c, is_prompt=False)
            if r: stats[r] += 1
            else: kept_r.add(c)

        kept_all = kept_p | kept_r
        for c in kept_all:
            concept_df[c] += 1
        filtered.append((kept_p, kept_r, kept_all))

    max_df = int(n * max_df_pct)
    surviving = {c for c, df in concept_df.items()
                 if min_df <= df <= max_df}

    idf = {c: math.log(n / concept_df[c]) for c in surviving}
    high_idf = {c for c, s in idf.items() if s >= min_idf}

    for idx, item in enumerate(data):
        kept_p, kept_r, _ = filtered[idx]
        prompt_c = kept_p & high_idf
        response_c = kept_r & high_idf
        all_c = prompt_c | response_c

        scores = {}
        for c in all_c:
            boost = prompt_boost if c in prompt_c else 1.0
            scores[c] = idf.get(c, 0) * boost

        if len(scores) > max_concepts:
            top = sorted(scores.items(), key=lambda x: x[1],
                         reverse=True)[:max_concepts]
            scores = dict(top)
            all_c = set(scores.keys())

        total = sum(scores.values())
        weights = {c: round(s/total, 6) for c, s in scores.items()} \
                  if total > 0 else {}

        item['features']['concepts_all'] = sorted(all_c)
        item['features']['concept_weights'] = weights

    cpc = [len(item['features']['concepts_all']) for item in data]
    n_empty = sum(1 for c in cpc if c == 0)

    if verbose:
        print(f"  Concepts: {len(high_idf):,}, per query: "
              f"mean={np.mean(cpc):.1f}, med={np.median(cpc):.0f}")
        print(f"  Empty: {n_empty:,} ({n_empty/n*100:.1f}%)")
        print(f"  [{time.time()-t0:.1f}s]")

    return data, idf, high_idf


# ═════════════════════════════════════════════════════════════════════
# IMPROVEMENT A: CONVERSATION PROPAGATION
# ═════════════════════════════════════════════════════════════════════

def propagate_conversation_concepts(data, idf, high_idf,
                                     inherit_weight=0.5,
                                     min_concepts_to_receive=3,
                                     max_concepts=30,
                                     verbose=True):
    """
    For prompts with few concepts, inherit from richest prompt
    in same conversation. Follow-ups like "continue", "more details"
    get topic signal from their parent prompts.
    """
    t0 = time.time()
    n = len(data)

    # Build conversation index: conv_id → [indices]
    conv_map = defaultdict(list)
    for idx, item in enumerate(data):
        cid = item.get('conv_id')
        if cid:
            conv_map[cid].append(idx)

    n_convs = len(conv_map)
    conv_sizes = [len(v) for v in conv_map.values()]
    multi_turn = sum(1 for s in conv_sizes if s > 1)

    if verbose:
        print(f"\n{'='*70}")
        print(f"CONVERSATION PROPAGATION")
        print(f"{'='*70}")
        print(f"  Conversations: {n_convs:,} "
              f"(multi-turn: {multi_turn:,})")
        print(f"  Conv sizes: mean={np.mean(conv_sizes):.1f}, "
              f"max={max(conv_sizes)}")

    # Find concept-poor prompts
    poor_before = sum(1 for item in data
                      if len(item['features'].get('concepts_all', []))
                      < min_concepts_to_receive)

    n_propagated = 0
    n_concepts_added = 0

    for cid, indices in conv_map.items():
        if len(indices) < 2:
            continue

        # Find richest prompt in conversation (concept donor)
        richest_idx = max(indices,
            key=lambda i: len(data[i]['features'].get('concepts_all', [])))
        donor = data[richest_idx]['features']
        donor_concepts = set(donor.get('concepts_all', []))
        donor_weights = donor.get('concept_weights', {})

        if len(donor_concepts) < 3:
            continue  # Donor too poor

        # Propagate to concept-poor members
        for idx in indices:
            if idx == richest_idx:
                continue
            item = data[idx]
            current = set(item['features'].get('concepts_all', []))
            current_weights = dict(item['features'].get(
                'concept_weights', {}))

            if len(current) >= min_concepts_to_receive:
                continue  # Already has enough

            # Inherit missing concepts at reduced weight
            new_concepts = donor_concepts - current
            if not new_concepts:
                continue

            for c in new_concepts:
                w = donor_weights.get(c, 0.05) * inherit_weight
                current_weights[c] = round(w, 6)

            current = current | new_concepts

            # Cap at max
            if len(current) > max_concepts:
                top = sorted(current_weights.items(),
                             key=lambda x: x[1], reverse=True
                             )[:max_concepts]
                current_weights = dict(top)
                current = set(current_weights.keys())

            # Renormalize
            total = sum(current_weights.values())
            if total > 0:
                current_weights = {c: round(w/total, 6)
                                   for c, w in current_weights.items()}

            item['features']['concepts_all'] = sorted(current)
            item['features']['concept_weights'] = current_weights
            n_propagated += 1
            n_concepts_added += len(new_concepts)

    poor_after = sum(1 for item in data
                     if len(item['features'].get('concepts_all', []))
                     < min_concepts_to_receive)
    cpc = [len(item['features']['concepts_all']) for item in data]

    if verbose:
        print(f"  Propagated to {n_propagated:,} prompts "
              f"({n_concepts_added:,} concepts added)")
        print(f"  Concept-poor (<{min_concepts_to_receive}): "
              f"{poor_before:,} → {poor_after:,} "
              f"({poor_before - poor_after:,} fixed)")
        print(f"  Concepts/query: mean={np.mean(cpc):.1f}, "
              f"med={np.median(cpc):.0f}")
        print(f"  [{time.time()-t0:.1f}s]")

    return data


# ═════════════════════════════════════════════════════════════════════
# CORE: BUILD CONCEPT VECTORS
# ═════════════════════════════════════════════════════════════════════

def build_concept_vectors(data, verbose=True):
    t0 = time.time()
    n_q = len(data)
    all_concepts = set()
    for item in data:
        if 'features' in item:
            all_concepts.update(item['features'].get('concepts_all', []))

    concepts = sorted(all_concepts)
    c2i = {c: i for i, c in enumerate(concepts)}
    n_c = len(concepts)

    rows, cols, vals = [], [], []
    for qi, item in enumerate(data):
        if 'features' not in item:
            continue
        weights = item['features'].get('concept_weights', {})
        for c in item['features'].get('concepts_all', []):
            ci = c2i.get(c)
            if ci is not None:
                rows.append(qi)
                cols.append(ci)
                vals.append(weights.get(c, 0.1))

    QC = csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows), np.array(cols))),
        shape=(n_q, n_c))
    QC_norm = sk_normalize(QC, norm='l2', axis=1)

    if verbose:
        nnz = np.diff(QC.indptr)
        n_empty = int(np.sum(nnz == 0))
        print(f"\n  Concept vectors: {n_q:,} × {n_c:,}")
        print(f"    Per query: mean={nnz.mean():.1f}, "
              f"med={np.median(nnz):.0f}, max={nnz.max()}")
        print(f"    Empty: {n_empty:,} ({n_empty/n_q*100:.1f}%)")

    return QC, QC_norm, concepts, c2i


# ═════════════════════════════════════════════════════════════════════
# CORE: kNN GRAPH + COMMUNITY DETECTION
# ═════════════════════════════════════════════════════════════════════

def build_knn_graph(X_norm, k, sigma=0.0, verbose=True):
    """Direct kNN graph — never builds full sim matrix."""
    t0 = time.time()
    n = X_norm.shape[0]
    if verbose:
        print(f"    kNN (k={k}, σ={sigma})...", end='', flush=True)

    nn = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)

    rows, cols, vals = [], [], []
    for i in range(n):
        for j_pos in range(1, distances.shape[1]):
            j = indices[i, j_pos]
            sim = 1.0 - distances[i, j_pos]
            if sim >= sigma and j != i:
                a, b = min(i, j), max(i, j)
                rows.append(a); cols.append(b); vals.append(sim)

    graph = csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n, n))

    sym = graph + graph.T
    deg = np.diff(sym.tocsr().indptr)
    iso = int(np.sum(deg == 0))
    if verbose:
        print(f" {graph.nnz:,} edges, iso={iso:,} [{time.time()-t0:.1f}s]")
    return graph, iso


def community_detect(sim_upper, n, resolution=1.0):
    if sim_upper.nnz == 0:
        return np.arange(n, dtype=int)
    sym = sim_upper + sim_upper.T
    try:
        import igraph as ig
        import leidenalg
        g = ig.Graph.Weighted_Adjacency(sym, mode='undirected')
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=resolution,
            seed=42, n_iterations=3)
        return np.array(part.membership)
    except ImportError:
        pass
    import networkx as nx
    import networkx.algorithms.community as nx_comm
    G = nx.from_scipy_sparse_array(sym, edge_attribute='weight')
    comms = nx_comm.louvain_communities(
        G, weight='weight', resolution=resolution, seed=42)
    labels = np.zeros(n, dtype=int)
    for cid, members in enumerate(comms):
        for m in members:
            labels[m] = cid
    return labels


# ═════════════════════════════════════════════════════════════════════
# IMPROVEMENT D: TWO-PASS SINGLETON ASSIGNMENT
# ═════════════════════════════════════════════════════════════════════

def assign_singletons(labels, QC_norm, min_sim=0.15, verbose=True):
    """
    Pass 2: Assign singletons to nearest cluster centroid.
    Only assigns if cosine similarity > min_sim.
    """
    t0 = time.time()
    n = len(labels)

    sizes = Counter(labels)
    singletons = [i for i in range(n) if sizes[labels[i]] == 1]
    non_singletons = [i for i in range(n) if sizes[labels[i]] > 1]

    if not singletons or not non_singletons:
        if verbose:
            print(f"    Pass 2: nothing to assign")
        return labels

    # Compute cluster centroids (mean of member vectors)
    real_clusters = sorted(set(labels[i] for i in non_singletons))
    cl_map = {cl: idx for idx, cl in enumerate(real_clusters)}
    n_cl = len(real_clusters)

    # Build centroid matrix
    centroid_rows, centroid_vals = [], []
    for cl in real_clusters:
        members = [i for i in range(n) if labels[i] == cl]
        centroid = QC_norm[members].mean(axis=0)
        centroid_rows.append(np.asarray(centroid).flatten())

    centroids = np.array(centroid_rows, dtype=np.float32)
    # Normalize centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms == 0] = 1
    centroids = centroids / norms

    # For each singleton, find nearest centroid
    singleton_vecs = QC_norm[singletons]
    if hasattr(singleton_vecs, 'toarray'):
        singleton_vecs = singleton_vecs.toarray()

    # Cosine similarities: singleton_vecs @ centroids.T
    sims = singleton_vecs @ centroids.T  # (n_sing, n_cl)

    new_labels = labels.copy()
    n_assigned = 0
    for i, sing_idx in enumerate(singletons):
        best_cl_pos = np.argmax(sims[i])
        best_sim = sims[i, best_cl_pos]
        if best_sim >= min_sim:
            new_labels[sing_idx] = real_clusters[best_cl_pos]
            n_assigned += 1

    if verbose:
        print(f"    Pass 2: {n_assigned:,}/{len(singletons):,} "
              f"singletons assigned (min_sim={min_sim}) "
              f"[{time.time()-t0:.1f}s]")

    return new_labels


# ═════════════════════════════════════════════════════════════════════
# GT EVALUATION
# ═════════════════════════════════════════════════════════════════════

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
                            pred.append(int(labels[qi]))
                            true.append(enc[t])
                    if len(pred) >= 10:
                        results[f'nmi_{gran}'] = round(
                            normalized_mutual_info_score(
                                np.array(true), np.array(pred)), 4)
                        results[f'ari_{gran}'] = round(
                            adjusted_rand_score(
                                np.array(true), np.array(pred)), 4)
                except Exception as e:
                    print(f"    GT err ({gran}): {e}")
                break
    return results


# ═════════════════════════════════════════════════════════════════════
# METRICS + TABLE
# ═════════════════════════════════════════════════════════════════════

def compute_metrics(labels, n_total):
    sizes = Counter(labels)
    sa = np.array(sorted(sizes.values(), reverse=True))
    k_out = len(sizes)
    n_real = sum(1 for s in sa if s > 1)
    singletons = sum(1 for s in sa if s == 1)
    coverage = sum(s for s in sa if s > 1) / n_total
    max_pct = sa[0] / n_total
    probs = sa / n_total
    eff_k = np.exp(-np.sum(probs * np.log(probs + 1e-12)))
    return {
        'k_query': k_out, 'n_real': n_real,
        'coverage': round(coverage, 4),
        'max_pct': round(max_pct, 4),
        'effective_k': round(eff_k, 1),
        'singletons': singletons,
    }


def print_table(results):
    if not results:
        return
    has = {g: any(f'nmi_{g}' in r for r in results)
           for g in ['coarse','mid','fine']}
    bests = {}
    for g in ['coarse','mid','fine']:
        vals = [r.get(f'nmi_{g}', 0) for r in results
                if isinstance(r.get(f'nmi_{g}', 0), (int, float))]
        bests[g] = max(vals) if vals else 0

    best_bal = max((r.get('nmi_fine',0)*r.get('coverage',0)
                    for r in results), default=0)

    print(f"\n{'━'*150}")
    print(f"  RESULTS ({len(results)} runs)  "
          f"[prev best: NMI=0.526, cov=70.7%, bal=0.372]")
    print(f"{'━'*150}")

    hdr = (f"  {'#':>3} {'Mode':<22} {'σ':>4} {'kNN':>3} "
           f"{'Res':>4} {'P2':>3} │ {'k_q':>7} {'Real':>5} "
           f"{'Max%':>5} {'Cov%':>5} {'Eff_k':>6}")
    sep = (f"  {'─'*3} {'─'*22} {'─'*4} {'─'*3} "
           f"{'─'*4} {'─'*3} │ {'─'*7} {'─'*5} "
           f"{'─'*5} {'─'*5} {'─'*6}")
    for g, tag in [('coarse','NMI_c'),('mid','NMI_m'),('fine','NMI_f')]:
        if has[g]:
            hdr += f" │ {tag:>6}"
            sep += f" │ {'─'*6}"
    hdr += f" │ {'Bal':>7} {'Time':>5}"
    sep += f" │ {'─'*7} {'─'*5}"
    print(hdr); print(sep)

    for r in results:
        nf = r.get('nmi_fine', 0)
        cv = r.get('coverage', 0)
        bal = nf * cv if isinstance(nf, (int, float)) else 0

        line = (f"  {r['run']:>3} {r.get('mode',''):<22} "
                f"{r['sigma_q']:>4.2f} {r.get('query_knn',''):>3} "
                f"{r.get('resolution',1.0):>4.1f} "
                f"{'Y' if r.get('pass2') else 'N':>3} │ "
                f"{r['k_query']:>7,} {r.get('n_real',0):>5,} "
                f"{r.get('max_pct',0)*100:>4.1f}% "
                f"{cv*100:>4.1f}% "
                f"{r.get('effective_k',0):>5.0f}")
        for g in ['coarse','mid','fine']:
            if has[g]:
                nmi = r.get(f'nmi_{g}','')
                if nmi == '':
                    line += f" │ {'':>6}"
                else:
                    star = '★' if isinstance(nmi,(int,float)) \
                           and abs(nmi-bests[g])<0.0001 else ' '
                    line += f" │ {nmi:>5.4f}{star}"
        star_b = '★' if abs(bal - best_bal) < 0.0001 else ' '
        line += f" │ {bal:>6.4f}{star_b} {r.get('total_time',0):>4.0f}s"
        print(line)
    print(f"{'━'*150}\n")


# ═════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Concept kNN v2 — All Improvements')
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--skip-refilter', action='store_true')
    parser.add_argument('--skip-propagation', action='store_true')

    # V5-relaxed params
    parser.add_argument('--max-df-pct', type=float, default=0.08)
    parser.add_argument('--min-idf', type=float, default=0.5)
    parser.add_argument('--max-concepts', type=int, default=30)

    # Sweep params
    parser.add_argument('--sigma-q', nargs='+', type=float,
                        default=[0.35, 0.4, 0.45, 0.5])
    parser.add_argument('--query-knn', nargs='+', type=int,
                        default=[30])
    parser.add_argument('--resolutions', nargs='+', type=float,
                        default=[1.0, 2.0, 5.0])
    parser.add_argument('--pass2-sims', nargs='+', type=float,
                        default=[0.0, 0.15],
                        help='0=no pass2, >0=assign singletons above this sim')

    # Eval
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='results_v2')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load ───
    print("Loading features...")
    t0 = time.time()
    with open(args.features_file) as f:
        data = json.load(f)
    print(f"  {len(data):,} items [{time.time()-t0:.1f}s]")

    # ─── Refilter ───
    idf, high_idf = {}, set()
    if not args.skip_refilter:
        data, idf, high_idf = refilter_v5_relaxed(
            data, max_df_pct=args.max_df_pct,
            min_idf=args.min_idf,
            max_concepts=args.max_concepts)

    # ─── Improvement A: Conv propagation ───
    if not args.skip_propagation:
        data = propagate_conversation_concepts(
            data, idf, high_idf)

    # ─── Build vectors ───
    QC, QC_norm, concepts, c2i = build_concept_vectors(data)

    # ─── Sweep ───
    n_total = len(data)
    n_cfg = (len(args.sigma_q) * len(args.query_knn) *
             len(args.resolutions) * len(args.pass2_sims))
    print(f"\n{'#'*70}")
    print(f"SWEEP: {n_cfg} configurations")
    print(f"  σ: {args.sigma_q}")
    print(f"  kNN: {args.query_knn}")
    print(f"  resolutions: {args.resolutions}")
    print(f"  pass2: {args.pass2_sims}")
    print(f"{'#'*70}")

    results = []
    run_num = 0
    knn_cache = {}  # cache kNN graph per (knn, sigma)

    for knn in args.query_knn:
        for sigma in args.sigma_q:
            # Build kNN graph once per (knn, sigma)
            cache_key = (knn, sigma)
            if cache_key not in knn_cache:
                graph, n_iso = build_knn_graph(
                    QC_norm, k=knn, sigma=sigma)
                knn_cache[cache_key] = graph
            graph = knn_cache[cache_key]

            for res in args.resolutions:
                # Cluster
                t_run = time.time()
                print(f"\n  ▶ σ={sigma}, kNN={knn}, "
                      f"res={res}")
                labels = community_detect(graph, n_total,
                                          resolution=res)

                for p2_sim in args.pass2_sims:
                    run_num += 1
                    do_pass2 = p2_sim > 0

                    if do_pass2:
                        final_labels = assign_singletons(
                            labels, QC_norm, min_sim=p2_sim)
                    else:
                        final_labels = labels.copy()

                    metrics = compute_metrics(final_labels, n_total)
                    gt_res = evaluate_gt(final_labels, args.gt_dir)
                    run_time = time.time() - t_run

                    nf = gt_res.get('nmi_fine', 0)
                    cv = metrics['coverage']
                    bal = nf * cv

                    mode = f"conv+knn"
                    if do_pass2:
                        mode += f"+p2({p2_sim})"
                    mode += f"_r{res}"

                    print(f"    Run {run_num}: {mode} → "
                          f"NMI_f={nf:.4f}, cov={cv:.1%}, "
                          f"bal={bal:.4f}")

                    summary = {
                        'run': run_num, 'mode': mode,
                        'sigma_q': sigma, 'query_knn': knn,
                        'resolution': res,
                        'pass2': do_pass2,
                        'pass2_sim': p2_sim if do_pass2 else 0,
                        'total_time': round(run_time, 1),
                        **metrics, **gt_res,
                    }
                    results.append(summary)

                    if run_num % 6 == 0 or run_num == n_cfg:
                        print_table(results)

    # ─── Final ───
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print_table(results)

    if results:
        best = max(results,
                   key=lambda r: r.get('nmi_fine',0)*r.get('coverage',0))
        nf = best.get('nmi_fine', 0)
        cv = best.get('coverage', 0)
        bal = nf * cv

        print(f"\n  BEST: NMI_f={nf:.4f} × cov={cv:.1%} = {bal:.4f}")
        print(f"    σ={best['sigma_q']}, kNN={best.get('query_knn','')}, "
              f"res={best.get('resolution',1.0)}, "
              f"pass2={'Y' if best.get('pass2') else 'N'}")

        print(f"\n  vs PREVIOUS BEST:")
        print(f"    Old: NMI=0.526 × cov=70.7% = 0.372")
        print(f"    New: NMI={nf:.3f} × cov={cv:.1%} = {bal:.3f}")
        delta = ((bal - 0.372) / 0.372 * 100)
        print(f"    Δ = {'+' if delta > 0 else ''}{delta:.1f}%")

    sweep_file = os.path.join(args.output_dir, 'sweep_v2.json')
    with open(sweep_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved: {sweep_file}")


if __name__ == '__main__':
    main()
