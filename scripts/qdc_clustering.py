#!/usr/bin/env python3
"""
Concept kNN Clustering — Best of Everything
=============================================

WHAT WE LEARNED:
  - TF-IDF alone: NMI=0.156 (terrible — clusters by syntax, not topic)
  - Concepts V5 strict: NMI=0.628 but coverage=50% (too few concepts)
  - Super-topics: NMI=0.361 (collapsed discriminative signal)
  - Full sim matrix: OOM on 161K queries

THIS SCRIPT:
  1. Loads V4 features (pre-V5-filtering)
  2. Applies V5-RELAXED filtering (keep good filters, relax thresholds)
  3. Builds concept vectors with IDF weights
  4. Uses sklearn NearestNeighbors for kNN graph (no full sim matrix)
  5. Leiden/Louvain clustering
  6. Evaluates against ground truth

KEY RELAXATIONS:
  V5-strict → V5-relaxed
    max_df:        3% → 8%       (keep broader concepts)
    min_idf:       1.5 → 0.5     (keep mid-frequency concepts)
    max_concepts:  15 → 30       (richer per-query vectors)
    min_df:        5 → 3         (keep slightly rarer concepts)

  Expected: concepts/query 7→14+, coverage 50%→75%+

USAGE:
  python concept_knn.py \\
    --features-file features_v4.json \\
    --sigma-q 0.1 0.15 0.2 0.3 \\
    --query-knn 30 50 \\
    --gt-dir ground_truth/

  # Or with already-filtered V5:
  python concept_knn.py \\
    --features-file features_v5.json \\
    --skip-refilter \\
    --sigma-q 0.1 0.15 0.2 0.3 \\
    --query-knn 30 50 \\
    --gt-dir ground_truth/
"""

import json, os, sys, time, gc, math, re
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional
from scipy.sparse import csr_matrix, hstack as sp_hstack
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import scipy.sparse as sp


# =============================================================================
# V5-RELAXED CONCEPT FILTERING
# =============================================================================

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
    'perfect choice','great choice','good choice',
    'quick summary','brief overview','short answer',
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
    """Returns None if concept is OK, or reason string if filtered."""
    c = concept.lower().strip()
    
    # Too short
    if len(c) < min_len or len(c.replace(' ', '')) < 2:
        return 'short'
    
    # Pure numbers
    if c.replace(' ','').replace('.','').replace('-','').isdigit():
        return 'number'
    
    # Person names
    words = c.split()
    if len(words) == 1 and words[0] in COMMON_FIRST_NAMES:
        return 'person_name'
    if len(words) == 2 and words[0] in COMMON_FIRST_NAMES:
        if words[1] not in COMMON_FIRST_NAMES and len(words[1]) <= 10:
            if not any(words[1].endswith(s) for s in
                      ('ing','tion','ment','ness','ity','ics','ism')):
                return 'person_name'
    
    # LLM filler
    if c in LLM_FILLER:
        return 'llm_filler'
    for pat in LLM_FILLER_PATTERNS:
        if re.search(pat, c):
            return 'llm_filler'
    
    # Single-word noise
    if c in SINGLE_WORD_NOISE:
        return 'noise'
    
    # Geographic noise (only from response)
    if not is_prompt and len(words) == 1 and words[0] in GEO_NOISE:
        return 'geo_noise'
    
    return None


def refilter_v5_relaxed(data, max_df_pct=0.08, min_df=3,
                         min_idf=0.5, max_concepts=30,
                         prompt_boost=3.0, verbose=True):
    """
    Apply V5-style filtering with RELAXED thresholds.
    
    Loads from V4 features (which have concepts_prompt, concepts_response).
    """
    t0 = time.time()
    n = len(data)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"V5-RELAXED CONCEPT FILTERING ({n:,} items)")
        print(f"  max_df={max_df_pct:.0%}, min_df={min_df}, "
              f"min_idf={min_idf}, max_concepts={max_concepts}")
        print(f"{'='*70}")
    
    # Check if V4-style features exist
    sample = data[0].get('features', {})
    has_split = 'concepts_prompt' in sample and 'concepts_response' in sample
    
    if not has_split:
        if verbose:
            print("  No prompt/response split — using concepts_all directly")
    
    # Step 1: Content filter
    if verbose:
        print(f"\n  Step 1: Content filtering...")
    
    stats = Counter()
    concept_df = Counter()
    filtered = []
    
    for item in data:
        feat = item.get('features', {})
        
        if has_split:
            prompt_c = set(feat.get('concepts_prompt', []))
            response_c = set(feat.get('concepts_response', []))
        else:
            all_c = set(feat.get('concepts_all', []))
            prompt_c = all_c  # treat all as prompt-origin
            response_c = set()
        
        kept_p, kept_r = set(), set()
        
        for c in prompt_c:
            reason = filter_concept(c, is_prompt=True)
            if reason:
                stats[reason] += 1
            else:
                kept_p.add(c)
        
        for c in response_c:
            reason = filter_concept(c, is_prompt=False)
            if reason:
                stats[reason] += 1
            else:
                kept_r.add(c)
        
        kept_all = kept_p | kept_r
        for c in kept_all:
            concept_df[c] += 1
        
        filtered.append((kept_p, kept_r, kept_all))
    
    if verbose:
        print(f"    Unique concepts: {len(concept_df):,}")
        for reason, cnt in stats.most_common(8):
            print(f"      {reason:20s} {cnt:,}")
    
    # Step 2: Document frequency filter
    max_df = int(n * max_df_pct)
    surviving = set()
    too_rare = too_common = 0
    for c, df in concept_df.items():
        if df < min_df:
            too_rare += 1
        elif df > max_df:
            too_common += 1
        else:
            surviving.add(c)
    
    if verbose:
        print(f"\n  Step 2: DF filter (min={min_df}, "
              f"max={max_df} = {max_df_pct:.0%})")
        print(f"    Removed {too_rare:,} rare, {too_common:,} common")
        print(f"    Surviving: {len(surviving):,}")
    
    # Step 3: IDF
    idf = {}
    for c in surviving:
        idf[c] = math.log(n / concept_df[c])
    
    high_idf = {c for c, score in idf.items() if score >= min_idf}
    
    if verbose:
        idf_vals = [idf[c] for c in high_idf]
        print(f"\n  Step 3: IDF filter (min={min_idf})")
        print(f"    Surviving: {len(high_idf):,}")
        if idf_vals:
            print(f"    IDF range: {min(idf_vals):.2f} — "
                  f"{max(idf_vals):.2f}, med={np.median(idf_vals):.2f}")
    
    # Step 4: Update items
    concepts_per = []
    for idx, item in enumerate(data):
        kept_p, kept_r, _ = filtered[idx]
        
        prompt_c = kept_p & high_idf
        response_c = kept_r & high_idf
        all_c = prompt_c | response_c
        
        # Score: IDF × prompt_boost
        scores = {}
        for c in all_c:
            boost = prompt_boost if c in prompt_c else 1.0
            scores[c] = idf.get(c, 0) * boost
        
        # Top N
        if len(scores) > max_concepts:
            top = sorted(scores.items(), key=lambda x: x[1],
                         reverse=True)[:max_concepts]
            scores = dict(top)
            all_c = set(scores.keys())
        
        # Normalize
        total = sum(scores.values())
        if total > 0:
            weights = {c: round(s / total, 6) for c, s in scores.items()}
        else:
            weights = {}
        
        item['features']['concepts_all'] = sorted(all_c)
        item['features']['concept_weights'] = weights
        concepts_per.append(len(all_c))
    
    n_empty = sum(1 for c in concepts_per if c == 0)
    
    if verbose:
        print(f"\n  ┌────────────────────────────────────────┐")
        print(f"  │ V5-RELAXED SUMMARY                     │")
        print(f"  │ Concepts: {len(high_idf):>6,}                     │")
        print(f"  │ Per query: mean={np.mean(concepts_per):.1f}, "
              f"med={np.median(concepts_per):.0f}, "
              f"max={max(concepts_per)}  │")
        print(f"  │ Empty: {n_empty:>6,} ({n_empty/n*100:.1f}%)              │")
        print(f"  │ Time: {time.time()-t0:.1f}s                          │")
        print(f"  └────────────────────────────────────────┘")
        
        # Top concepts
        print(f"\n  Top 20 concepts by DF:")
        for c, df in sorted(((c, concept_df[c]) for c in high_idf),
                            key=lambda x: x[1], reverse=True)[:20]:
            print(f"    {c:35s} df={df:>5,}  idf={idf[c]:.2f}")
    
    return data


# =============================================================================
# COMMUNITY DETECTION
# =============================================================================

def _community_detect(sim_upper, n, resolution=1.0):
    if sim_upper.nnz == 0:
        return np.arange(n, dtype=int)
    sym = sim_upper + sim_upper.T
    try:
        import igraph as ig
        import leidenalg
        g = ig.Graph.Weighted_Adjacency(sym, mode='undirected')
        part = leidenalg.find_partition(
            g, leidenalg.ModularityVertexPartition,
            weights='weight', seed=42, n_iterations=3)
        return np.array(part.membership)
    except ImportError:
        pass
    try:
        import community as community_louvain
        import networkx as nx
        G = nx.from_scipy_sparse_array(sym, edge_attribute='weight')
        part = community_louvain.best_partition(
            G, resolution=resolution, random_state=42)
        return np.array([part[i] for i in range(n)])
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


# =============================================================================
# kNN GRAPH BUILDER
# =============================================================================

def build_knn_graph(X_norm, k, sigma=0.0, verbose=True):
    """Build kNN graph via sklearn — NEVER builds full sim matrix."""
    t0 = time.time()
    n = X_norm.shape[0]
    
    if verbose:
        print(f"    kNN graph (k={k}, σ={sigma})...")
    
    nn = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        metric='cosine',
        algorithm='brute',
        n_jobs=-1,
    )
    nn.fit(X_norm)
    if verbose:
        print(f"      fit [{time.time()-t0:.1f}s]")
    
    t_q = time.time()
    distances, indices = nn.kneighbors(X_norm)
    if verbose:
        print(f"      query [{time.time()-t_q:.1f}s]")
    
    # Build upper-triangle sparse matrix
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_pos in range(1, distances.shape[1]):
            j = indices[i, j_pos]
            sim = 1.0 - distances[i, j_pos]
            if sim >= sigma and j != i:
                a, b = min(i, j), max(i, j)
                rows.append(a)
                cols.append(b)
                vals.append(sim)
    
    graph = csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32),
          np.array(cols, dtype=np.int32))),
        shape=(n, n))
    
    # Dedup (max per edge)
    graph = graph.tocsr()
    
    sym = graph + graph.T
    deg = np.diff(sym.tocsr().indptr)
    iso = int(np.sum(deg == 0))
    
    if verbose:
        print(f"      {graph.nnz:,} edges, mean_deg={deg.mean():.1f}, "
              f"max={deg.max()}, iso={iso:,} "
              f"[{time.time()-t0:.1f}s total]")
    
    return graph


def build_knn_with_min_shared(X_norm, k, sigma, binary_mat,
                               min_shared, verbose=True):
    """kNN graph with min_shared concept filter."""
    graph = build_knn_graph(X_norm, k=k, sigma=sigma, verbose=verbose)
    
    if min_shared <= 0:
        return graph
    
    if verbose:
        print(f"    min_shared={min_shared}...")
    t0 = time.time()
    n_before = graph.nnz
    
    coo = graph.tocoo()
    batch = 100000
    mask = np.ones(len(coo.data), dtype=bool)
    
    for start in range(0, len(coo.data), batch):
        end = min(start + batch, len(coo.data))
        r_batch = coo.row[start:end]
        c_batch = coo.col[start:end]
        shared = np.array(
            binary_mat[r_batch].multiply(
                binary_mat[c_batch]
            ).sum(axis=1)
        ).flatten()
        mask[start:end] = shared >= min_shared
    
    filtered = csr_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])),
        shape=graph.shape)
    
    if verbose:
        print(f"      {n_before:,} → {filtered.nnz:,} "
              f"[{time.time()-t0:.1f}s]")
    
    return filtered


# =============================================================================
# CONCEPT VECTOR BUILDER
# =============================================================================

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
    QC_binary = QC.astype(bool).astype(np.float32)
    
    if verbose:
        nnz = np.diff(QC.indptr)
        print(f"\n  Concept vectors: {n_q:,} × {n_c:,}, "
              f"nnz={QC.nnz:,}")
        print(f"    Concepts/query: mean={nnz.mean():.1f}, "
              f"median={np.median(nnz):.0f}, "
              f"max={nnz.max()}")
        n_empty = int(np.sum(nnz == 0))
        print(f"    Empty: {n_empty:,} ({n_empty/n_q*100:.1f}%)")
        print(f"    [{time.time()-t0:.1f}s]")
    
    return QC, QC_norm, QC_binary, concepts, c2i


# =============================================================================
# GT EVALUATION
# =============================================================================

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
                    print(f"    GT error ({gran}): {e}")
                break
    return results


# =============================================================================
# CLUSTER DISPLAY
# =============================================================================

def show_clusters(labels, data, QC=None, concepts=None, top_n=10):
    print(f"\n    Top {top_n} clusters:")
    for cl_id in sorted(set(labels),
                         key=lambda x: np.sum(labels == x),
                         reverse=True)[:top_n]:
        members = np.where(labels == cl_id)[0]
        n_m = len(members)
        
        if QC is not None and concepts is not None:
            profile = QC[members].toarray().mean(axis=0)
            top_idx = np.argsort(-profile)[:4]
            label_str = ', '.join(
                f"{concepts[i]}({profile[i]:.3f})"
                for i in top_idx if profile[i] > 0)
        else:
            label_str = ''
        
        prompts = [data[i].get('prompt', '')[:70] for i in members[:3]]
        print(f"      Cl {cl_id} ({n_m:,}): {label_str}")
        for p in prompts:
            print(f"        → {p}...")


# =============================================================================
# COMPARISON TABLE
# =============================================================================

def print_table(results):
    if not results:
        return
    has = {g: any(f'nmi_{g}' in r for r in results)
           for g in ['coarse', 'mid', 'fine']}
    bests = {}
    for g in ['coarse', 'mid', 'fine']:
        vals = [r.get(f'nmi_{g}', 0) for r in results
                if isinstance(r.get(f'nmi_{g}', 0), (int, float))]
        bests[g] = max(vals) if vals else 0
    
    print(f"\n{'━'*145}")
    print(f"  RESULTS ({len(results)} runs)")
    print(f"{'━'*145}")
    
    hdr = (f"  {'#':>3} {'Mode':<20} {'σ_q':>4} {'kNN':>4} "
           f"{'Shrd':>4} │ {'k_q':>7} {'Real':>5} {'Max%':>5} "
           f"{'Covr%':>5} {'Eff_k':>6}")
    sep = (f"  {'─'*3} {'─'*20} {'─'*4} {'─'*4} "
           f"{'─'*4} │ {'─'*7} {'─'*5} {'─'*5} "
           f"{'─'*5} {'─'*6}")
    for g, tag in [('coarse','NMI_c'),('mid','NMI_m'),('fine','NMI_f')]:
        if has[g]:
            hdr += f" │ {tag:>6} {'ARI':>6}"
            sep += f" │ {'─'*6} {'─'*6}"
    hdr += f" │ {'NMI×Cov':>7} {'Time':>5}"
    sep += f" │ {'─'*7} {'─'*5}"
    print(hdr); print(sep)
    
    for r in results:
        nf = r.get('nmi_fine', 0)
        cv = r.get('coverage', 0)
        bal = nf * cv if isinstance(nf, (int, float)) else 0
        
        line = (f"  {r['run']:>3} {r.get('mode','?'):<20} "
                f"{r['sigma_q']:>4.2f} "
                f"{r.get('query_knn',''):>4} "
                f"{r.get('min_shared',0):>4} │ "
                f"{r['k_query']:>7,} {r.get('n_real',0):>5} "
                f"{r.get('max_pct',0)*100:>4.1f}% "
                f"{r.get('coverage',0)*100:>4.1f}% "
                f"{r.get('effective_k',0):>5.0f}")
        for g in ['coarse','mid','fine']:
            if has[g]:
                nmi = r.get(f'nmi_{g}','')
                ari = r.get(f'ari_{g}','')
                star = ''
                if isinstance(nmi,(int,float)) and nmi > 0:
                    star = '★' if abs(nmi-bests[g])<0.0001 else ' '
                if nmi == '':
                    line += f" │ {'':>6} {'':>6}"
                else:
                    line += f" │ {nmi:>6.4f}{star}{ari:>6.4f}"
        line += f" │ {bal:>7.4f} {r.get('total_time',0):>4.0f}s"
        print(line)
    print(f"{'━'*145}\n")


# =============================================================================
# MAIN SWEEP
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Concept kNN Clustering (best of everything)')
    
    # Input
    parser.add_argument('--features-file', required=True,
                        help='V4 or V5 features JSON')
    parser.add_argument('--skip-refilter', action='store_true',
                        help='Skip V5 re-filtering (use features as-is)')
    
    # V5-relaxed params
    parser.add_argument('--max-df-pct', type=float, default=0.08,
                        help='Max DF %% (default: 0.08 = 8%%)')
    parser.add_argument('--min-df', type=int, default=3)
    parser.add_argument('--min-idf', type=float, default=0.5)
    parser.add_argument('--max-concepts', type=int, default=30)
    parser.add_argument('--prompt-boost', type=float, default=3.0)
    
    # Clustering params
    parser.add_argument('--sigma-q', nargs='+', type=float,
                        default=[0.1, 0.15, 0.2, 0.3])
    parser.add_argument('--query-knn', nargs='+', type=int,
                        default=[30, 50])
    parser.add_argument('--min-shared', nargs='+', type=int,
                        default=[0, 3])
    
    # Eval
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='concept_knn_results')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load
    print("Loading features...")
    t0 = time.time()
    with open(args.features_file) as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} items [{time.time()-t0:.1f}s]")
    
    # Re-filter if needed
    if not args.skip_refilter:
        data = refilter_v5_relaxed(
            data,
            max_df_pct=args.max_df_pct,
            min_df=args.min_df,
            min_idf=args.min_idf,
            max_concepts=args.max_concepts,
            prompt_boost=args.prompt_boost,
        )
    
    # Build concept vectors
    QC, QC_norm, QC_binary, concepts, c2i = build_concept_vectors(data)
    
    # Sweep
    n_cfg = (len(args.sigma_q) * len(args.query_knn) *
             len(args.min_shared))
    print(f"\n{'#'*70}")
    print(f"CONCEPT kNN SWEEP: {n_cfg} configurations")
    print(f"  σ_q: {args.sigma_q}")
    print(f"  kNN: {args.query_knn}")
    print(f"  min_shared: {args.min_shared}")
    print(f"{'#'*70}")
    
    results = []
    run_num = 0
    
    for knn in args.query_knn:
        for sigma in args.sigma_q:
            for ms in args.min_shared:
                run_num += 1
                mode = f'concept_knn_ms{ms}'
                print(f"\n  ▶ Run {run_num}: σ={sigma}, "
                      f"kNN={knn}, ms={ms}")
                
                t_run = time.time()
                
                if ms > 0:
                    graph = build_knn_with_min_shared(
                        QC_norm, k=knn, sigma=sigma,
                        binary_mat=QC_binary,
                        min_shared=ms)
                else:
                    graph = build_knn_graph(
                        QC_norm, k=knn, sigma=sigma)
                
                t_cl = time.time()
                labels = _community_detect(graph, len(data))
                del graph; gc.collect()
                
                # Metrics
                sizes = Counter(labels)
                sa = np.array(sorted(sizes.values(), reverse=True))
                n_q = len(labels)
                k_out = len(sizes)
                n_real = sum(1 for s in sa if s > 1)
                singletons = sum(1 for s in sa if s == 1)
                coverage = sum(s for s in sa if s > 1) / n_q
                max_pct = sa[0] / n_q
                probs = sa / n_q
                eff_k = np.exp(-np.sum(probs * np.log(probs + 1e-12)))
                
                run_time = time.time() - t_run
                
                print(f"    → {k_out:,} clusters (real={n_real:,}, "
                      f"sing={singletons:,}) "
                      f"[cluster {time.time()-t_cl:.1f}s, "
                      f"total {run_time:.1f}s]")
                print(f"    Cov={coverage:.1%}, Max={max_pct:.1%}")
                
                gt_res = evaluate_gt(labels, args.gt_dir)
                for g in ['coarse','mid','fine']:
                    if f'nmi_{g}' in gt_res:
                        print(f"    NMI_{g}={gt_res[f'nmi_{g}']:.4f}, "
                              f"ARI_{g}={gt_res[f'ari_{g}']:.4f}")
                
                show_clusters(labels, data, QC=QC, concepts=concepts)
                
                summary = {
                    'run': run_num, 'mode': mode,
                    'sigma_q': sigma, 'query_knn': knn,
                    'min_shared': ms,
                    'k_query': k_out, 'n_real': n_real,
                    'coverage': round(coverage, 4),
                    'max_pct': round(max_pct, 4),
                    'effective_k': round(eff_k, 1),
                    'total_time': round(run_time, 1),
                    **gt_res,
                }
                results.append(summary)
                print_table(results)
                
                # Save labels
                with open(os.path.join(args.output_dir,
                          f'run_{run_num}.json'), 'w') as f:
                    json.dump({'config': summary,
                               'labels': labels.tolist()}, f)
    
    # Final
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*70}")
    print_table(results)
    
    if results:
        best = max(results,
                   key=lambda r: r.get('nmi_fine',0) * r.get('coverage',0))
        nf = best.get('nmi_fine', 0)
        cv = best.get('coverage', 0)
        print(f"\n  Best balanced: NMI_f={nf:.4f} × cov={cv:.1%} "
              f"= {nf*cv:.4f}")
        print(f"    Config: σ={best['sigma_q']}, "
              f"kNN={best.get('query_knn','')}, "
              f"ms={best.get('min_shared',0)}")
        
        # vs baselines
        print(f"\n  Comparison vs baselines:")
        print(f"    TF-IDF best:       NMI=0.156 × cov=100% = 0.156")
        print(f"    Concepts V5 best:  NMI=0.628 × cov=50%  = 0.314")
        print(f"    Super-topics best: NMI=0.361 × cov=92%  = 0.332")
        print(f"    THIS (concept-kNN): NMI={nf:.3f} × "
              f"cov={cv:.0%} = {nf*cv:.3f}")
    
    sweep_file = os.path.join(args.output_dir, 'sweep_summary.json')
    with open(sweep_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Saved: {sweep_file}")


if __name__ == '__main__':
    main()