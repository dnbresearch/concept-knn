"""
Shared evaluation — identical to concept_knn_v5.py / llm_ground_truth.py
=========================================================================
All baselines import this to ensure consistent NMI computation.
"""

import os, json
import numpy as np
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score


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
    # Convert string keys to int
    pl = {}
    for k, v in gt.get('prompt_labels', {}).items():
        try: pl[int(k)] = v
        except: pl[k] = v
    gt['prompt_labels'] = pl
    return gt


def evaluate_gt(labels, gt_dir):
    """Evaluate cluster labels against GT files in gt_dir.
    Returns dict with nmi_coarse, nmi_mid, nmi_fine."""
    results = {}
    if not gt_dir or not os.path.isdir(gt_dir):
        return results
    for gran in ['coarse', 'mid', 'fine']:
        for fname in sorted(os.listdir(gt_dir)):
            if f'gt_{gran}_' in fname and fname.endswith('.json') and '_prompt_labels' not in fname:
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
                        results[f'nmi_{gran}'] = round(normalized_mutual_info_score(
                            np.array(true), np.array(pred)), 4)
                except Exception as e:
                    print(f"    GT err ({gran}): {e}")
                break
    return results


def compute_metrics(labels, n_total):
    sizes = Counter(labels)
    coverage = sum(s for s in sizes.values() if s > 1) / n_total
    return {
        'k_clusters': len(sizes),
        'coverage': round(coverage, 4),
        'singletons': sum(1 for s in sizes.values() if s == 1),
    }


def load_features(path):
    """Load features_v4.json → (data, prompts)."""
    print(f"Loading {path}...")
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        prompt = item.get('prompt', '')
        if not prompt:
            prompt = ' '.join(item.get('features', {}).get('concepts_all', []))
        prompts.append(prompt)
    print(f"  {len(prompts):,} items loaded")
    return data, prompts


def report(method, metrics, gt_results, extra=""):
    """Standard one-line result report."""
    nf = gt_results.get('nmi_fine', 0)
    cov = metrics['coverage']
    bal = round(nf * cov, 4)
    print(f"  {method:<35s} NMI_f={nf:.4f} cov={cov:.4f} bal={bal:.4f} {extra}")
    return bal
