#!/usr/bin/env python3
"""
Slim down cluster_profiles.json for paper examples.
Keeps top 30 clusters by size, 3 prompts each, 10 concepts each.

USAGE:
  python slim_profiles.py cluster_profiles.json cluster_profiles_slim.json
"""
import json, sys

inp = sys.argv[1] if len(sys.argv) > 1 else 'cluster_profiles.json'
out = sys.argv[2] if len(sys.argv) > 2 else 'cluster_profiles_slim.json'

with open(inp) as f:
    data = json.load(f)

slim_clusters = []
for p in data['clusters'][:30]:  # top 30 by size
    slim_clusters.append({
        'cluster_id': p['cluster_id'],
        'size': p['size'],
        'top_concepts': [c['concept'] for c in p['top_concepts'][:10]],
        'example_prompts': [
            {'prompt': ex['prompt'][:200], 'concepts': ex['concepts'][:6]}
            for ex in p['example_prompts'][:3]
        ],
        'gt_coarse': p['gt_coarse']['dominant_label'],
        'gt_coarse_purity': p['gt_coarse']['purity'],
        'gt_fine': p['gt_fine']['dominant_label'],
        'gt_fine_purity': p['gt_fine']['purity'],
    })

result = {'summary': data['summary'], 'clusters': slim_clusters}
with open(out, 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Saved {len(slim_clusters)} clusters to {out}")
