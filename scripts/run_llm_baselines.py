#!/usr/bin/env python3
"""
Run all LLM-based clustering baselines
=======================================
Same interface as concept_knn_v5.py:
  --features-file features_v4.json
  --gt-dir ground_truth/

Usage:
  python run_llm_baselines.py \\
    --features-file features_v4.json \\
    --gt-dir ground_truth/ \\
    --api-key YOUR_OPENAI_KEY \\
    --run clusterllm tntllm clio

Cost estimates (gpt-4o-mini per dataset):
  ClusterLLM:  ~1024 triplets  → ~$0.03 + SBERT fine-tuning time
  TnT-LLM:    ~1000 summaries + 3000 labels → ~$0.50-1.00
  Clio:        ~N items × 300 tok each
               5K sample: ~$0.45  |  100K all: ~$9  |  162K all: ~$15

Requirements:
  pip install openai sentence-transformers scikit-learn numpy tqdm
  eval_shared.py must be in same directory
"""

import argparse, json, os, subprocess, sys, time


def count_items(path):
    with open(path) as f:
        return len(json.load(f))


def estimate_costs(n_items, clio_sample):
    cs = clio_sample if clio_sample > 0 else n_items
    costs = {
        "ClusterLLM": 1024 * 250 * 0.0003 / 1e6,
        "TnT-LLM":   (1000*150 + 3000*300) * 0.0003 / 1e6,
        "Clio":       cs * 300 * 0.0003 / 1e6,
    }
    print(f"\n  Cost estimates (gpt-4o-mini):")
    for m, c in costs.items():
        print(f"    {m:<15s}: ~${c:.2f}")
    print(f"    {'TOTAL':<15s}: ~${sum(costs.values()):.2f}\n")


def run_method(name, script, features_file, gt_dir, output_dir, api_key,
               model, extra_args=None):
    cmd = [sys.executable, script,
           "--features-file", features_file,
           "--output-dir", os.path.join(output_dir, name),
           "--api-key", api_key, "--model", model]
    if gt_dir:
        cmd.extend(["--gt-dir", gt_dir])
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"  RUNNING: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    print(f"  {name} {'completed' if result.returncode==0 else 'FAILED'} in {elapsed:.0f}s")
    return elapsed, result.returncode


def collect_results(output_dir):
    print(f"\n{'='*70}")
    print(f"  ALL RESULTS")
    print(f"{'='*70}")
    print(f"  {'Method':<30s} {'bal':>8s}")
    print(f"  {'-'*30} {'-'*8}")

    for name in ["clusterllm", "tntllm", "clio"]:
        rf = os.path.join(output_dir, name, "results.json")
        if os.path.exists(rf):
            with open(rf) as f:
                d = json.load(f)
            # Extract best bal
            if "finetuned" in d:
                bal = d["finetuned"]["bal"]
                label = "ClusterLLM (finetuned)"
            elif "bal" in d:
                bal = d["bal"]
                label = d.get("method", name)
            elif "clio_best" in d:
                bal = d["clio_best"]["bal"]
                label = "Clio-style"
            else:
                bal = "?"
                label = name
            print(f"  {label:<30s} {bal:>8.4f}" if isinstance(bal, float)
                  else f"  {label:<30s} {'?':>8s}")

    print(f"  {'-'*30} {'-'*8}")
    print(f"  {'Concept-kNN (ours):':<30s}")
    print(f"    {'ShareGPT':<28s} {'0.627':>8s}")
    print(f"    {'LMSYS':<28s} {'0.635':>8s}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-dir', default=None)
    parser.add_argument('--output-dir', default='results_baselines')
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--run', nargs='+', default=['all'],
                        choices=['clusterllm', 'tntllm', 'clio', 'all'])
    parser.add_argument('--clio-sample', type=int, default=5000)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key: raise ValueError("Set --api-key or OPENAI_API_KEY")
    os.makedirs(args.output_dir, exist_ok=True)

    methods = args.run
    if "all" in methods: methods = ["clusterllm", "tntllm", "clio"]

    n = count_items(args.features_file)
    print(f"Dataset: {n:,} items")
    estimate_costs(n, args.clio_sample)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if "clusterllm" in methods:
        run_method("clusterllm",
                   os.path.join(script_dir, "baseline_clusterllm.py"),
                   args.features_file, args.gt_dir, args.output_dir,
                   api_key, args.model)

    if "tntllm" in methods:
        run_method("tntllm",
                   os.path.join(script_dir, "baseline_tntllm.py"),
                   args.features_file, args.gt_dir, args.output_dir,
                   api_key, args.model)

    if "clio" in methods:
        run_method("clio",
                   os.path.join(script_dir, "baseline_clio.py"),
                   args.features_file, args.gt_dir, args.output_dir,
                   api_key, args.model,
                   extra_args=["--facet-sample", str(args.clio_sample)])

    collect_results(args.output_dir)


if __name__ == "__main__":
    main()
