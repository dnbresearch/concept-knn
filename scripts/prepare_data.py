"""
Data preparation helper — converts your existing data format
into the JSONL + GT JSON format expected by the baseline scripts.

Adapt this script to your specific data format.
"""
import json, os, sys

def prepare_from_concept_knn_data(concepts_dir, output_dir, dataset_name="sharegpt"):
    """
    If your Concept-kNN pipeline saves intermediate data, 
    adapt this function to extract prompts and GT labels.
    
    Expected outputs:
      {output_dir}/{dataset_name}_prompts.jsonl
      {output_dir}/{dataset_name}_gt_fine.json
      {output_dir}/{dataset_name}_gt_mid.json
      {output_dir}/{dataset_name}_gt_coarse.json
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # === ADAPT THIS SECTION TO YOUR DATA ===
    
    # Option A: If you have a list of prompts in a JSON/pickle file
    # import pickle
    # with open(f"{concepts_dir}/prompts.pkl", "rb") as f:
    #     prompts = pickle.load(f)
    
    # Option B: If you have a CSV
    # import pandas as pd
    # df = pd.read_csv(f"{concepts_dir}/data.csv")
    # prompts = df["prompt"].tolist()
    
    # Option C: If prompts are in your concept extraction output
    # with open(f"{concepts_dir}/extracted_concepts.json") as f:
    #     data = json.load(f)
    # prompts = [item["prompt"] for item in data]
    
    # === Write prompts JSONL ===
    # with open(f"{output_dir}/{dataset_name}_prompts.jsonl", "w") as f:
    #     for prompt in prompts:
    #         f.write(json.dumps({"prompt": prompt}) + "\n")
    
    # === Write GT labels ===
    # For ground truth, you need {index: label} mapping
    # This is the output of your LLM-based annotation pipeline
    
    # with open(f"{concepts_dir}/gt_labels_fine.json") as f:
    #     gt_fine = json.load(f)
    # with open(f"{output_dir}/{dataset_name}_gt_fine.json", "w") as f:
    #     json.dump(gt_fine, f)
    
    print("Adapt this script to your data format!")
    print("Required outputs:")
    print(f"  {output_dir}/{dataset_name}_prompts.jsonl  — one {{\"prompt\": ...}} per line")
    print(f"  {output_dir}/{dataset_name}_gt_fine.json   — {{\"0\": \"label\", \"1\": \"label\", ...}}")


def verify_data(prompts_path, gt_path):
    """Verify data files are correctly formatted."""
    # Check prompts
    n_prompts = 0
    with open(prompts_path) as f:
        for line in f:
            obj = json.loads(line)
            assert "prompt" in obj, f"Missing 'prompt' field: {line[:100]}"
            n_prompts += 1
    
    # Check GT
    with open(gt_path) as f:
        gt = json.load(f)
    
    n_gt = len(gt)
    n_labels = len(set(gt.values()))
    
    print(f"Prompts: {n_prompts}")
    print(f"GT labels: {n_gt} items, {n_labels} unique categories")
    print(f"GT coverage: {n_gt/n_prompts*100:.1f}%")
    
    # Check index alignment
    gt_indices = set(int(k) for k in gt.keys())
    missing = [i for i in range(n_prompts) if i not in gt_indices]
    if missing:
        print(f"Warning: {len(missing)} prompts have no GT label "
              f"(first 5: {missing[:5]})")
    
    print("✓ Data format looks correct!")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        verify_data(sys.argv[1], sys.argv[2])
    else:
        print("Usage:")
        print("  Verify:  python prepare_data.py prompts.jsonl gt_fine.json")
        print("  Prepare: Adapt prepare_from_concept_knn_data() to your format")
