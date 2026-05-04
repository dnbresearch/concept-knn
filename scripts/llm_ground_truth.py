"""
LLM-Based Ground Truth Generator for QDC Clustering Evaluation
================================================================

Updated for V4 pipeline: works from features JSON, no graph needed.

Input:  features_v4.json (from feature_extraction_v4.py)
Output: gt_{granularity}_{timestamp}.json with:
          prompt_labels: {prompt_idx → category}
          concept_categories: {concept → category}
          clusters: {category → [prompt_indices]}

Pipeline:
    1. Load features JSON → extract all unique concepts
    2. Batch concepts to LLM for semantic categorization
    3. Merge LLM categories across batches
    4. Propagate concept labels → prompt labels (dominant category)
    5. Save at requested granularities

Usage:
    python llm_ground_truth.py --features features_v4.json --model gpt-5.2
    python llm_ground_truth.py --features features_v4.json --provider anthropic --model claude-sonnet-4-20250514
    python llm_ground_truth.py --features features_v4.json --granularity coarse mid fine
    python llm_ground_truth.py --features features_v4.json --dry-run
"""

import json, os, sys, time, random, hashlib, re
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path


# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Unified LLM client for OpenAI and Anthropic APIs"""

    def __init__(self, provider='openai', model='gpt-5.2',
                 api_key='', max_retries=3, rate_limit_delay=1.0):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get(
            'OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY', '')
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.total_tokens = 0
        self.total_calls = 0
        if not self.api_key:
            raise ValueError(
                f"No API key. Set "
                f"{'OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY'} "
                f"or pass --api-key")

    def call(self, system_prompt, user_prompt, temperature=0.3,
             max_tokens=4096, json_mode=False):
        for attempt in range(self.max_retries):
            try:
                if self.provider == 'openai':
                    return self._call_openai(system_prompt, user_prompt,
                                              temperature, max_tokens, json_mode)
                else:
                    return self._call_anthropic(system_prompt, user_prompt,
                                                 temperature, max_tokens)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = (attempt + 1) * self.rate_limit_delay * 2
                    print(f"    API error (attempt {attempt+1}): {e}")
                    time.sleep(wait)
                else:
                    raise

    def _call_openai(self, system_prompt, user_prompt, temperature,
                      max_tokens, json_mode):
        import urllib.request, urllib.error
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        is_new = any(x in self.model for x in
                      ['gpt-4o', 'gpt-4.1', 'gpt-5', 'o1', 'o3', 'o4'])
        tok_key = "max_completion_tokens" if is_new else "max_tokens"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tok_key: max_tokens,
        }
        if not any(x in self.model for x in ['o1', 'o3', 'o4']):
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err = e.read().decode('utf-8', errors='replace')
            try:
                err = json.loads(err).get('error', {}).get('message', err)
            except:
                pass
            raise RuntimeError(f"OpenAI {e.code}: {err}")
        self.total_tokens += data.get('usage', {}).get('total_tokens', 0)
        self.total_calls += 1
        time.sleep(self.rate_limit_delay)
        return data['choices'][0]['message']['content']

    def _call_anthropic(self, system_prompt, user_prompt, temperature,
                         max_tokens):
        import urllib.request, urllib.error
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        body = json.dumps({
            "model": self.model, "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": temperature,
        }).encode()
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            err = e.read().decode('utf-8', errors='replace')
            try:
                err = json.loads(err).get('error', {}).get('message', err)
            except:
                pass
            raise RuntimeError(f"Anthropic {e.code}: {err}")
        self.total_tokens += data.get('usage', {}).get('input_tokens', 0)
        self.total_tokens += data.get('usage', {}).get('output_tokens', 0)
        self.total_calls += 1
        time.sleep(self.rate_limit_delay)
        return data['content'][0]['text']


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT_COARSE = """You categorize LLM conversation topics.

Categorize each concept into ONE of these categories:
programming, math, science, creative_writing, business, health, education,
entertainment, history, geography, technology, philosophy, psychology,
language, food_cooking, sports, art_design, music, legal, finance,
relationships, travel, nature, politics, religion, daily_life, career, gaming, other

Respond with FLAT JSON: {"concept": "category", ...}
Do NOT use arrays or nested objects."""

SYSTEM_PROMPT_MID = """You categorize LLM conversation topics into mid-level categories.

Use "broad:subcategory" format, targeting 80-150 total subcategories.
Examples:
- "python" -> "programming:python"
- "sort algorithm" -> "programming:algorithms"
- "build website" -> "programming:web_development"
- "calculus" -> "math:calculus"
- "anxiety" -> "psychology:mental_health"
- "stock market" -> "finance:investing"
- "social media marketing" -> "business:marketing"
- "write poem" -> "creative_writing:poetry"
- "save conversation" -> "technology:chat_features"

Broad categories: programming, math, science, creative_writing, business,
health, education, entertainment, history, geography, technology, philosophy,
psychology, language, food_cooking, sports, art_design, music, legal, finance,
relationships, travel, nature, politics, religion, daily_life, career, gaming, other

Rules:
1. ALWAYS use "broad:specific" with colon
2. Target 3-8 subcategories per broad category
3. Respond with FLAT JSON: {"concept": "broad:subcategory", ...}
Do NOT use arrays or nested objects."""

SYSTEM_PROMPT_FINE = """You categorize LLM conversation topics into specific subcategories.

Use "broad:subcategory" format. Be specific but not too narrow.
Examples:
- "python" -> "programming:python"
- "web scraping" -> "programming:web_scraping"
- "calculus" -> "math:calculus"
- "nutrition" -> "health:nutrition"
- "anxiety" -> "psychology:mental_health"
- "stock market" -> "finance:investing"
- "solar system" -> "science:astronomy"
- "resume tips" -> "career:job_search"
- "save conversation" -> "technology:chat_features"
- "social proof" -> "business:marketing"

Broad categories: programming, math, science, creative_writing, business,
health, education, entertainment, history, geography, technology, philosophy,
psychology, language, food_cooking, sports, art_design, music, legal, finance,
relationships, travel, nature, politics, religion, daily_life, career, gaming, other

Rules:
1. ALWAYS use "broad:specific" with colon
2. Target 5-15 subcategories per broad category
3. Respond with FLAT JSON: {"concept": "broad:subcategory", ...}
Do NOT use arrays or nested objects."""

MERGE_SYSTEM_PROMPT = """Merge similar subcategory names into canonical forms.
Keep "broad:specific" format. Merge near-duplicates only.
Respond with FLAT JSON: {"old_name": "canonical_name", ...}
Only include entries that actually change."""


# ============================================================================
# GROUND TRUTH GENERATOR
# ============================================================================

class GroundTruthGenerator:
    """
    Generate ground truth from features JSON using LLM categorization.

    Pipeline:
        1. Load features JSON -> extract unique concepts
        2. Batch to LLM for categorization
        3. Merge categories across batches
        4. Propagate concept labels -> prompt labels (dominant category)
        5. Save
    """

    def __init__(self, features_file, output_dir='ground_truth',
                 provider='openai', model='gpt-5.2', api_key='',
                 batch_size=200, cache_dir='.gt_cache', granularity='mid'):
        self.features_file = features_file
        self.output_dir = output_dir
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.granularity = granularity

        self.data = []
        self.concepts = []
        self.concept_freq = {}
        self.assignments = {}
        self.prompt_labels = {}
        self.reference_clusters = {}

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

    def run(self, dry_run=False):
        self._load_data()
        self._extract_concepts()
        if dry_run:
            self._dry_run_report()
            return {}
        self._batch_llm_categorize()
        self._merge_categories()
        self._propagate_to_prompts()
        return self._save()

    # ── Load ──

    def _load_data(self):
        print("=" * 70)
        print(f"LLM GROUND TRUTH GENERATOR (granularity={self.granularity})")
        print("=" * 70)
        with open(self.features_file) as f:
            self.data = json.load(f)
        print(f"  Loaded {len(self.data):,} prompts from {self.features_file}")

    # ── Extract concepts ──

    def _extract_concepts(self):
        freq = Counter()
        for item in self.data:
            if 'features' not in item:
                continue
            for c in item['features'].get('concepts_all', []):
                freq[c] += 1

        self.concepts = sorted(freq.keys())
        self.concept_freq = dict(freq)

        freqs = sorted(freq.values(), reverse=True)
        n_prompts_with_features = sum(1 for i in self.data if 'features' in i)
        avg_concepts = np.mean([
            len(i['features'].get('concepts_all', []))
            for i in self.data if 'features' in i])

        print(f"  Unique concepts: {len(self.concepts):,}")
        print(f"  Concepts/prompt: {avg_concepts:.1f} avg")
        print(f"  Concept freq: max={freqs[0]:,}, "
              f"median={np.median(freqs):.0f}, "
              f"singletons={sum(1 for f in freqs if f == 1):,}")
        print(f"\n  Top 15 concepts:")
        for c, f in freq.most_common(15):
            print(f"    {c:35s} df={f:,}")

    # ── Batch LLM ──

    def _batch_llm_categorize(self):
        client = LLMClient(self.provider, self.model, self.api_key)

        # High-freq concepts first (better LLM context)
        concepts_sorted = sorted(
            self.concepts,
            key=lambda c: self.concept_freq.get(c, 0), reverse=True)

        n_batches = (len(concepts_sorted) + self.batch_size - 1) // self.batch_size

        sys_prompt = {
            'coarse': SYSTEM_PROMPT_COARSE,
            'mid': SYSTEM_PROMPT_MID,
            'fine': SYSTEM_PROMPT_FINE,
        }[self.granularity]

        print(f"\n  Categorizing {len(concepts_sorted):,} concepts "
              f"in {n_batches} batches")

        all_results = {}
        failed = 0

        for bi in range(n_batches):
            start = bi * self.batch_size
            end = min(start + self.batch_size, len(concepts_sorted))
            batch = concepts_sorted[start:end]

            # Cache
            cache_key = hashlib.md5(
                json.dumps(sorted(batch)).encode()).hexdigest()[:12]
            cache_file = os.path.join(
                self.cache_dir,
                f'b_{self.granularity}_{cache_key}.json')

            if os.path.exists(cache_file):
                with open(cache_file) as f:
                    result = json.load(f)
                n_unk = sum(1 for v in result.values() if v == 'unknown')
                if n_unk <= len(result) * 0.5:
                    cats = len(set(result.values()) - {'unknown'})
                    print(f"    [{bi+1}/{n_batches}] Cached "
                          f"({len(result)} concepts, {cats} cats)")
                    all_results.update(result)
                    continue
                os.remove(cache_file)

            lines = '\n'.join(f'{i+1}. {c}' for i, c in enumerate(batch))
            user_prompt = (
                f"Categorize each concept. Return FLAT JSON: "
                f"concept -> category.\n\n{lines}")

            max_tok = max(4096, len(batch) * 25)
            print(f"    [{bi+1}/{n_batches}] {len(batch)} concepts...",
                  end=' ', flush=True)

            try:
                resp = client.call(
                    sys_prompt, user_prompt,
                    max_tokens=max_tok,
                    json_mode=(self.provider == 'openai'))
                result = self._parse_json(resp, batch)

                n_unk = sum(1 for v in result.values() if v == 'unknown')
                n_cats = len(set(v for v in result.values() if v != 'unknown'))

                if n_unk > len(batch) * 0.5:
                    print(f"warn: {n_unk}/{len(batch)} unknown")
                    failed += 1
                else:
                    with open(cache_file, 'w') as f:
                        json.dump(result, f)
                    print(f"-> {n_cats} cats")
            except Exception as e:
                print(f"error: {e}")
                result = {c: 'unknown' for c in batch}
                failed += 1

            all_results.update(result)

        self.assignments = all_results
        n_unk = sum(1 for v in all_results.values() if v == 'unknown')
        cats = set(v for v in all_results.values() if v != 'unknown')
        print(f"\n  Results: {len(all_results)-n_unk:,}/{len(all_results):,} "
              f"categorized, {len(cats)} categories, "
              f"{failed} failed batches")
        print(f"  API: {client.total_calls} calls, "
              f"{client.total_tokens:,} tokens")

    def _parse_json(self, response, batch):
        """Extract concept->category JSON from LLM response."""
        text = response.strip()

        # Strip markdown
        if '```' in text:
            m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if m:
                text = m.group(1).strip()

        # Try direct parse
        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Find largest {...}
            depth = 0; start = None; best = ""
            for i, ch in enumerate(text):
                if ch == '{':
                    if depth == 0: start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start is not None:
                        blk = text[start:i+1]
                        if len(blk) > len(best): best = blk
            if best:
                try:
                    parsed = json.loads(best)
                except:
                    try:
                        parsed = json.loads(re.sub(r',\s*}', '}', best))
                    except:
                        pass

        if isinstance(parsed, dict) and parsed:
            first_val = next(iter(parsed.values()))
            if isinstance(first_val, str):
                return parsed
            elif isinstance(first_val, list):
                # Inverted format
                result = {}
                for cat, concepts in parsed.items():
                    if isinstance(concepts, list):
                        for c in concepts:
                            if isinstance(c, str):
                                result[c] = cat
                if result:
                    return result

        # Line fallback
        result = {}
        batch_set = set(batch)
        for line in text.split('\n'):
            m = re.match(r'"([^"]+)"\s*:\s*"([^"]+)"', line.strip().rstrip(','))
            if m and m.group(1) in batch_set:
                result[m.group(1)] = m.group(2)
        if len(result) >= len(batch) * 0.3:
            return result

        return {c: 'unknown' for c in batch}

    # ── Merge categories ──

    def _merge_categories(self):
        cats = sorted(set(v for v in self.assignments.values() if v != 'unknown'))
        n = len(cats)
        target = {'coarse': (15, 40), 'mid': (80, 150), 'fine': (200, 500)
                  }[self.granularity]

        print(f"\n  Categories before merge: {n} (target: {target[0]}-{target[1]})")
        if n <= target[1]:
            print(f"  Within target — skip merge")
            return

        cat_samples = defaultdict(list)
        for c, cat in self.assignments.items():
            if cat != 'unknown' and len(cat_samples[cat]) < 3:
                cat_samples[cat].append(c)

        client = LLMClient(self.provider, self.model, self.api_key)
        merge_map = {}
        batch_sz = 300
        items = sorted(cat_samples.items())

        for i in range(0, len(items), batch_sz):
            batch = items[i:i+batch_sz]
            cat_list = '\n'.join(
                f'- {cat} (e.g., {", ".join(samp)})'
                for cat, samp in batch)
            prompt = (f"Merge similar subcategories. "
                      f"Target {target[0]}-{target[1]} total.\n\n"
                      f"Subcategories:\n{cat_list}\n\n"
                      f"Return FLAT JSON: old -> canonical. "
                      f"Only include changes.")
            bn = i // batch_sz + 1
            total = (len(items) + batch_sz - 1) // batch_sz
            print(f"    Merge {bn}/{total}...", end=' ', flush=True)
            resp = client.call(MERGE_SYSTEM_PROMPT, prompt,
                               max_tokens=8192,
                               json_mode=(self.provider == 'openai'))
            bm = self._parse_json(resp, [c for c, _ in batch])
            merge_map.update(bm)
            print(f"-> {len(set(bm.values()))} canonical")

        if merge_map:
            for c in self.assignments:
                old = self.assignments[c]
                if old in merge_map:
                    self.assignments[c] = merge_map[old]

        final = set(v for v in self.assignments.values() if v != 'unknown')
        print(f"  After merge: {len(final)} categories")

    # ── Propagate to prompts ──

    def _propagate_to_prompts(self):
        """Label each prompt by its dominant concept category."""
        print(f"\n  Propagating concept labels -> prompt labels...")

        self.prompt_labels = {}
        self.reference_clusters = defaultdict(list)
        n_labeled = 0

        for qi, item in enumerate(self.data):
            if 'features' not in item:
                continue
            concepts = item['features'].get('concepts_all', [])
            weights = item['features'].get('concept_weights', {})

            cat_w = defaultdict(float)
            for c in concepts:
                cat = self.assignments.get(c, 'unknown')
                if cat == 'unknown':
                    continue
                cat_w[cat] += weights.get(c, 0.1)

            if cat_w:
                label = max(cat_w, key=cat_w.get)
                self.prompt_labels[qi] = label
                self.reference_clusters[label].append(qi)
                n_labeled += 1

        self.reference_clusters = dict(self.reference_clusters)
        print(f"    Labeled: {n_labeled:,}/{len(self.data):,} "
              f"({n_labeled/len(self.data)*100:.1f}%)")
        print(f"    Categories: {len(self.reference_clusters)}")

        print(f"\n  Top 15 prompt categories:")
        for cat, members in sorted(self.reference_clusters.items(),
                                    key=lambda x: len(x[1]), reverse=True)[:15]:
            sample = self.data[members[0]]['prompt'][:55]
            print(f"    {cat:35s}: {len(members):>5,d}  e.g. {sample}...")

    # ── Save ──

    def _save(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        prefix = f'gt_{self.granularity}_{ts}'

        gt = {
            'granularity': self.granularity,
            'model': self.model,
            'provider': self.provider,
            'timestamp': datetime.now().isoformat(),
            'n_prompts': len(self.data),
            'n_concepts': len(self.concepts),
            'n_categories': len(self.reference_clusters),
            'n_labeled_prompts': len(self.prompt_labels),
            'concept_categories': self.assignments,
            'prompt_labels': {str(k): v for k, v in self.prompt_labels.items()},
            'clusters': self.reference_clusters,
            'category_sizes': {
                cat: len(ms) for cat, ms in sorted(
                    self.reference_clusters.items(),
                    key=lambda x: len(x[1]), reverse=True)},
        }

        gt_file = os.path.join(self.output_dir, f'{prefix}.json')
        with open(gt_file, 'w') as f:
            json.dump(gt, f, indent=2)
        print(f"\n  Saved: {gt_file}")

        labels_file = os.path.join(self.output_dir,
                                    f'{prefix}_prompt_labels.json')
        with open(labels_file, 'w') as f:
            json.dump({str(k): v for k, v in self.prompt_labels.items()}, f)
        print(f"  Saved: {labels_file}")

        return gt

    # ── Dry run ──

    def _dry_run_report(self):
        n_batches = (len(self.concepts) + self.batch_size - 1) // self.batch_size
        est_tok = n_batches * 500 + 2000
        print(f"\n  === DRY RUN ===")
        print(f"  Concepts: {len(self.concepts):,}")
        print(f"  Batches: {n_batches} x {self.batch_size}")
        print(f"  Est. tokens: ~{est_tok:,}")
        print(f"  Est. cost: ~${est_tok * 0.005 / 1000:.2f}")
        print(f"\n  Sample concepts:")
        random.seed(42)
        for c in random.sample(self.concepts, min(25, len(self.concepts))):
            print(f"    {c:40s} (df={self.concept_freq.get(c,0):,})")


# ============================================================================
# EVALUATION HELPERS (used by qdc_clustering.py)
# ============================================================================

def load_ground_truth(gt_file):
    """Load GT for evaluation. Returns dict with prompt_labels, clusters."""
    with open(gt_file) as f:
        data = json.load(f)
    prompt_labels = {}
    for k, v in data.get('prompt_labels', {}).items():
        try:
            prompt_labels[int(k)] = v
        except:
            prompt_labels[k] = v
    return {
        'prompt_labels': prompt_labels,
        'concept_categories': data.get('concept_categories', {}),
        'clusters': data.get('clusters', {}),
        'n_categories': data.get('n_categories', 0),
        'granularity': data.get('granularity', 'unknown'),
    }


def evaluate_against_gt(query_labels, gt_file):
    """Compute NMI/ARI of query_labels vs ground truth."""
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    gt = load_ground_truth(gt_file)
    pl = gt['prompt_labels']

    pred, true = [], []
    enc = {}
    for qi in range(len(query_labels)):
        if qi in pl:
            t = pl[qi]
            if t not in enc: enc[t] = len(enc)
            pred.append(query_labels[qi])
            true.append(enc[t])

    if len(pred) < 10:
        return {'error': f'Only {len(pred)} matched'}

    pred, true = np.array(pred), np.array(true)
    return {
        'nmi': round(normalized_mutual_info_score(true, pred), 4),
        'ari': round(adjusted_rand_score(true, pred), 4),
        'n_pred_clusters': len(set(pred)),
        'n_gt_clusters': len(set(true)),
        'n_matched': len(pred),
        'granularity': gt['granularity'],
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='LLM ground truth for QDC clustering evaluation')
    parser.add_argument('--features', required=True,
                        help='Features JSON (from feature_extraction_v4.py)')
    parser.add_argument('--output-dir', default='ground_truth')
    parser.add_argument('--provider', default='openai',
                        choices=['openai', 'anthropic'])
    parser.add_argument('--model', default='gpt-5.2')
    parser.add_argument('--api-key', default='')
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--granularity', nargs='+', default=['mid'],
                        choices=['coarse', 'mid', 'fine'])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-api', action='store_true')
    args = parser.parse_args()

    if args.test_api:
        print("Testing API...")
        client = LLMClient(args.provider, args.model,
                            args.api_key or os.environ.get(
                                'OPENAI_API_KEY' if args.provider == 'openai'
                                else 'ANTHROPIC_API_KEY', ''))
        try:
            r = client.call(
                SYSTEM_PROMPT_MID,
                'Categorize: 1. python\n2. calculus\n3. anxiety\nReturn FLAT JSON.',
                max_tokens=200,
                json_mode=(args.provider == 'openai'))
            print(f"  OK: {r[:300]}")
        except Exception as e:
            print(f"  Failed: {e}")
        return

    for gran in args.granularity:
        gen = GroundTruthGenerator(
            features_file=args.features,
            output_dir=args.output_dir,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            batch_size=args.batch_size,
            granularity=gran,
        )
        gen.run(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
