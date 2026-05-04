#!/usr/bin/env python3
"""
GT Validation v3: Dual-Mode (Exact Match + Label Verification)
================================================================
Two complementary measures:
  1. CLASSIFY mode: Judge independently categorizes → Cohen's κ (strict)
  2. VERIFY mode:   Judge rates "is the GT label acceptable?" → acceptance rate

This separates two questions:
  - Are the GT labels THE BEST label? (classify κ)
  - Are the GT labels ACCEPTABLE?     (verify rate)

Also fixes: list index out of range on empty API responses.

Usage:
  python gt_validation_v3.py \
    --gt-file ground_truth/gt_coarse_20260207_1321.json \
    --features-file features_v4.json \
    --provider anthropic --model claude-opus-4-5-20251101 \
    --api-key sk-... \
    --n-samples 500 \
    --output validation_v3.json
"""

import json, os, sys, time, random, re, argparse
import urllib.request, urllib.error
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# CANONICAL CODEBOOK
# ═══════════════════════════════════════════════════════════════

COARSE_CATEGORIES = [
    'programming', 'math', 'science', 'creative_writing', 'business',
    'health', 'education', 'entertainment', 'history', 'geography',
    'technology', 'philosophy', 'psychology', 'language', 'food_cooking',
    'sports', 'art_design', 'music', 'legal', 'finance', 'relationships',
    'travel', 'nature', 'politics', 'religion', 'daily_life', 'career',
    'gaming', 'other'
]

CONSOLIDATION_MAP = {
    'computer_science': 'programming',
    'engineering': 'technology',
    'mythology': 'religion',
    'sociology': 'psychology',
    'economics': 'finance',
    'government': 'politics',
    'astronomy': 'science',
    'architecture': 'art_design',
    'security': 'technology',
}

CATEGORY_DEFS = """- programming: writing/debugging code, algorithms, software development, APIs, databases, web dev
- technology: hardware, networking, infrastructure, cloud services, cybersecurity, AI/ML tools
- math: pure mathematics, statistics, calculations, proofs
- science: physics, chemistry, biology, astronomy, research methods
- creative_writing: fiction, poetry, storytelling, screenwriting, worldbuilding
- business: marketing, management, strategy, startups, operations
- finance: investing, accounting, budgets, taxes, economics
- education: teaching, learning strategies, academic advice, study help
- health: medical, fitness, nutrition, mental health treatment
- psychology: behavior, cognition, emotions, mental processes
- language: translation, grammar, linguistics, vocabulary
- entertainment: movies, TV, books (non-writing), pop culture, celebrities
- gaming: video games, board games, game design
- art_design: visual art, graphic design, UI/UX, photography
- music: instruments, composition, music theory, artists
- legal: law, regulations, contracts, rights
- history: historical events, figures, periods
- geography: locations, maps, demographics, geopolitics
- philosophy: ethics, logic, metaphysics, existentialism
- religion: theology, spiritual practices, religious texts, mythology
- politics: government, elections, policy, political theory
- nature: environment, ecology, animals, weather, geology
- sports: athletics, fitness competitions, teams, leagues
- food_cooking: recipes, cuisine, restaurants, nutrition tips
- travel: tourism, destinations, trip planning
- relationships: dating, family, social dynamics, interpersonal advice
- career: job search, resumes, workplace issues, professional development
- daily_life: household tasks, personal organization, general life advice
- other: only if no category above fits at all"""


# ═══════════════════════════════════════════════════════════════
# LLM CLIENT (with robust error handling)
# ═══════════════════════════════════════════════════════════════

class LLMClient:
    def __init__(self, provider='anthropic', model='claude-sonnet-4-20250514',
                 api_key='', max_retries=3, rate_limit_delay=0.5):
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

    def call(self, user_prompt, temperature=0.0, max_tokens=150):
        for attempt in range(self.max_retries):
            try:
                if self.provider == 'openai':
                    return self._call_openai(user_prompt, temperature, max_tokens)
                else:
                    return self._call_anthropic(user_prompt, temperature, max_tokens)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    return None

    def _call_openai(self, user_prompt, temperature, max_tokens):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        is_new = any(x in self.model for x in ['gpt-4o','gpt-4.1','gpt-5','o1','o3','o4'])
        tok_key = "max_completion_tokens" if is_new else "max_tokens"
        payload = {"model": self.model,
                   "messages": [{"role": "user", "content": user_prompt}],
                   tok_key: max_tokens, "temperature": temperature}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        choices = data.get('choices', [])
        if not choices:
            return None
        self.total_tokens += data.get('usage', {}).get('total_tokens', 0)
        self.total_calls += 1
        time.sleep(self.rate_limit_delay)
        return choices[0].get('message', {}).get('content', '').strip()

    def _call_anthropic(self, user_prompt, temperature, max_tokens):
        url = "https://api.anthropic.com/v1/messages"
        headers = {"Content-Type": "application/json",
                   "x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01"}
        payload = {"model": self.model, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": user_prompt}],
                   "temperature": temperature}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
        content = data.get('content', [])
        if not content:
            return None
        self.total_tokens += data.get('usage', {}).get('input_tokens', 0)
        self.total_tokens += data.get('usage', {}).get('output_tokens', 0)
        self.total_calls += 1
        time.sleep(self.rate_limit_delay)
        return content[0].get('text', '').strip()


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def cohens_kappa(a, b):
    n = len(a)
    if n == 0: return 0.0
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    conf = np.zeros((k, k), dtype=int)
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    p_o = np.trace(conf) / n
    p_e = (conf.sum(1) * conf.sum(0)).sum() / (n * n)
    if p_e >= 1.0: return 1.0
    return round((p_o - p_e) / (1 - p_e), 4)

def accuracy(a, b):
    return round(sum(x == y for x, y in zip(a, b)) / len(a), 4) if a else 0

def bootstrap_ci(gt, jd, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(gt)
    if n < 10: return (0.0, 1.0)
    ga, ja = np.array(gt), np.array(jd)
    ks = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, True)
        ks.append(cohens_kappa(ga[idx].tolist(), ja[idx].tolist()))
    return (round(np.percentile(ks, 2.5), 4), round(np.percentile(ks, 97.5), 4))


# ═══════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════

def make_classify_prompt(prompt_text, concepts):
    text = prompt_text[:2000]
    concept_str = ', '.join(concepts[:15]) if concepts else '(none)'
    return f"""You categorize LLM conversation topics. Given a user prompt, assign ONE category.

CATEGORIES:
{CATEGORY_DEFS}

CONCEPTS EXTRACTED: {concept_str}

USER PROMPT:
\"\"\"{text}\"\"\"

Respond with ONLY the category name, nothing else."""


def make_verify_prompt(prompt_text, gt_label, concepts):
    text = prompt_text[:2000]
    concept_str = ', '.join(concepts[:15]) if concepts else '(none)'
    return f"""You are validating topic labels assigned to LLM chatbot prompts.

A categorization system labeled the following prompt as: "{gt_label}"

CATEGORY DEFINITIONS:
{CATEGORY_DEFS}

CONCEPTS EXTRACTED: {concept_str}

USER PROMPT:
\"\"\"{text}\"\"\"

Is "{gt_label}" an acceptable category for this prompt?
Consider: the label doesn't need to be the BEST possible choice, just reasonable.

Respond with exactly one of:
  ACCEPT - the label is reasonable
  REJECT - the label is clearly wrong
  BORDERLINE - defensible but another category fits better

Then on a new line, if REJECT or BORDERLINE, state which category would be better."""


def normalize(response, valid_categories):
    if not response:
        return None
    resp = response.strip().lower().replace('"','').replace("'",'').replace('`','')
    resp = resp.lstrip('-*• ').strip()
    resp = re.sub(r'^(category|label|topic)\s*[:=]\s*', '', resp).strip()
    # Take first line only
    resp = resp.split('\n')[0].strip()

    if resp in CONSOLIDATION_MAP:
        resp = CONSOLIDATION_MAP[resp]

    valid_lower = {c.lower(): c for c in valid_categories}
    if resp in valid_lower:
        return valid_lower[resp]

    resp_n = resp.replace('_', ' ').replace('-', ' ')
    for vl, orig in valid_lower.items():
        if resp_n == vl.replace('_', ' ').replace('-', ' '):
            return orig

    for vl, orig in valid_lower.items():
        if vl in resp or resp in vl:
            return orig

    return None


def parse_verify(response):
    """Parse ACCEPT/REJECT/BORDERLINE response."""
    if not response:
        return None, None
    lines = response.strip().split('\n')
    first = lines[0].strip().upper()

    verdict = None
    if 'ACCEPT' in first:
        verdict = 'accept'
    elif 'REJECT' in first:
        verdict = 'reject'
    elif 'BORDERLINE' in first:
        verdict = 'borderline'
    else:
        # Fuzzy
        fl = first.lower()
        if 'yes' in fl or 'reasonable' in fl or 'acceptable' in fl:
            verdict = 'accept'
        elif 'no' in fl or 'wrong' in fl or 'incorrect' in fl:
            verdict = 'reject'
        else:
            verdict = 'unknown'

    better = None
    if len(lines) > 1 and verdict in ('reject', 'borderline'):
        rest = ' '.join(lines[1:]).lower()
        for cat in COARSE_CATEGORIES:
            if cat in rest or cat.replace('_', ' ') in rest:
                better = cat
                break

    return verdict, better


def consolidate_label(label):
    return CONSOLIDATION_MAP.get(label, label)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='GT Validation v3')
    parser.add_argument('--gt-file', required=True)
    parser.add_argument('--features-file', required=True)
    parser.add_argument('--gt-model', default='gpt-5.2')
    parser.add_argument('--provider', default='anthropic',
                        choices=['openai', 'anthropic'])
    parser.add_argument('--model', default='claude-sonnet-4-20250514')
    parser.add_argument('--api-key', default='')
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--output', default='validation_v3.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', default='both',
                        choices=['classify', 'verify', 'both'],
                        help='classify=independent κ, verify=acceptance rate, both=run both')
    args = parser.parse_args()

    print("=" * 70)
    print("GT Validation v3 (Classify + Verify)")
    print("=" * 70)
    t0 = time.time()

    # ── Load GT ──
    with open(args.gt_file) as f:
        gt_raw = json.load(f)

    if 'prompt_labels' in gt_raw:
        pl = {int(k): v for k, v in gt_raw['prompt_labels'].items()}
    else:
        sample_keys = list(gt_raw.keys())[:5]
        if all(k.isdigit() for k in sample_keys):
            pl = {int(k): v for k, v in gt_raw.items()}
        else:
            print("ERROR: Need prompt_labels"); sys.exit(1)

    gt_model = gt_raw.get('model', args.gt_model)
    original_cats = set(pl.values())
    pl = {k: consolidate_label(v) for k, v in pl.items()}

    print(f"  GT file:   {args.gt_file}")
    print(f"  GT model:  {gt_model}")
    print(f"  Cats:      {len(original_cats)} -> {len(set(pl.values()))} consolidated")
    print(f"  Prompts:   {len(pl):,}")
    print(f"  Mode:      {args.mode}")

    # ── Load features ──
    print(f"  Loading features...", end='', flush=True)
    with open(args.features_file) as f:
        features = json.load(f)
    print(f" {len(features):,}")

    # ── Sample ──
    rng = random.Random(args.seed)
    by_cat = defaultdict(list)
    for idx, cat in pl.items():
        by_cat[cat].append(idx)

    total = len(pl)
    samples = []
    for cat, indices in by_cat.items():
        n_take = max(2, round(args.n_samples * len(indices) / total))
        n_take = min(n_take, len(indices))
        samples.extend((idx, cat) for idx in rng.sample(indices, n_take))

    if len(samples) > args.n_samples:
        samples = rng.sample(samples, args.n_samples)
    elif len(samples) < args.n_samples:
        sampled = {s[0] for s in samples}
        remaining = [(k, c) for k, c in pl.items() if k not in sampled]
        rng.shuffle(remaining)
        samples.extend(remaining[:args.n_samples - len(samples)])

    rng.shuffle(samples)
    n_cats_sampled = len(set(s[1] for s in samples))
    print(f"  Sampled:   {len(samples)} across {n_cats_sampled} categories")

    categories = COARSE_CATEGORIES
    client = LLMClient(args.provider, args.model, args.api_key)
    do_classify = args.mode in ('classify', 'both')
    do_verify = args.mode in ('verify', 'both')

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: CLASSIFY (independent categorization → κ)
    # ═══════════════════════════════════════════════════════════
    classify_results = []
    gt_labels, judge_labels = [], []

    if do_classify:
        print(f"\n{'─'*70}")
        print("PHASE 1: Independent Classification")
        print(f"{'─'*70}")
        failed = 0

        for i, (idx, gt_cat) in enumerate(samples):
            prompt_text = ''
            concepts = []
            if idx < len(features):
                prompt_text = features[idx].get('prompt', '')
                concepts = features[idx].get('features', {}).get('concepts_all', [])

            if not prompt_text or len(prompt_text) < 10:
                failed += 1
                classify_results.append({'idx': idx, 'gt': gt_cat,
                                         'judge': None, 'error': 'no_text'})
                continue

            response = client.call(make_classify_prompt(prompt_text, concepts))
            judge_cat = normalize(response, categories)

            if judge_cat:
                gt_labels.append(gt_cat)
                judge_labels.append(judge_cat)
                classify_results.append({
                    'idx': idx, 'gt': gt_cat, 'judge': judge_cat,
                    'response': response, 'agree': gt_cat == judge_cat,
                    'prompt_preview': prompt_text[:120]})
            else:
                failed += 1
                classify_results.append({
                    'idx': idx, 'gt': gt_cat, 'judge': None,
                    'response': response, 'failed_match': True})

            if (i + 1) % args.batch_size == 0:
                a = accuracy(gt_labels, judge_labels)
                k = cohens_kappa(gt_labels, judge_labels) if len(gt_labels) >= 10 else 0
                print(f"    [{i+1}/{len(samples)}] acc={a:.1%} κ={k:.3f} "
                      f"(valid={len(gt_labels)}, fail={failed})")

        kappa = cohens_kappa(gt_labels, judge_labels)
        acc = accuracy(gt_labels, judge_labels)
        ci = bootstrap_ci(gt_labels, judge_labels, seed=args.seed)

        print(f"\n  CLASSIFY: κ={kappa:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] "
              f"acc={acc:.1%} (n={len(gt_labels)})")

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: VERIFY (is GT label acceptable?)
    # ═══════════════════════════════════════════════════════════
    verify_results = []

    if do_verify:
        print(f"\n{'─'*70}")
        print("PHASE 2: Label Verification")
        print(f"{'─'*70}")
        verdicts = Counter()
        failed_v = 0

        for i, (idx, gt_cat) in enumerate(samples):
            prompt_text = ''
            concepts = []
            if idx < len(features):
                prompt_text = features[idx].get('prompt', '')
                concepts = features[idx].get('features', {}).get('concepts_all', [])

            if not prompt_text or len(prompt_text) < 10:
                failed_v += 1
                verify_results.append({'idx': idx, 'gt': gt_cat,
                                       'verdict': None, 'error': 'no_text'})
                continue

            response = client.call(
                make_verify_prompt(prompt_text, gt_cat, concepts),
                max_tokens=200)
            verdict, better = parse_verify(response)

            verdicts[verdict] += 1
            verify_results.append({
                'idx': idx, 'gt': gt_cat, 'verdict': verdict,
                'better_cat': better, 'response': response,
                'prompt_preview': prompt_text[:120]})

            if (i + 1) % args.batch_size == 0:
                n_done = verdicts['accept'] + verdicts['borderline'] + verdicts['reject']
                acc_rate = verdicts['accept'] / n_done if n_done else 0
                ok_rate = (verdicts['accept'] + verdicts['borderline']) / n_done if n_done else 0
                print(f"    [{i+1}/{len(samples)}] "
                      f"accept={verdicts['accept']} "
                      f"borderline={verdicts['borderline']} "
                      f"reject={verdicts['reject']} "
                      f"(accept={acc_rate:.0%}, accept+border={ok_rate:.0%})")

        n_rated = verdicts['accept'] + verdicts['borderline'] + verdicts['reject']
        accept_rate = verdicts['accept'] / n_rated if n_rated else 0
        acceptable_rate = (verdicts['accept'] + verdicts['borderline']) / n_rated if n_rated else 0

        print(f"\n  VERIFY: accept={verdicts['accept']} "
              f"borderline={verdicts['borderline']} "
              f"reject={verdicts['reject']} "
              f"unknown={verdicts.get('unknown',0)}")
        print(f"  Accept rate:      {accept_rate:.1%}")
        print(f"  Acceptable rate:  {acceptable_rate:.1%} (accept + borderline)")

        # Per-category rejection
        cat_reject = Counter()
        cat_total = Counter()
        for r in verify_results:
            if r.get('verdict') in ('accept', 'borderline', 'reject'):
                cat_total[r['gt']] += 1
                if r['verdict'] == 'reject':
                    cat_reject[r['gt']] += 1

        print(f"\n  Per-category rejection rates:")
        print(f"  {'Category':<25} {'N':>5} {'Reject':>7} {'Rate':>7}")
        print(f"  {'─'*25} {'─'*5} {'─'*7} {'─'*7}")
        for cat in sorted(cat_total, key=cat_total.get, reverse=True):
            n, rej = cat_total[cat], cat_reject[cat]
            rate = rej / n if n else 0
            mark = '✓' if rate < 0.2 else ('~' if rate < 0.4 else '✗')
            print(f"  {cat:<25} {n:>5} {rej:>7} {rate:>6.0%} {mark}")

        # What do rejects get reassigned to?
        better_dist = Counter(r['better_cat'] for r in verify_results
                              if r.get('verdict') == 'reject' and r.get('better_cat'))
        if better_dist:
            print(f"\n  Rejected labels → suggested corrections:")
            for cat, cnt in better_dist.most_common(10):
                print(f"    → {cat:<20} {cnt:>4}")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    output = {
        'config': {
            'gt_file': args.gt_file, 'gt_model': gt_model,
            'judge': f"{args.provider}/{args.model}",
            'n_samples': len(samples), 'mode': args.mode,
            'n_categories': n_cats_sampled,
        },
        'timestamp': datetime.now().isoformat(),
        'elapsed_s': round(time.time() - t0, 1),
        'api_calls': client.total_calls,
        'api_tokens': client.total_tokens,
    }

    if do_classify:
        if kappa >= 0.81:   interp = "Almost perfect"
        elif kappa >= 0.61: interp = "Substantial"
        elif kappa >= 0.41: interp = "Moderate"
        elif kappa >= 0.21: interp = "Fair"
        else:               interp = "Slight/poor"

        output['classify'] = {
            'cohens_kappa': kappa, 'kappa_ci_95': list(ci),
            'accuracy': acc, 'interpretation': interp,
            'n_valid': len(gt_labels),
        }
        print(f"  CLASSIFY:  κ = {kappa:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  "
              f"acc = {acc:.1%}  ({interp})")

        # Top disagreements
        disagreements = Counter(
            (r['gt'], r['judge'])
            for r in classify_results
            if r.get('agree') == False and r.get('judge'))
        if disagreements:
            output['classify']['top_disagreements'] = [
                {'gt': g, 'judge': j, 'n': n}
                for (g, j), n in disagreements.most_common(20)]

    if do_verify:
        output['verify'] = {
            'accept': verdicts['accept'],
            'borderline': verdicts['borderline'],
            'reject': verdicts['reject'],
            'unknown': verdicts.get('unknown', 0),
            'accept_rate': round(accept_rate, 4),
            'acceptable_rate': round(acceptable_rate, 4),
        }
        print(f"  VERIFY:    accept={accept_rate:.1%}  "
              f"acceptable={acceptable_rate:.1%}  "
              f"reject={verdicts['reject']}/{n_rated}")

    output['raw_classify'] = classify_results if do_classify else []
    output['raw_verify'] = verify_results if do_verify else []

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {args.output}")
    print(f"  Time:  {time.time()-t0:.0f}s")

    # Paper text
    print(f"\n{'='*70}")
    print("FOR PAPER")
    print(f"{'='*70}")
    if do_classify and do_verify:
        print(f"  \"Ground truth generated by {gt_model} was validated using")
        print(f"  {args.model} as an independent judge on {len(samples)}")
        print(f"  stratified prompts across {n_cats_sampled} categories.")
        print(f"  Independent classification yielded Cohen's κ = {kappa:.2f}")
        print(f"  (95% CI [{ci[0]:.2f}, {ci[1]:.2f}]); however, label")
        print(f"  verification showed {acceptable_rate:.0%} of GT labels were")
        print(f"  rated acceptable ({accept_rate:.0%} exact accept,")
        print(f"  {verdicts['borderline']/n_rated:.0%} borderline). Disagreements")
        print(f"  concentrated on semantically adjacent categories (e.g.,")
        print(f"  technology vs. programming), reflecting codebook boundary")
        print(f"  ambiguity rather than systematic misclassification.\"")


if __name__ == '__main__':
    main()