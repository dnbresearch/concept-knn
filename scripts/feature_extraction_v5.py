#!/usr/bin/env python3
"""
Feature Extraction V5 — Prompt-Anchored Concept Scoring via PMI.

ROOT CAUSE of bad clusters:
  Response concepts like "error message", "good idea", "same time" appear
  in responses across ALL topics, creating false bridges between unrelated
  prompts. The QDC algorithm faithfully clusters these bridges.

SOLUTION (adapted from the QDC paper's WF function, Eq. 1):
  Weight response concepts by RELEVANCE to the prompt, measured via
  Pointwise Mutual Information (PMI).

  PMI(concept, prompt_word) = log2[ P(c,w) / (P(c) * P(w)) ]
  
  High PMI: "beautifulsoup" + "python" → always co-occur → keep
  Low PMI: "error message" + "python" → co-occur by chance → remove

PIPELINE:
  Phase 1: Content filters (person names, LLM filler, geographic noise)
  Phase 2: Corpus-level PMI computation (prompt_words × response_concepts)
  Phase 3: Per-item prompt-anchored scoring + IDF weighting
  Phase 4: Document frequency filters + final output

USAGE:
  python feature_extraction_v5.py --input features_v4.json [--output features_v5.json]
  
  Runs in ~60s on 161K items (no NLP, just filtering + PMI + reweighting).
"""

import json
import math
import re
import sys
import os
import time
import argparse
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, Set, List, Tuple, Optional
import scipy.sparse as sp


# =============================================================================
# PERSON NAME FILTER
# =============================================================================

COMMON_FIRST_NAMES = {
    # Male
    'james', 'john', 'robert', 'michael', 'david', 'william', 'richard',
    'joseph', 'thomas', 'charles', 'christopher', 'daniel', 'matthew',
    'anthony', 'mark', 'donald', 'steven', 'paul', 'andrew', 'joshua',
    'kenneth', 'kevin', 'brian', 'george', 'timothy', 'ronald', 'edward',
    'jason', 'jeffrey', 'ryan', 'jacob', 'gary', 'nicholas', 'eric',
    'jonathan', 'stephen', 'larry', 'justin', 'scott', 'brandon', 'benjamin',
    'samuel', 'raymond', 'gregory', 'frank', 'alexander', 'patrick', 'jack',
    'dennis', 'jerry', 'tyler', 'aaron', 'jose', 'adam', 'nathan', 'henry',
    'peter', 'zachary', 'douglas', 'harold', 'kyle', 'noah', 'carl',
    'arthur', 'gerald', 'roger', 'keith', 'jeremy', 'terry', 'lawrence',
    'sean', 'christian', 'albert', 'jesse', 'ralph', 'roy', 'eugene',
    'randy', 'philip', 'harry', 'vincent', 'bobby', 'dylan', 'billy',
    'bruce', 'willie', 'jordan', 'dave', 'mike', 'bob', 'tom', 'alex',
    'max', 'luke', 'jake', 'ethan', 'logan', 'mason', 'liam', 'owen',
    'leo', 'eli', 'oscar', 'sam', 'ben', 'charlie', 'oliver', 'finn',
    'marcus', 'victor', 'felix', 'kai', 'cole', 'blake', 'derek',
    'chris', 'phil', 'dan', 'matt', 'steve', 'rob', 'joe', 'ken',
    'jim', 'tim', 'jeff', 'greg', 'tony', 'ray', 'troy',
    'neil', 'brad', 'chad', 'brent', 'drew', 'wade', 'dean', 'dale',
    'ross', 'glen', 'seth', 'kirk', 'kurt', 'earl', 'joel', 'ivan',
    'leon', 'hugo', 'rex', 'otto', 'hans', 'lars', 'axel', 'dirk',
    # Female
    'mary', 'patricia', 'jennifer', 'linda', 'barbara', 'elizabeth',
    'susan', 'jessica', 'sarah', 'karen', 'lisa', 'nancy', 'betty',
    'margaret', 'sandra', 'ashley', 'dorothy', 'kimberly', 'emily',
    'donna', 'michelle', 'carol', 'amanda', 'melissa', 'deborah',
    'stephanie', 'rebecca', 'sharon', 'laura', 'cynthia', 'kathleen',
    'amy', 'angela', 'shirley', 'anna', 'brenda', 'pamela', 'emma',
    'nicole', 'helen', 'samantha', 'katherine', 'christine', 'debra',
    'rachel', 'carolyn', 'janet', 'catherine', 'maria', 'heather',
    'diane', 'ruth', 'julie', 'olivia', 'joyce', 'virginia', 'victoria',
    'kelly', 'lauren', 'christina', 'joan', 'evelyn', 'judith',
    'megan', 'andrea', 'cheryl', 'hannah', 'jacqueline', 'martha',
    'gloria', 'teresa', 'ann', 'sara', 'madison', 'frances', 'kathryn',
    'janice', 'jean', 'abigail', 'alice', 'judy', 'sophia', 'grace',
    'denise', 'amber', 'doris', 'marilyn', 'danielle', 'beverly',
    'isabella', 'theresa', 'diana', 'natalie', 'brittany', 'charlotte',
    'marie', 'kayla', 'alexis', 'lori', 'lily', 'claire', 'elena',
    'maya', 'rosa', 'ivy', 'luna', 'aria', 'mia', 'zoe', 'chloe',
    'ruby', 'stella', 'hazel', 'nora', 'violet',
    'tara', 'jill', 'beth', 'dawn', 'gail', 'faye', 'jade', 'sage',
    'hope', 'faith', 'joy', 'misty', 'candy', 'holly', 'april', 'june',
    'fiona', 'elsa', 'gwen', 'iris', 'lena', 'nina', 'tina', 'vera',
    'wendy', 'yvonne', 'zelda', 'freya', 'astrid', 'ingrid', 'sasha',
}

FICTION_NAMES = {
    'enchanted', 'whispered', 'murmured', 'exclaimed',
    'narrator', 'protagonist', 'antagonist', 'hero', 'heroine',
    'sir', 'lady', 'lord', 'prince', 'princess', 'king', 'queen',
}


# =============================================================================
# LLM OUTPUT FILLER
# =============================================================================

LLM_FILLER = {
    # Claude/GPT hedging & response patterns
    'good', 'great', 'nice', 'fine', 'excellent', 'wonderful', 'fantastic',
    'amazing', 'awesome', 'perfect', 'interesting', 'important',
    'consider', 'consider following', 'consider using',
    'further questions', 'further assistance', 'feel free',
    'detailed explanation', 'detailed answer', 'comprehensive guide',
    'correct answer', 'right answer', 'best answer',
    'perfect choice', 'great choice', 'good choice',
    'quick summary', 'brief overview', 'short answer',
    'happy help', 'glad help', 'hope helps', 'hope helpful',
    'let know', 'let me know', 'don hesitate',
    'first column', 'second column', 'third column',
    'first step', 'second step', 'third step', 'next step',
    'key point', 'key points', 'main point', 'main points',
    'key takeaway', 'key takeaways', 'main takeaway',
    'important note', 'important thing', 'keep mind',
    'real world', 'real life', 'day life',
    'wide range', 'broad range', 'large number',
    'various types', 'different types', 'many types',
    'long run', 'short term', 'long term',
    'best practice', 'best practices', 'common practice',
    'well known', 'widely used', 'commonly used',
    'high quality', 'low cost', 'high performance',
    'make sure', 'take look', 'take care',
    'play role', 'play important role', 'crucial role',
    'provide information', 'gain insight',
    'read', 'agree', 'finally', 'update', 'guide',
    'pro con', 'pros cons',
    # Code-response patterns
    'previous response', 'following code', 'following example',
    'code snippet', 'code block', 'code above', 'code below',
    'updated version', 'modified version', 'updated code', 'modified code',
    'original code', 'complete code', 'full code', 'sample code',
    'example code', 'following output', 'expected output',
    'return value', 'return type',
    # Politeness/meta
    'additional information', 'good luck', 'remember', 'best',
    'alternatively', 'instead', 'therefore',
    'furthermore', 'moreover', 'additionally', 'consequently',
    'nevertheless', 'specifically', 'essentially', 'basically',
    'particularly', 'significantly', 'effectively', 'efficiently',
    'simple words', 'simple terms', 'plain english',
    'easy understand', 'easy way', 'simple way',
    'common example', 'typical example', 'real example',
    'main idea', 'main concept', 'main goal', 'main purpose',
    'main advantage', 'main disadvantage', 'main difference',
    'right approach', 'good approach', 'better approach',
    'important factor', 'key factor', 'major factor',
    'common mistake', 'common error', 'common issue',
    'potential issue', 'potential problem',
    'good idea', 'error message', 'same time', 'one way',
    'these steps', 'specific needs', 'specific requirements',
    'following command', 'total number', 'code example',
    'ultimately',
    # Formatting artifacts
    'bullet point', 'bullet points', 'numbered list',
    'table format', 'markdown', 'bold text',
}

LLM_FILLER_PATTERNS = [
    r'^(here|there)\s+(is|are|was|were)\b',
    r'^(you|we|i)\s+(can|could|should|may|might)\b',
    r'\b(straightforward|comprehensive|robust|scalable)\b',
    r'^(please|kindly|simply)\s+',
    r'^(following|above|below|updated|modified|complete|sample|original)\s+(code|version|example|output|implementation|snippet)',
    r'^(code|version|example)\s+(above|below|snippet|block)',
    r'^(additional|basic|simple|common|typical|general|specific|particular)\s+(information|example|case|way|approach|method)',
    r'^(good|great|best|right|better|correct|proper)\s+(luck|choice|approach|way|practice|answer)',
    r'^(key|main|important|major|critical|crucial|significant)\s+(point|factor|takeaway|difference|advantage|concept|idea|thing|note)',
    r'^(potential|possible|common)\s+(issue|problem|error|mistake|solution)',
]


# =============================================================================
# GEOGRAPHIC NOISE + LOW QUALITY
# =============================================================================

GEO_NOISE_WORDS = {
    'united', 'kingdom', 'states', 'america', 'european',
    'american', 'british', 'english', 'french', 'german',
    'chinese', 'japanese', 'indian', 'african', 'asian',
    'western', 'eastern', 'northern', 'southern',
    'york', 'london', 'california', 'texas',
    'india', 'china', 'japan', 'canada', 'australia',
    'europe', 'brazil', 'russia', 'mexico', 'italy',
    'spain', 'germany', 'france', 'korea', 'africa',
    'global', 'worldwide', 'international', 'national',
    'city', 'country', 'region', 'continent',
}

SINGLE_WORD_NOISE = {
    'set', 'get', 'run', 'use', 'add', 'put', 'let', 'try',
    'new', 'old', 'big', 'top', 'end', 'way', 'day', 'lot',
    'bit', 'part', 'case', 'base', 'line', 'time', 'place',
    'work', 'call', 'need', 'help', 'show', 'make', 'take',
    'give', 'find', 'keep', 'tell', 'come', 'look', 'know',
    'turn', 'move', 'play', 'live', 'feel', 'hold', 'bring',
    'happen', 'start', 'begin', 'change', 'follow', 'include',
    'continue', 'provide', 'require', 'ensure',
    'value', 'number', 'level', 'type', 'form', 'kind',
    'area', 'point', 'fact', 'hand', 'side', 'head',
}


# =============================================================================
# PROMPT WORD EXTRACTION (for PMI)
# =============================================================================

_PROMPT_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'was', 'are', 'this', 'that',
    'be', 'as', 'not', 'do', 'has', 'have', 'had', 'can', 'could', 'would',
    'should', 'will', 'shall', 'may', 'might', 'must', 'about', 'above',
    'after', 'again', 'all', 'also', 'am', 'any', 'because', 'been',
    'before', 'being', 'between', 'both', 'did', 'does', 'doing', 'down',
    'during', 'each', 'few', 'he', 'her', 'here', 'hers', 'herself',
    'him', 'himself', 'his', 'how', 'if', 'into', 'its', 'itself',
    'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'now',
    'off', 'once', 'only', 'other', 'our', 'out', 'own', 'same', 'she',
    'so', 'some', 'such', 'than', 'their', 'them', 'then', 'there',
    'these', 'they', 'those', 'through', 'too', 'under', 'until', 'up',
    'us', 'very', 'we', 'what', 'when', 'where', 'which', 'while', 'who',
    'whom', 'why', 'you', 'your', 'yours', 'yourself',
    # Conversational meta
    'please', 'write', 'explain', 'describe', 'tell', 'give', 'show',
    'make', 'create', 'help', 'need', 'want', 'like', 'know',
    'using', 'used', 'following', 'based', 'example', 'code',
    'ok', 'okay', 'yes', 'thanks', 'thank', 'hi', 'hello',
    'sure', 'sorry', 'right', 'well', 'actually', 'really',
}


def extract_prompt_words(prompt: str) -> Set[str]:
    """Extract content words from prompt for PMI computation."""
    if not prompt:
        return set()
    words = set(re.findall(r'[a-z]{2,}', prompt.lower()))
    words -= _PROMPT_STOPWORDS
    return words


# =============================================================================
# V5 PROCESSOR
# =============================================================================

class FeatureV5Processor:
    """
    Post-processes V4 features using prompt-anchored concept scoring.
    
    Pipeline:
      Phase 1: Content filters (names, filler, geographic, low-quality)
      Phase 2: Corpus-level PMI(prompt_word, response_concept)
      Phase 3: Per-item scoring: relevance = PMI × IDF, with prompt boost
      Phase 4: DF filters + top-K selection + output
    """
    
    def __init__(self,
                 min_df: int = 5,
                 max_df_pct: float = 0.03,
                 max_concepts_per_prompt: int = 15,
                 prompt_boost: float = 3.0,
                 min_pmi: float = 0.0,
                 min_idf: float = 1.5):
        self.min_df = min_df
        self.max_df_pct = max_df_pct
        self.max_concepts_per_prompt = max_concepts_per_prompt
        self.prompt_boost = prompt_boost
        self.min_pmi = min_pmi
        self.min_idf = min_idf
        self.stats = defaultdict(int)
    
    def _filter_concept(self, concept: str, is_from_prompt: bool) -> Optional[str]:
        """Returns None if OK, or string reason for removal."""
        c = concept.lower().strip()
        words = c.split()
        
        # Person names
        if len(words) == 1 and words[0] in COMMON_FIRST_NAMES:
            return 'person_name'
        if len(words) == 2 and words[0] in COMMON_FIRST_NAMES:
            if not any(words[1].endswith(s) for s in
                      ('ing', 'tion', 'ment', 'ness', 'ity', 'ics', 'ism')):
                return 'person_name'
        if c in FICTION_NAMES:
            return 'person_name'
        
        # LLM filler
        if c in LLM_FILLER:
            return 'llm_filler'
        for pat in LLM_FILLER_PATTERNS:
            if re.search(pat, c):
                return 'llm_filler'
        
        # Geographic noise (response only)
        if not is_from_prompt and len(words) == 1 and words[0] in GEO_NOISE_WORDS:
            return 'geographic_noise'
        
        # Low quality
        if len(c) < 3:
            return 'low_quality'
        if c.replace(' ', '').replace('.', '').replace('-', '').isdigit():
            return 'low_quality'
        if c in SINGLE_WORD_NOISE:
            return 'low_quality'
        
        return None
    
    def process(self, data: List[Dict], verbose: bool = True) -> List[Dict]:
        t_total = time.time()
        n = len(data)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"FEATURE EXTRACTION V5 — Prompt-Anchored Concept Scoring")
            print(f"  {n:,} items | PMI-based response concept filtering")
            print(f"{'='*70}")
        
        # ══════════════════════════════════════════════════════════
        # PHASE 1: Content filtering
        # ══════════════════════════════════════════════════════════
        if verbose:
            print(f"\n  Phase 1: Content filtering...")
        
        item_prompt_concepts = []
        item_response_concepts = []
        item_prompt_words = []
        
        for item in data:
            features = item.get('features', {})
            p_concepts = set(features.get('concepts_prompt', []))
            r_concepts = set(features.get('concepts_response', []))
            
            kept_p = {c for c in p_concepts
                      if not self._filter_concept(c, is_from_prompt=True)}
            kept_r = {c for c in r_concepts
                      if not self._filter_concept(c, is_from_prompt=False)}
            
            # Count removals
            for c in p_concepts - kept_p:
                reason = self._filter_concept(c, is_from_prompt=True)
                self.stats[f'removed_{reason}'] += 1
            for c in r_concepts - kept_r:
                reason = self._filter_concept(c, is_from_prompt=False)
                self.stats[f'removed_{reason}'] += 1
            
            p_words = extract_prompt_words(item.get('prompt', ''))
            
            item_prompt_concepts.append(kept_p)
            item_response_concepts.append(kept_r)
            item_prompt_words.append(p_words)
        
        if verbose:
            for reason, count in sorted(self.stats.items(),
                                        key=lambda x: x[1], reverse=True)[:8]:
                print(f"    {reason:30s} {count:,}")
            n_resp_concepts = len(set().union(*item_response_concepts))
            n_prompt_concepts = len(set().union(*item_prompt_concepts))
            print(f"    Unique prompt concepts: {n_prompt_concepts:,}")
            print(f"    Unique response concepts: {n_resp_concepts:,}")
        
        # ══════════════════════════════════════════════════════════
        # PHASE 2: Corpus-level PMI
        # ══════════════════════════════════════════════════════════
        if verbose:
            print(f"\n  Phase 2: Computing PMI(prompt_word, response_concept)...")
        t_pmi = time.time()
        
        # Build vocabularies
        all_pw = sorted(set().union(*item_prompt_words))
        all_rc = sorted(set().union(*item_response_concepts))
        pw2idx = {w: i for i, w in enumerate(all_pw)}
        rc2idx = {c: i for i, c in enumerate(all_rc)}
        n_pw = len(all_pw)
        n_rc = len(all_rc)
        
        if verbose:
            print(f"    Prompt words: {n_pw:,}")
            print(f"    Response concepts: {n_rc:,}")
        
        # Count marginals and co-occurrences
        pw_counts = np.zeros(n_pw, dtype=np.float64)
        rc_counts = np.zeros(n_rc, dtype=np.float64)
        cooc_rows, cooc_cols = [], []
        
        for i in range(n):
            pw_idxs = np.array([pw2idx[w] for w in item_prompt_words[i] if w in pw2idx],
                               dtype=np.int32)
            rc_idxs = np.array([rc2idx[c] for c in item_response_concepts[i] if c in rc2idx],
                               dtype=np.int32)
            
            if len(pw_idxs) > 0:
                pw_counts[pw_idxs] += 1
            if len(rc_idxs) > 0:
                rc_counts[rc_idxs] += 1
            
            # Cartesian product of indices (vectorized)
            if len(pw_idxs) > 0 and len(rc_idxs) > 0:
                r = np.repeat(pw_idxs, len(rc_idxs))
                c = np.tile(rc_idxs, len(pw_idxs))
                cooc_rows.append(r)
                cooc_cols.append(c)
        
        if cooc_rows:
            all_rows = np.concatenate(cooc_rows)
            all_cols = np.concatenate(cooc_cols)
        else:
            all_rows = np.array([], dtype=np.int32)
            all_cols = np.array([], dtype=np.int32)
        
        # Sparse co-occurrence matrix (auto-sums duplicates)
        cooc = sp.csr_matrix(
            (np.ones(len(all_rows), dtype=np.float64),
             (all_rows, all_cols)),
            shape=(n_pw, n_rc))
        
        if verbose:
            print(f"    Co-occurrence nnz: {cooc.nnz:,} [{time.time()-t_pmi:.1f}s]")
        
        # Compute Positive PMI (PPMI)
        # PMI(w,c) = log2[ P(w,c) / (P(w)*P(c)) ]
        #          = log2[ cooc(w,c)*N / (pw_count(w)*rc_count(c)) ]
        if verbose:
            print(f"    Computing PPMI...")
        t_pmi2 = time.time()
        
        coo = cooc.tocoo()
        
        # Vectorized PPMI: log2(joint * N / (pw_count * rc_count)), clamped to 0
        joint = coo.data
        pw_c = pw_counts[coo.row]
        rc_c = rc_counts[coo.col]
        denom = pw_c * rc_c
        # Avoid division by zero
        valid = denom > 0
        pmi_vals = np.zeros(len(joint), dtype=np.float64)
        pmi_vals[valid] = np.maximum(0, np.log2(joint[valid] * n / denom[valid]))
        
        pmi_matrix = sp.csr_matrix(
            (pmi_vals, (coo.row, coo.col)), shape=(n_pw, n_rc))
        pmi_matrix.eliminate_zeros()
        
        if verbose:
            nz = np.count_nonzero(pmi_vals)
            pos_vals = pmi_vals[pmi_vals > 0]
            print(f"    PPMI entries: {nz:,} / {len(pmi_vals):,}")
            if len(pos_vals) > 0:
                print(f"    PPMI range: {pos_vals.min():.2f} — {pos_vals.max():.2f}, "
                      f"median={np.median(pos_vals):.2f}")
            print(f"    [{time.time()-t_pmi2:.1f}s]")
        
        # Build dict-of-dicts for fast lookup: concept_idx → {pw_idx: pmi}
        # Much faster than sparse matrix element access in Phase 4
        concept_pmi_dict = defaultdict(dict)
        for idx in range(len(coo.data)):
            if pmi_vals[idx] > 0:
                concept_pmi_dict[coo.col[idx]][coo.row[idx]] = pmi_vals[idx]
        
        if verbose:
            print(f"    PMI dict: {len(concept_pmi_dict):,} concepts with PMI entries")
        
        # ══════════════════════════════════════════════════════════
        # PHASE 3: DF filter + IDF
        # ══════════════════════════════════════════════════════════
        if verbose:
            print(f"\n  Phase 3: Document frequency filtering + IDF...")
        
        all_concepts = set()
        concept_df = Counter()
        for i in range(n):
            combined = item_prompt_concepts[i] | item_response_concepts[i]
            all_concepts.update(combined)
            for c in combined:
                concept_df[c] += 1
        
        max_df = int(n * self.max_df_pct)
        surviving = {c for c, df in concept_df.items()
                     if self.min_df <= df <= max_df}
        
        idf = {c: math.log(n / concept_df[c]) for c in surviving}
        surviving = {c for c in surviving if idf.get(c, 0) >= self.min_idf}
        
        too_rare = sum(1 for c, df in concept_df.items() if df < self.min_df)
        too_common = sum(1 for c, df in concept_df.items() if df > max_df)
        
        if verbose:
            print(f"    min_df={self.min_df}, max_df={max_df} ({self.max_df_pct:.0%})")
            print(f"    Removed {too_rare:,} too-rare, {too_common:,} too-common")
            print(f"    After min_idf={self.min_idf}: {len(surviving):,} concepts")
        
        # ══════════════════════════════════════════════════════════
        # PHASE 4: Per-item prompt-anchored scoring
        # ══════════════════════════════════════════════════════════
        if verbose:
            print(f"\n  Phase 4: Prompt-anchored scoring...")
        t_score = time.time()
        
        concepts_per_prompt = []
        pmi_kept = 0
        pmi_removed = 0
        
        for i in range(n):
            p_concepts = item_prompt_concepts[i] & surviving
            r_concepts = item_response_concepts[i] & surviving
            p_words = item_prompt_words[i]
            
            concept_scores = {}
            
            # Prompt concepts: always keep, boosted
            for c in p_concepts:
                concept_scores[c] = idf.get(c, 0) * self.prompt_boost
            
            # Pre-compute prompt word indices for PMI lookups
            pw_idxs = [pw2idx[w] for w in p_words if w in pw2idx]
            
            # Response concepts: score by PMI relevance to prompt
            for c in r_concepts:
                if c in concept_scores:
                    continue  # already scored as prompt concept
                
                c_idf = idf.get(c, 0)
                if c_idf == 0:
                    continue
                
                ci = rc2idx.get(c)
                if ci is None:
                    continue
                
                # Max PMI with any prompt word — dict lookup (O(1) per pair)
                max_pmi = 0.0
                ci_pmis = concept_pmi_dict.get(ci)
                if ci_pmis and pw_idxs:
                    for wi in pw_idxs:
                        v = ci_pmis.get(wi, 0.0)
                        if v > max_pmi:
                            max_pmi = v
                
                # Direct word overlap: concept word ∈ prompt words
                word_overlap = bool(set(c.lower().split()) & p_words)
                
                # Decision: keep if PMI > threshold OR word overlap
                if max_pmi > self.min_pmi or word_overlap:
                    overlap_bonus = 2.0 if word_overlap else 1.0
                    pmi_score = max(max_pmi, 0.5) if word_overlap else max_pmi
                    concept_scores[c] = pmi_score * c_idf * overlap_bonus
                    pmi_kept += 1
                else:
                    pmi_removed += 1
            
            # Top-K by score
            if len(concept_scores) > self.max_concepts_per_prompt:
                top = sorted(concept_scores.items(), key=lambda x: x[1],
                           reverse=True)[:self.max_concepts_per_prompt]
                concept_scores = dict(top)
            
            kept_all = set(concept_scores.keys())
            
            # Normalize to sum to 1
            total = sum(concept_scores.values())
            if total > 0:
                concept_weights = {c: round(s / total, 6)
                                  for c, s in concept_scores.items()}
            else:
                concept_weights = {}
            
            data[i]['features']['concepts_all'] = sorted(kept_all)
            data[i]['features']['concepts_prompt'] = sorted(
                item_prompt_concepts[i] & kept_all)
            data[i]['features']['concepts_response'] = sorted(
                item_response_concepts[i] & kept_all)
            data[i]['features']['concept_weights'] = concept_weights
            
            concepts_per_prompt.append(len(kept_all))
        
        if verbose:
            print(f"    Response concepts kept by PMI: {pmi_kept:,}")
            print(f"    Response concepts removed by PMI: {pmi_removed:,}")
            print(f"    PMI filter rate: {pmi_removed/(pmi_kept+pmi_removed)*100:.1f}%"
                  if (pmi_kept + pmi_removed) > 0 else "")
            print(f"    [{time.time()-t_score:.1f}s]")
        
        # ══════════════════════════════════════════════════════════
        # SUMMARY
        # ══════════════════════════════════════════════════════════
        total_time = time.time() - t_total
        n_empty = sum(1 for c in concepts_per_prompt if c == 0)
        
        # Recount final surviving concepts
        final_df = Counter()
        for item in data:
            for c in item.get('features', {}).get('concepts_all', []):
                final_df[c] += 1
        
        if verbose:
            print(f"\n{'─'*60}")
            print(f"  V5 SUMMARY (Prompt-Anchored)")
            print(f"{'─'*60}")
            print(f"  Items: {n:,}")
            print(f"  Final unique concepts: {len(final_df):,}")
            print(f"  Concepts/prompt: mean={np.mean(concepts_per_prompt):.1f}, "
                  f"median={np.median(concepts_per_prompt):.0f}, "
                  f"max={max(concepts_per_prompt) if concepts_per_prompt else 0}")
            print(f"  Empty prompts: {n_empty:,} ({n_empty/n*100:.1f}%)")
            print(f"  Time: {total_time:.1f}s")
            
            print(f"\n  Top 20 concepts by final document frequency:")
            for c, df in final_df.most_common(20):
                i_val = idf.get(c, 0)
                print(f"    {c:35s} df={df:6,}  idf={i_val:.2f}")
            
            # PMI examples
            print(f"\n  PMI filtering examples:")
            shown = 0
            for i in range(min(n, 500)):
                r_surviving = item_response_concepts[i] & surviving
                r_kept = set(data[i]['features']['concepts_response'])
                r_removed = r_surviving - r_kept - item_prompt_concepts[i]
                if r_kept and r_removed:
                    p_text = data[i].get('prompt', '')[:70]
                    print(f"    Prompt: \"{p_text}\"")
                    print(f"      ✓ Kept:    {sorted(r_kept)[:6]}")
                    print(f"      ✗ Removed: {sorted(r_removed)[:6]}")
                    shown += 1
                    if shown >= 3:
                        break
        
        return data


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='V5 Feature Post-processor — Prompt-Anchored Concept Scoring')
    parser.add_argument('--input', '-i', required=True,
                       help='V4 features JSON file')
    parser.add_argument('--output', '-o', default=None,
                       help='Output V5 features JSON')
    parser.add_argument('--min-df', type=int, default=5)
    parser.add_argument('--max-df-pct', type=float, default=0.03)
    parser.add_argument('--max-concepts', type=int, default=15)
    parser.add_argument('--prompt-boost', type=float, default=3.0)
    parser.add_argument('--min-pmi', type=float, default=0.0,
                       help='Min PMI for response concepts (default: 0.0)')
    parser.add_argument('--min-idf', type=float, default=1.5)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    print(f"Loading V4 features from {args.input}...")
    t0 = time.time()
    with open(args.input) as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} items [{time.time()-t0:.1f}s]")
    
    if 'concepts_prompt' not in data[0].get('features', {}):
        print("ERROR: Needs V4 features with concepts_prompt/concepts_response")
        sys.exit(1)
    
    processor = FeatureV5Processor(
        min_df=args.min_df,
        max_df_pct=args.max_df_pct,
        max_concepts_per_prompt=args.max_concepts,
        prompt_boost=args.prompt_boost,
        min_pmi=args.min_pmi,
        min_idf=args.min_idf,
    )
    
    data = processor.process(data, verbose=True)
    
    if not args.dry_run:
        if args.output:
            output_path = args.output
        else:
            base = os.path.splitext(args.input)[0]
            if '_v4' in base:
                output_path = base.replace('_v4', '_v5') + '.json'
            elif '_v5' in base:
                output_path = base.rsplit('_v5', 1)[0] + '_v5.json'
            else:
                output_path = base + '_v5.json'
        
        print(f"\nSaving to {output_path}...")
        t0 = time.time()
        with open(output_path, 'w') as f:
            json.dump(data, f, separators=(',', ':'))
        fsize = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved {fsize:.1f} MB [{time.time()-t0:.1f}s]")
        print(f"  ✓ Use: python qdc_clustering.py --features-file {output_path}")
    else:
        print("\n  [Dry run — not saved]")


if __name__ == '__main__':
    main()