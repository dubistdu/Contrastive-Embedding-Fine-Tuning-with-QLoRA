"""
Synthetic data evaluation: full JSON report (data_quality, diversity, bias_detection, linguistic_quality, optional real_life_matching, overall_scores).
Usage: python scripts/evaluate_synthetic_data.py --data data/dating_pairs.jsonl --output data/eval_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _json_serializable(obj):
    """Convert numpy/pandas scalars and nan to native Python for json.dump."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
        return None
    if type(obj).__name__ in ("bool_", "int64", "float64", "int32", "float32"):
        return bool(obj) if "bool" in type(obj).__name__ else (int(obj) if "int" in type(obj).__name__ else float(obj))
    return obj


def load_jsonl(path: Path) -> list:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _t1(r):
    return (r.get("text_1") or r.get("text_a") or "").strip()
def _t2(r):
    return (r.get("text_2") or r.get("text_b") or "").strip()
def _words(text):
    return re.findall(r"[a-z']+", text.lower()) if text else []


def data_quality(data: list) -> dict:
    n = len(data)
    if n == 0:
        return {"total_records": 0}

    # Completeness per field
    fields = ["text_1", "text_2", "label", "category"]
    completeness = {}
    for f in fields:
        if f == "label":
            missing = sum(1 for r in data if r.get("label") not in (0, 1))
        else:
            key = "text_1" if f == "text_1" else ("text_2" if f == "text_2" else f)
            missing = sum(1 for r in data if not (r.get(key) or r.get("text_a" if key == "text_1" else "text_b" if key == "text_2" else key)))
        completeness[f] = {"missing_count": missing, "completeness_rate": 1.0 - (missing / n) if n else 0.0}

    # Format validation
    labels = [r.get("label") for r in data if r.get("label") in (0, 1)]
    label_dist = Counter(str(int(x)) for x in labels)
    len1 = [len(_t1(r).split()) for r in data]
    len2 = [len(_t2(r).split()) for r in data]
    def stats(arr):
        if not arr: return {"mean": 0, "std": 0, "min": 0, "max": 0}
        a = [x for x in arr if x is not None]
        if not a: return {"mean": 0, "std": 0, "min": 0, "max": 0}
        return {"mean": sum(a)/len(a), "std": (sum((x - sum(a)/len(a))**2 for x in a)/len(a))**0.5, "min": min(a), "max": max(a)}
    categories = [r.get("category") or "" for r in data]
    subcategories = [r.get("subcategory") or "" for r in data]
    cat_counts = Counter(c for c in categories if c)
    subcat_counts = Counter(s for s in subcategories if s)
    format_validation = {
        "valid_labels": len(labels) == n,
        "label_distribution": dict(label_dist),
        "text_length_stats": {"text_1": stats(len1), "text_2": stats(len2)},
        "category_consistency": {
            "unique_categories": len(cat_counts),
            "unique_subcategories": len(subcat_counts),
            "category_distribution": dict(cat_counts),
            "subcategory_distribution": dict(subcat_counts),
        },
    }

    # Duplicates
    pairs = [(_t1(r), _t2(r)) for r in data]
    unique = len(set(pairs))
    duplicates = {"total_pairs": n, "unique_pairs": unique, "duplicate_rate": 1.0 - (unique / n) if n else 0.0}

    return {
        "total_records": n,
        "completeness": completeness,
        "format_validation": format_validation,
        "consistency": {},
        "duplicates": duplicates,
    }


def diversity(data: list) -> dict:
    if not data:
        return {}
    all_words = []
    for r in data:
        all_words.extend(_words(_t1(r)))
        all_words.extend(_words(_t2(r)))
    n_words = len(all_words)
    w_counts = Counter(all_words)
    unique = len(w_counts)
    vocab_richness = unique / n_words if n_words else 0
    hapax = sum(1 for c in w_counts.values() if c == 1)
    most_common = [[w, c] for w, c in w_counts.most_common(20)]
    vocabulary_diversity = {
        "total_words": n_words,
        "unique_words": unique,
        "vocabulary_richness": vocab_richness,
        "most_common_words": most_common,
        "hapax_legomena": hapax,
        "average_word_frequency": n_words / unique if unique else 0,
    }

    categories = [r.get("category") or "" for r in data if r.get("category")]
    cat_counts = Counter(categories)
    total_c = sum(cat_counts.values())
    probs = [c / total_c for c in cat_counts.values()] if total_c else []
    import math
    entropy = -sum(p * math.log2(p) for p in probs if p > 0) if probs else 0
    gini = 1 - sum(p**2 for p in probs) if probs else 0
    balance_score = min(cat_counts.values()) / max(cat_counts.values()) if cat_counts else 0
    category_distribution = {
        "category_counts": dict(cat_counts),
        "entropy": entropy,
        "gini_coefficient": gini,
        "balance_score": balance_score,
    }

    labels = [r.get("label") for r in data if r.get("label") in (0, 1)]
    n1 = sum(1 for x in labels if x == 1)
    n0 = len(labels) - n1
    label_balance = {
        "label_distribution": {"1": n1, "0": n0},
        "balance_ratio": min(n1, n0) / max(n1, n0) if max(n1, n0) else 0,
        "majority_class_percentage": max(n1, n0) / len(labels) if labels else 0,
    }

    lens = [len(_t1(r).split()) + len(_t2(r).split()) for r in data]
    avg_len = sum(lens) / len(lens) if lens else 0
    var_len = sum((x - avg_len)**2 for x in lens) / len(lens) if lens else 0
    word_lens = [len(w) for w in all_words] if all_words else [0]
    text_complexity = {
        "avg_word_count": sum(len(_t1(r).split()) + len(_t2(r).split()) for r in data) / (2 * len(data)) if data else 0,
        "avg_word_length": sum(word_lens) / len(word_lens) if word_lens else 0,
        "avg_sentence_length": avg_len / 2 if data else 0,
        "complexity_variance": var_len,
    }

    # Semantic diversity: optional embedding-based (skip on numerical issues)
    semantic_diversity = {}
    try:
        import warnings
        import numpy as np
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [_t1(r) + " " + _t2(r) for r in data[:500]]
        emb = model.encode(texts)
        emb = np.asarray(emb, dtype=np.float64)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        emb = emb / norms
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(emb)
        if not (np.isfinite(sim).all() and sim.size > 0):
            raise ValueError("Invalid similarity matrix")
        np.fill_diagonal(sim, 0)
        n_sim = sim.shape[0]
        avg_sim = float(np.nanmean(sim)) if n_sim > 1 else 0.0
        semantic_diversity = {
            "average_similarity": avg_sim,
            "similarity_std": float(np.nanstd(sim)),
        }
        import math
        from sklearn.cluster import KMeans
        k = min(10, max(2, len(emb) // 50))
        km = KMeans(n_clusters=k, random_state=42).fit(emb)
        cl_counts = Counter(km.labels_)
        n_emb = len(emb)
        semantic_diversity["cluster_entropy"] = float(-sum((c/n_emb) * math.log2(c/n_emb) for c in cl_counts.values() if c > 0))
        semantic_diversity["cluster_distribution"] = dict((str(k), int(v)) for k, v in cl_counts.items())
    except Exception:
        pass

    return {
        "vocabulary_diversity": vocabulary_diversity,
        "category_distribution": category_distribution,
        "label_balance": label_balance,
        "text_complexity": text_complexity,
        **({"semantic_diversity": semantic_diversity} if semantic_diversity else {}),
    }


def bias_detection(data: list) -> dict:
    if not data:
        return {}
    n = len(data)
    n1 = sum(1 for r in data if r.get("label") == 1)
    n0 = n - n1
    compat_rate = n1 / n if n else 0

    # Gender: every row has boy/girl in this dataset
    gender_bias = {
        "boy": {"count": n, "compatible_rate": compat_rate, "categories": dict(Counter(r.get("category") or "" for r in data if r.get("category")))},
        "girl": {"count": n, "compatible_rate": compat_rate, "categories": dict(Counter(r.get("category") or "" for r in data if r.get("category")))},
    }

    category_bias = {}
    for cat, group in [(c, [r for r in data if (r.get("category") or "") == c]) for c in set(r.get("category") or "" for r in data) if c]:
        g1 = sum(1 for r in group if r.get("label") == 1)
        rate = g1 / len(group) if group else 0
        category_bias[cat] = {"count": len(group), "compatible_rate": rate, "bias_score": abs(rate - 0.5)}

    label_bias = {
        "total_samples": n, "compatible_samples": n1, "incompatible_samples": n0,
        "compatible_rate": compat_rate, "bias_severity": abs(compat_rate - 0.5), "is_balanced": bool(0.4 <= compat_rate <= 0.6),
    }

    len_compat = [len(_t1(r)) + len(_t2(r)) for r in data if r.get("label") == 1]
    len_incompat = [len(_t1(r)) + len(_t2(r)) for r in data if r.get("label") == 0]
    m1 = sum(len_compat) / len(len_compat) if len_compat else 0
    m0 = sum(len_incompat) / len(len_incompat) if len_incompat else 0
    try:
        from scipy import stats as scipy_stats
        t = scipy_stats.ttest_ind(len_compat, len_incompat)
        length_bias = {
            "compatible_avg_length": m1, "incompatible_avg_length": m0, "length_difference": m0 - m1,
            "t_statistic": float(t.statistic), "t_pvalue": float(t.pvalue), "significant_bias": bool(t.pvalue < 0.05),
        }
    except Exception:
        length_bias = {"compatible_avg_length": m1, "incompatible_avg_length": m0, "length_difference": m0 - m1}

    compat_words = Counter()
    incompat_words = Counter()
    for r in data:
        label = r.get("label")
        if label not in (0, 1):
            continue
        for w in _words(_t1(r)) + _words(_t2(r)):
            if len(w) < 3 or w in ("the", "and", "for", "are", "but", "not", "you", "can"):
                continue
            if label == 1:
                compat_words[w] += 1
            else:
                incompat_words[w] += 1
    most_biased = {}
    for w in set(compat_words) | set(incompat_words):
        c1, c0 = compat_words[w], incompat_words[w]
        if c1 + c0 < 5:
            continue
        rate = c1 / (c1 + c0)
        most_biased[w] = {"compatible_count": c1, "incompatible_count": c0, "compatible_rate": rate, "bias_score": abs(rate - 0.5)}
    most_biased = dict(sorted(most_biased.items(), key=lambda x: -x[1]["bias_score"])[:20])
    vocabulary_bias = {
        "total_biased_words": len(most_biased),
        "most_biased_words": most_biased,
        "compatible_vocab_size": len(compat_words),
        "incompatible_vocab_size": len(incompat_words),
    }

    return {
        "gender_bias": gender_bias,
        "category_bias": category_bias,
        "label_bias": label_bias,
        "length_bias": length_bias,
        "vocabulary_bias": vocabulary_bias,
    }


def linguistic_quality(data: list) -> dict:
    if not data:
        return {}
    # Readability: approximate Flesch (syllables ~ vowel groups)
    def syllables(t):
        return max(1, len(re.findall(r"[aeiouy]+", t.lower())))
    flesch_scores = []
    coherences = []
    for r in data:
        t1, t2 = _t1(r), _t2(r)
        w1, w2 = _words(t1), _words(t2)
        s1, s2 = t1.count(".") + 1, t2.count(".") + 1
        for t, w, s in [(t1, w1, s1), (t2, w2, s2)]:
            if not w:
                continue
            syl = syllables(t)
            flesch_scores.append(206.835 - 1.015 * (len(w) / max(1, s)) - 84.6 * (syl / len(w)))
        if w1 and w2:
            coherences.append(len(set(w1) & set(w2)) / len(set(w1) | set(w2)))
    readability = {}
    if flesch_scores:
        readability = {
            "average_flesch_score": sum(flesch_scores) / len(flesch_scores),
            "readability_std": (sum((x - sum(flesch_scores)/len(flesch_scores))**2 for x in flesch_scores) / len(flesch_scores)) ** 0.5,
            "readability_range": [min(flesch_scores), max(flesch_scores)],
        }
    coherence_avg = sum(coherences) / len(coherences) if coherences else 0
    coherence_std = (sum((x - coherence_avg)**2 for x in coherences) / len(coherences)) ** 0.5 if coherences else 0
    low_coherence = sum(1 for x in coherences if x < 0.1)
    coherence = {"average_coherence": coherence_avg, "coherence_std": coherence_std, "low_coherence_pairs": low_coherence}

    contractions = sum(1 for r in data if "'" in _t1(r) or "'" in _t2(r)) / len(data) if data else 0
    informal = sum(1 for r in data if "i'm" in _t1(r).lower() or "i'm" in _t2(r).lower() or "don't" in _t1(r).lower() or "don't" in _t2(r).lower()) / len(data) if data else 0
    first_person = sum(1 for r in data if " i " in (" " + _t1(r).lower() + " ") or " i " in (" " + _t2(r).lower() + " ")) / len(data) if data else 0
    naturalness = {"contractions": contractions, "informal_words": informal, "first_person": first_person, "questions": 0.0, "exclamations": 0.0}

    starters = Counter()
    phrases = Counter()
    for r in data:
        t1, t2 = _t1(r), _t2(r)
        for t in [t1, t2]:
            if ":" in t:
                starters[t.split(":")[0].strip() + ":"] += 1
            for i in range(len(t.split()) - 1):
                phrases[" ".join(t.lower().split()[i:i+2])] += 1
    short = sum(1 for r in data if len(_t1(r).split()) + len(_t2(r).split()) < 10)
    long_ = sum(1 for r in data if len(_t1(r).split()) + len(_t2(r).split()) > 30)
    medium = len(data) - short - long_
    grammatical_patterns = {
        "top_sentence_starters": dict(starters.most_common(5)),
        "top_phrases": dict(phrases.most_common(10)),
        "sentence_structure_distribution": {"short": short, "medium": medium, "long": long_},
    }

    exact_repeat = Counter((_t1(r), _t2(r)) for r in data)
    exact_repetitions = sum(c - 1 for c in exact_repeat.values() if c > 1)
    most_repeated_texts = {}
    for (a, b), c in exact_repeat.most_common(10):
        if c > 1:
            most_repeated_texts[f"{a[:80]} | {b[:80]}"] = c
    phrase_repeat = Counter()
    for r in data:
        for t in [_t1(r), _t2(r)]:
            toks = t.lower().split()
            for i in range(len(toks) - 2):
                phrase_repeat[" ".join(toks[i:i+3])] += 1
    phrase_rep_count = sum(c - 1 for c in phrase_repeat.values() if c > 1)
    most_repeated_phrases = dict(phrase_repeat.most_common(10))
    repetition_rate = (exact_repetitions + phrase_rep_count) / (2 * len(data)) if data else 0
    repetition_analysis = {
        "exact_text_repetitions": exact_repetitions,
        "most_repeated_texts": most_repeated_texts,
        "phrase_repetitions": phrase_rep_count,
        "most_repeated_phrases": most_repeated_phrases,
        "repetition_rate": repetition_rate,
    }

    return {
        "readability": readability,
        "coherence": coherence,
        "naturalness": naturalness,
        "grammatical_patterns": grammatical_patterns,
        "repetition_analysis": repetition_analysis,
    }


def real_life_matching(data: list, reference: list | None) -> dict:
    if not reference:
        return {}
    # Vocabulary overlap
    v_syn = set()
    v_ref = set()
    for r in data:
        v_syn.update(_words(_t1(r)))
        v_syn.update(_words(_t2(r)))
    for r in reference:
        v_ref.update(_words(_t1(r)))
        v_ref.update(_words(_t2(r)))
    overlap = len(v_syn & v_ref)
    jaccard = overlap / len(v_syn | v_ref) if (v_syn | v_ref) else 0
    coverage = overlap / len(v_ref) if v_ref else 0
    vocab_overlap = {"synthetic_vocab_size": len(v_syn), "reference_vocab_size": len(v_ref), "overlap_size": overlap, "jaccard_similarity": jaccard, "coverage_of_reference": coverage}

    # Label distribution
    l_syn = Counter(str(int(r.get("label"))) for r in data if r.get("label") in (0, 1))
    l_ref = Counter(str(int(r.get("label"))) for r in reference if r.get("label") in (0, 1))
    try:
        from scipy import stats as scipy_stats
        obs = [[l_syn.get("0", 0), l_syn.get("1", 0)], [l_ref.get("0", 0), l_ref.get("1", 0)]]
        chi2, p = scipy_stats.chi2_contingency(obs)[:2]
        label_sim = {"synthetic_labels": dict(l_syn), "reference_labels": dict(l_ref), "chi2_statistic": float(chi2), "chi2_pvalue": float(p)}
    except Exception:
        label_sim = {"synthetic_labels": dict(l_syn), "reference_labels": dict(l_ref)}

    return {
        "statistical_similarity": {},
        "vocabulary_overlap": vocab_overlap,
        "category_alignment": {},
        "label_distribution_similarity": label_sim,
        "text_style_similarity": {},
    }


def overall_scores(dq: dict, div: dict, bias: dict, ling: dict) -> dict:
    # Score 0-100 per dimension (simplified)
    n = dq.get("total_records") or 0
    comp = (dq.get("completeness") or {}).get("label", {}).get("completeness_rate", 0) or (dq.get("completeness") or {}).get("text_1", {}).get("completeness_rate", 1)
    dup_rate = (dq.get("duplicates") or {}).get("duplicate_rate", 0) or 0
    dq_score = 100 * (0.5 * comp + 0.5 * (1 - min(dup_rate * 2, 1))) if n else 0

    vocab = (div.get("vocabulary_diversity") or {}).get("vocabulary_richness", 0) or 0
    bal = (div.get("label_balance") or {}).get("balance_ratio", 0) or 0
    div_score = 50 * min(vocab * 20, 1) + 50 * bal if n else 0

    lb = bias.get("label_bias") or {}
    bias_sev = lb.get("bias_severity", 0.5) or 0.5
    bias_score = 100 * (1 - min(bias_sev * 2, 1)) if n else 0

    rep = (ling.get("repetition_analysis") or {}).get("repetition_rate", 0) or 0
    coh = (ling.get("coherence") or {}).get("average_coherence", 0) or 0
    ling_score = 100 * (1 - min(rep * 5, 1)) * 0.5 + 50 * min(coh * 3, 1) if n else 0

    overall = (dq_score + div_score + bias_score + ling_score) / 4.0
    return {
        "data_quality_score": round(dq_score, 2),
        "diversity_score": round(div_score, 2),
        "bias_score": round(bias_score, 2),
        "linguistic_quality_score": round(ling_score, 2),
        "overall_score": round(overall, 2),
    }


def generate_recommendations(dq: dict, div: dict, bias: dict, ling: dict, scores: dict) -> list:
    """Produce actionable recommendations from evaluation results."""
    recs = []
    n = dq.get("total_records") or 0
    if n == 0:
        return ["No data to evaluate; add records to the dataset."]

    # Data quality
    comp = (dq.get("completeness") or {}).get("label", {}).get("completeness_rate") or (dq.get("completeness") or {}).get("text_1", {}).get("completeness_rate")
    if comp is not None and comp < 0.95:
        recs.append("Completeness is below 95%. Fill missing required fields (text_1, text_2, label, category) for all records.")
    dup_rate = (dq.get("duplicates") or {}).get("duplicate_rate", 0) or 0
    if dup_rate > 0.05:
        recs.append(f"Duplicate pair rate is {dup_rate:.1%}. Consider deduplicating before training to avoid overfitting.")
    dq_score = (scores or {}).get("data_quality_score", 100) or 100
    if dq_score < 60:
        recs.append("Data quality score is below 60%. Improve completeness and reduce duplicates to meet typical requirements.")

    # Diversity
    vocab = (div.get("vocabulary_diversity") or {}).get("vocabulary_richness", 0) or 0
    if vocab is not None and vocab < 0.05 and n > 50:
        recs.append("Vocabulary richness is low. Add more varied wording, categories, or templates to improve diversity.")
    bal = (div.get("label_balance") or {}).get("balance_ratio", 0) or 0
    if bal is not None and bal < 0.4:
        recs.append("Label balance is skewed. Aim for a balance ratio near 1.0 (e.g. add more of the minority class or downsample the majority).")
    cat_dist = (div.get("category_distribution") or {}).get("balance_score")
    if cat_dist is not None and cat_dist < 0.3 and (div.get("category_distribution") or {}).get("category_counts"):
        recs.append("Category distribution is uneven. Ensure all categories are well-represented to avoid training bias.")

    # Bias
    lb = bias.get("label_bias") or {}
    bias_sev = lb.get("bias_severity", 0) or 0
    if bias_sev > 0.2:
        recs.append(f"Label bias severity is high (compatible rate far from 0.5). Rebalance labels so neither class dominates.")
    length_bias = bias.get("length_bias") or {}
    if length_bias.get("significant_bias") is True:
        recs.append("Length bias is statistically significant. Vary sentence lengths across compatible and incompatible pairs.")
    vocab_bias = bias.get("vocabulary_bias") or {}
    if (vocab_bias.get("total_biased_words") or 0) > 10:
        recs.append("Many words are strongly associated with one label. Review or diversify wording to reduce vocabulary bias.")
    bias_score = (scores or {}).get("bias_score", 100) or 100
    if bias_score < 60:
        recs.append("Bias score is below 60%. Address label, length, or vocabulary bias before training.")

    # Linguistic quality
    rep = (ling.get("repetition_analysis") or {}).get("repetition_rate", 0) or 0
    if rep > 0.15:
        # Metric can exceed 1.0 (e.g. many phrase repeats); cap display at 100%
        pct_display = min(rep, 1.0)
        msg = "very high (100%+)" if rep > 1.0 else f"high ({pct_display:.1%})"
        recs.append(f"Repetition rate is {msg}. Reduce exact or phrase-level repetition for more natural data.")
    coh = (ling.get("coherence") or {}).get("average_coherence", 0) or 0
    if coh is not None and coh < 0.1 and n > 20:
        recs.append("Pair coherence is low. Ensure text pairs are topically related and logically consistent.")
    ling_score = (scores or {}).get("linguistic_quality_score", 100) or 100
    if ling_score < 60:
        recs.append("Linguistic quality score is below 60%. Improve readability, coherence, or reduce repetition.")

    # Overall
    overall = (scores or {}).get("overall_score", 100) or 100
    if overall < 60:
        recs.append("Overall score is below 60%. Address data quality, diversity, bias, and linguistic issues before using this dataset for training.")
    if not recs:
        recs.append("No major issues detected. Dataset meets typical thresholds for quality, diversity, bias, and linguistic quality.")

    return recs


class SyntheticDataEvaluator:
    """Evaluates synthetic data on data quality, diversity, bias, linguistic quality, and optional real-life matching."""

    def __init__(self, synthetic_data_path: str | Path, reference_data_path: str | Path | None = None):
        path = Path(synthetic_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Synthetic data not found: {path}")
        self.data = load_jsonl(path)
        self.reference = load_jsonl(Path(reference_data_path)) if reference_data_path and Path(reference_data_path).exists() else None

    def evaluate_data_quality(self) -> dict:
        """Assess completeness, consistency, duplicates, format validation."""
        return data_quality(self.data)

    def evaluate_diversity(self) -> dict:
        """Measure vocabulary richness, category balance, label balance, text complexity."""
        return diversity(self.data)

    def detect_bias(self) -> dict:
        """Find unfair patterns: gender, category, length, vocabulary bias."""
        return bias_detection(self.data)

    def evaluate_real_life_matching(self) -> dict:
        """Compare with reference data (vocabulary overlap, label distribution)."""
        return real_life_matching(self.data, self.reference)

    def evaluate_linguistic_quality(self) -> dict:
        """Assess readability, coherence, naturalness, repetition."""
        return linguistic_quality(self.data)

    def get_report(self) -> dict:
        """Run all evaluations and return full report with overall_scores and recommendations."""
        dq = self.evaluate_data_quality()
        div = self.evaluate_diversity()
        bias = self.detect_bias()
        ling = self.evaluate_linguistic_quality()
        rlm = self.evaluate_real_life_matching()
        scores = overall_scores(dq, div, bias, ling)
        recommendations = generate_recommendations(dq, div, bias, ling, scores)
        report = {
            "data_quality": dq,
            "diversity": div,
            "bias_detection": bias,
            "linguistic_quality": ling,
            **({"real_life_matching": rlm} if rlm else {}),
            "overall_scores": scores,
            "recommendations": recommendations,
        }
        return _json_serializable(report)


def main():
    parser = argparse.ArgumentParser(description="Synthetic data evaluation: full JSON report")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "dating_pairs.jsonl")
    parser.add_argument("--reference", type=Path, default=None, help="Optional reference JSONL for real_life_matching")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    try:
        evaluator = SyntheticDataEvaluator(args.data, args.reference)
    except FileNotFoundError as e:
        print(e)
        return

    report = evaluator.get_report()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
