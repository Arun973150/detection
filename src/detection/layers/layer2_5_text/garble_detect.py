"""Garbled text detection for AI-generated image analysis.

AI generators frequently produce nonsensical, garbled, or impossible text.
This module analyzes extracted text for signs of AI generation.
"""

import math
import re
import string
import unicodedata
from collections import Counter

from detection.layers.layer2_5_text.ocr_engine import TextRegion


def analyze_garble(regions: list[TextRegion], confidence_threshold: float = 0.3) -> dict:
    """Analyze text regions for garbled/nonsensical content.

    Returns:
        Dict with garble analysis results including score and details.
    """
    if not regions:
        return {"score": 0.0, "garbled_regions": 0, "total_regions": 0}

    garbled_count = 0
    low_confidence_count = 0
    details = []

    for region in regions:
        text = region.text.strip()
        if not text:
            continue

        is_garbled = False
        reasons = []

        # Check 1: Very low OCR confidence
        if region.confidence < confidence_threshold:
            low_confidence_count += 1
            reasons.append("low_ocr_confidence")

        # Check 2: Mixed scripts (e.g., Latin + CJK in same word)
        if _has_mixed_scripts(text):
            is_garbled = True
            reasons.append("mixed_scripts")

        # Check 3: High character entropy (random characters)
        entropy = _char_entropy(text)
        if len(text) > 3 and entropy > 4.0:
            is_garbled = True
            reasons.append("high_entropy")

        # Check 4: Excessive non-alphanumeric characters
        if len(text) > 2:
            alnum_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
            if alnum_ratio < 0.5:
                is_garbled = True
                reasons.append("excessive_symbols")

        # Check 5: Repeated characters (stuttering artifacts)
        if _has_excessive_repeats(text):
            is_garbled = True
            reasons.append("character_repetition")

        # Check 6: No vowels in long text (likely not real words)
        if len(text) > 4 and not _has_vowels(text) and text.isalpha():
            is_garbled = True
            reasons.append("no_vowels")

        if is_garbled:
            garbled_count += 1

        details.append({
            "text": text[:50],
            "confidence": round(region.confidence, 3),
            "garbled": is_garbled,
            "entropy": round(entropy, 3),
            "reasons": reasons,
        })

    total = len([r for r in regions if r.text.strip()])
    if total == 0:
        return {"score": 0.0, "garbled_regions": 0, "total_regions": 0}

    garble_ratio = garbled_count / total
    low_conf_ratio = low_confidence_count / total

    # Combined score: garbled text + low confidence
    score = min(1.0, garble_ratio * 0.7 + low_conf_ratio * 0.3)

    return {
        "score": round(score, 4),
        "garbled_regions": garbled_count,
        "low_confidence_regions": low_confidence_count,
        "total_regions": total,
        "garble_ratio": round(garble_ratio, 4),
        "details": details,
    }


def _has_mixed_scripts(text: str) -> bool:
    """Check if text contains characters from multiple Unicode scripts."""
    scripts = set()
    for char in text:
        if char.isalpha():
            cat = unicodedata.category(char)
            name = unicodedata.name(char, "").split()[0] if unicodedata.name(char, "") else ""
            scripts.add(name)
    return len(scripts) > 2


def _char_entropy(text: str) -> float:
    """Compute Shannon entropy of character distribution."""
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _has_excessive_repeats(text: str) -> bool:
    """Check for stuttering artifacts (e.g., 'hellooo', 'aaabbb')."""
    if len(text) < 4:
        return False
    # Check for 3+ consecutive identical characters
    return bool(re.search(r"(.)\1{2,}", text))


def _has_vowels(text: str) -> bool:
    """Check if text contains vowels."""
    return bool(set(text.lower()) & set("aeiou"))
