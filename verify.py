"""
verify.py — Step 4 of the pipeline (anti-edit guard).

Cerebras is forbidden from editing content (spelling fixes only). This module
PROVES that by checking every chapter/lesson title actually appears in the OCR
transcription. Each title gets a `title_verified` flag; anything the model may
have invented or paraphrased is collected into `warnings`.

We allow minor differences (spelling/OCR typo fixes are permitted), so we use a
fuzzy ratio rather than exact equality.
"""

import re
import unicodedata
from difflib import SequenceMatcher

# Titles shorter than this are matched a bit more leniently (little signal).
SIMILARITY_THRESHOLD = 0.82

# Arabic diacritics (tashkeel) to strip before comparison.
_TASHKEEL = re.compile(r"[ً-ْٰـ]")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = _TASHKEEL.sub("", text)
    # Unify Arabic alef / yaa / taa-marbuta variants.
    text = (text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
                .replace("ى", "ي").replace("ة", "ه"))
    text = text.lower()
    text = re.sub(r"[^\w؀-ۿ]+", " ", text)  # drop punctuation/dots
    return re.sub(r"\s+", " ", text).strip()


def _best_ratio(needle: str, haystack: str) -> float:
    """Best fuzzy match of `needle` against any window of `haystack`."""
    if not needle:
        return 0.0
    if needle in haystack:
        return 1.0

    words = haystack.split()
    n = len(needle.split())
    if not words:
        return 0.0

    best = 0.0
    # Slide a window roughly the size of the needle across the source.
    span = max(n, 1)
    for i in range(0, max(len(words) - span + 1, 1)):
        window = " ".join(words[i:i + span + 2])
        ratio = SequenceMatcher(None, needle, window).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.99:
                break
    # Also compare against the whole thing as a fallback.
    return max(best, SequenceMatcher(None, needle, haystack).ratio())


def verify_titles(chapters: list[dict], transcription: str) -> list[str]:
    """Mutates chapters in place adding `title_verified`; returns warnings list."""
    source = _normalize(transcription)
    warnings: list[str] = []

    def check(label: str, item: dict):
        title = item.get("title", "")
        ratio = _best_ratio(_normalize(title), source)
        verified = ratio >= SIMILARITY_THRESHOLD
        item["title_verified"] = verified
        if not verified:
            warnings.append(
                f"{label} title not found verbatim in source "
                f"(similarity {ratio:.2f}): \"{title}\""
            )

    for ci, ch in enumerate(chapters, 1):
        check(f"Chapter {ci}", ch)
        for li, ls in enumerate(ch.get("lessons", []), 1):
            check(f"Chapter {ci} / lesson {li}", ls)

    return warnings
