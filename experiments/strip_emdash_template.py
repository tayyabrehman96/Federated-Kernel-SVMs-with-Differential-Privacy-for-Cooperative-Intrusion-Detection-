"""One-off: replace LaTeX --- (em dash) in template.tex with academic punctuation."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "template.tex"
text = path.read_text(encoding="utf-8")

# CRediT: use LaTeX en-dash (--) between tokens
text = text.replace("writing---original draft", "writing--original draft")
text = text.replace("writing---review and editing", "writing--review and editing")

# Comment headers: Unicode em dash
text = text.replace("\u2014", ": ")

# Table / placeholder em dashes
text = re.sub(r"(\s)&\s*---\s*\\\\", r"\1& n/a \\\\", text)
text = re.sub(r"(\s)&\s*---\s*&", r"\1& n/a &", text)

# Headings and labels: colon reads more cleanly than a comma after global replace
_heading_fixes = [
    ("\\textbf{Stage~1 --- Local certainty (edge).}", "\\textbf{Stage~1: Local certainty (edge).}"),
    ("\\textbf{Stage~2 --- Collective inference (fog).}", "\\textbf{Stage~2: Collective inference (fog).}"),
    ("\\textbf{Stage 1 --- Chunked ingestion.}", "\\textbf{Stage 1: Chunked ingestion.}"),
    ("\\textbf{Stage 2 --- Deduplication.}", "\\textbf{Stage 2: Deduplication.}"),
    ("\\textbf{Stage 3 --- Imputation and type normalisation.}", "\\textbf{Stage 3: Imputation and type normalisation.}"),
    ("\\textbf{Stage 4 --- Encoding and feature engineering.}", "\\textbf{Stage 4: Encoding and feature engineering.}"),
    ("\\textbf{Stage 5 --- Class balancing.}", "\\textbf{Stage 5: Class balancing.}"),
    ("\\textbf{Stage 6 --- Feature scaling.}", "\\textbf{Stage 6: Feature scaling.}"),
    ("\\textbf{Vector~1 --- Zone~A (volumetric DDoS).}", "\\textbf{Vector~1 (Zone~A, volumetric DDoS).}"),
    ("\\textbf{Vector~2 --- Zone~B (stealthy FDI).}", "\\textbf{Vector~2 (Zone~B, stealthy FDI).}"),
    ("\\textbf{Vector~3 --- Zone~C (port scan reconnaissance).}", "\\textbf{Vector~3 (Zone~C, port scan reconnaissance).}"),
    ("\\textbf{Zone~A --- DDoS (Stage~1):}", "\\textbf{Zone~A (DDoS, Stage~1):}"),
    ("\\textbf{Zone~B --- FDI (Stage~2):}", "\\textbf{Zone~B (FDI, Stage~2):}"),
    ("\\textbf{Zone~C --- Port scan (Stage~2):}", "\\textbf{Zone~C (Port scan, Stage~2):}"),
]
for old, new in _heading_fixes:
    text = text.replace(old, new)

# Phrases where parentheses or explicit connectives read better than an em dash
_phrase_fixes = [
    (
        "The flow signatures --- ultra-short durations, high SYN-flag counts, extreme packet rates --- are strongly anomalous",
        "The flow signatures (ultra-short durations, high SYN-flag counts, extreme packet rates) are strongly anomalous",
    ),
    ("detection threshold --- as intended", "detection threshold, as intended"),
    ("22\,ms --- demonstrating", "22\,ms, demonstrating"),
    ("FDI detection --- the most operationally", "FDI detection, which is the most operationally"),
    ("inference and offloading to the fog --- a meaningful", "inference and offloading to the fog, which is a meaningful"),
    (
        "percentage points --- a meaningful gain at the scale of 2.8 million test flows",
        "percentage points, which constitutes a meaningful gain at the scale of 2.8 million test flows",
    ),
    (
        "2.8 million test flows --- the practical operating bound",
        "2.8 million test flows; this figure defines the practical operating bound",
    ),
]
for old, new in _phrase_fixes:
    text = text.replace(old, new)

# Remaining triple hyphens (not ----): comma is usually closest to academic prose
text = re.sub(r"(?<![-])---(?![-])", ", ", text)

# Fix awkward spacing / double punctuation
fixes = [
    ("( , ", "(",),
    ("[ , ", "[",),
    (" , , ", ", ",),
    (" , ", ", ",),
    (", .", "."),
    ("..", "."),
    (", )", ")"),
    (",,", ","),
]
for a, b in fixes:
    text = text.replace(a, b)

path.write_text(text, encoding="utf-8")
print("Updated", path)
