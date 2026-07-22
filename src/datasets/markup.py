"""
Canonical markup → segment-table loader + leak-free file-level splitting for the
synthetic denoising-pair pipeline (Phase 4 / Wave-U-Net training).

Single source of truth for what the notebooks (02/03/04) each copied locally:
parse the per-file markup JSON into a flat segment table, with the clean-target
and noise-donor predicates from the listening-informed policy
(see .agent_memory/noise_recovery_training_data.md).

Splits are by FILE/uid (NOT by segment) to avoid the segment-level leak.
NOTE: uid is 1:1 with file here (no patient-level grouping in the metadata), so
this does NOT guarantee patient-disjoint splits — a known, stated limitation.
"""

import json
import math
import random
from pathlib import Path

import pandas as pd

# noise-donor categories for the first-pass synthetic pairs (no-pathology only),
# grouped per the listening policy. talk+crying = human vocal interference.
DONOR_CATEGORIES = {
    "electric_noise": {"electric noise"},
    "vocal": {"talk", "crying"},
    "heart": {"heart"},
    "cough": {"cough"},  # transient — verify it mixes as additive before trusting
}
# excluded as donors (policy): movement, quiet sound, other, unidentified, clothes


def _has_value(x):
    return x is not None and x != "" and not (isinstance(x, float) and math.isnan(x))


def load_segment_dataframe(json_path, wav_dir):
    """Parse the markup JSON into a flat per-segment DataFrame.

    Columns: segment_id, uid, filename, wav_path, wav_exists, start, length, end,
    phase, pathology, quality, noise_level, has_pathology.
    Rows whose wav is missing are dropped.
    """
    json_path, wav_dir = Path(json_path), Path(wav_dir)
    with open(json_path, encoding="utf-8-sig") as f:  # BOM-safe (dataset_findings)
        payload = json.load(f)

    rows = []
    for fe in payload["files"]:
        filename = fe["filename"]
        wav_path = wav_dir / filename
        for idx, seg in enumerate(fe.get("markup", [])):
            row = {
                "segment_id": f"{filename}:{idx}",
                "uid": fe.get("uid"),
                "filename": filename,
                "wav_path": str(wav_path),
                "wav_exists": wav_path.exists(),
                "start": float(seg.get("start", 0.0) or 0.0),
                "length": float(seg.get("length", 0.0) or 0.0),
                "phase": seg.get("phase"),
                "pathology": seg.get("pathology"),
                "quality": seg.get("quality"),
                "noise_level": seg.get("noise_level"),
            }
            row["end"] = row["start"] + row["length"]
            row["has_pathology"] = _has_value(row["pathology"]) and row[
                "pathology"
            ] not in ("None", "none")
            rows.append(row)

    df = pd.DataFrame(rows)
    return df[df["wav_exists"]].reset_index(drop=True)


def _is_broad_noisy(row):
    """Policy's 'broad noisy' predicate (matches noise_recovery_training_data.md):
    quality present and != clean, OR noise_level present and != low."""
    q, nl = row["quality"], row["noise_level"]
    return (_has_value(q) and q != "clean") or (_has_value(nl) and nl != "low")


def clean_target_segments(df):
    """Clean breathing/pathology targets: pathology present AND not broad_noisy.

    Uses the broader 'not broad_noisy' definition (~0.77h) for more training
    signal — chosen 2026-06-03 over the stricter quality∧noise_level variant
    (~0.33h). Trade-off: a few targets may carry mild residual noise, so the
    model learns noisy→slightly-cleaner rather than →pristine; acceptable for a
    first model, and on-the-fly mixing multiplies effective variety.
    """
    broad_noisy = df.apply(_is_broad_noisy, axis=1)
    return df[df["has_pathology"] & ~broad_noisy].reset_index(drop=True)


def noise_donor_segments(df, categories=None):
    """No-pathology noise-only donor segments, tagged with a 'donor_category'.

    categories: subset of DONOR_CATEGORIES keys; default all of them.
    """
    if categories is None:
        categories = list(DONOR_CATEGORIES)
    wanted = {}
    for cat in categories:
        for q in DONOR_CATEGORIES[cat]:
            wanted[q] = cat

    sub = df[~df["has_pathology"] & df["quality"].isin(wanted)].copy()
    sub["donor_category"] = sub["quality"].map(wanted)
    return sub.reset_index(drop=True)


def build_clean_donor_splits(
    json_path, wav_dir, ratios=(0.7, 0.15, 0.15), seed=42, donor_categories=None
):
    """One uid-level split of ALL files, then derive clean targets + noise donors
    WITHIN each partition. Guarantees a file is wholly in one split → train-clean
    only ever mixes with train-donor, etc. Fully leak-free across splits.

    Returns {'clean': {train/val/test df}, 'donor': {train/val/test df}}.
    """
    df = load_segment_dataframe(json_path, wav_dir)
    file_splits = split_by_file(df, ratios=ratios, seed=seed)  # split the WHOLE table
    clean, donor = {}, {}
    for part, part_df in file_splits.items():
        clean[part] = clean_target_segments(part_df)
        donor[part] = noise_donor_segments(part_df, categories=donor_categories)
    return {"clean": clean, "donor": donor}


def split_by_file(df, ratios=(0.7, 0.15, 0.15), seed=42, group_col="uid"):
    """Deterministic train/val/test split at the FILE/uid level (no segment leak).

    All segments of a given file/uid go to exactly one split. Returns
    {'train': df, 'val': df, 'test': df}.

    NOTE: uid is file-level here, so this is file-disjoint but NOT guaranteed
    patient-disjoint (no patient grouping in the metadata) — a stated limitation.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    groups = sorted(df[group_col].dropna().unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    split_groups = {
        "train": set(groups[:n_train]),
        "val": set(groups[n_train : n_train + n_val]),
        "test": set(groups[n_train + n_val :]),
    }
    return {
        name: df[df[group_col].isin(g)].reset_index(drop=True)
        for name, g in split_groups.items()
    }
