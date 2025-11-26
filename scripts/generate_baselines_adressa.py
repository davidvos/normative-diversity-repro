"""Generate baseline recommendation files for the Adressa dataset.

This module mirrors the functionality provided for the MIND corpus while
operating on the Adressa data layout. It can produce:

* Random baselines: shuffle the candidate list for each impression.
* Popularity baselines: rank candidates by global interaction frequency.

Usage (from repository root):

    python generate_baselines_adressa.py  \
        --val-pattern data/adressa/val/behaviors.tsv \
        --train-path data/adressa/train/behaviors_parsed.tsv \
        --output-dir data/recommendations/adressa

By default both `pop` and `random` baselines are generated. Use
`--skip-pop` or `--skip-random` to disable either output.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

DEFAULT_VAL_PATTERN = "data/adressa/val/behaviors.tsv"
DEFAULT_TRAIN_PATH = "data/adressa/train/behaviors_parsed.tsv"
DEFAULT_OUTPUT_DIR = "data/recommendations/adressa"


def _iter_behavior_rows(files: Sequence[Path]) -> Iterator[Tuple[str, str]]:
    """Yield (history_raw, candidates_raw) pairs from the provided TSV files."""
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                yield row[3], row[4]


def _iter_train_rows(path: Path) -> Iterator[Tuple[str, str]]:
    """Iterate over (history_raw, candidates_raw) pairs in the train behaviours file."""
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            history_raw = row[2]
            candidates_raw = row[3] if len(row) > 3 else ""
            yield history_raw, candidates_raw


def _normalise_tokens(raw: str) -> List[str]:
    """Return a list of whitespace-separated tokens for non-empty fields."""
    if not raw:
        return []
    stripped = raw.strip()
    if not stripped or stripped.lower() == "nan":
        return []
    return stripped.split()


def _candidate_ids(tokens: Iterable[str]) -> List[str]:
    """Extract article identifiers (before '-') from candidate tokens."""
    ids: List[str] = []
    for token in tokens:
        if not token:
            continue
        parts = token.split("-", 1)
        ids.append(parts[0])
    return ids


def _scores_from_ranks(ranks: Sequence[int]) -> List[float]:
    """Return monotonic scores aligned with provided 1-indexed ranks."""
    length = len(ranks)
    if length <= 0:
        return []
    denom = float(length)
    scores: List[float] = []
    for rank in ranks:
        try:
            value = int(rank)
        except (TypeError, ValueError):
            value = length
        value = max(1, min(value, length))
        scores.append(float(length - value + 1) / denom)
    return scores


def _write_prediction(handle, scored_handle, payload: Dict, ranks: Sequence[int]) -> None:
    json.dump(payload, handle)
    handle.write("\n")
    if scored_handle is not None:
        scored_payload = dict(payload)
        scored_payload["pred_rel"] = _scores_from_ranks(ranks)
        json.dump(scored_payload, scored_handle)
        scored_handle.write("\n")


def accumulate_popularity(
    train_path: Path,
    *,
    val_files: Sequence[Path] | None = None,
) -> Counter:
    """Build a global popularity counter using training logs only.

    `val_files` is accepted for backward compatibility but ignored; training
    interactions exclusively determine the popularity statistics.
    """
    counts: Counter = Counter()

    # Training split (different column offsets)
    for history_raw, candidates_raw in _iter_train_rows(train_path):
        for article_id in _normalise_tokens(history_raw):
            counts[article_id] += 1
        for article_id in _candidate_ids(_normalise_tokens(candidates_raw)):
            counts[article_id] += 1

    return counts


def _rank_from_order(article_ids: Sequence[str]) -> Dict[str, int]:
    """Map article identifiers to their 1-indexed positions."""
    return {article_id: position for position, article_id in enumerate(article_ids, start=1)}


def generate_random_baseline(
    val_files: Sequence[Path],
    output_path: Path,
    scored_output_path: Path,
    *,
    seed: int | None = None,
) -> None:
    """Write random baseline rankings for the provided validation files."""
    rng = random.Random(seed)
    impression_id = 0

    with output_path.open("w", encoding="utf-8") as handle, \
        scored_output_path.open("w", encoding="utf-8") as scored_handle:
        for path in val_files:
            with path.open("r", encoding="utf-8") as src:
                reader = csv.reader(src, delimiter="\t")
                for row in reader:
                    if len(row) < 5:
                        continue
                    history_ids = _normalise_tokens(row[3])
                    candidate_tokens = _normalise_tokens(row[4])
                    if not history_ids or not candidate_tokens:
                        continue

                    impression_id += 1
                    candidate_ids = _candidate_ids(candidate_tokens)
                    shuffled = candidate_ids.copy()
                    rng.shuffle(shuffled)
                    rank_map = _rank_from_order(shuffled)
                    pred_rank = [rank_map[candidate_id] for candidate_id in candidate_ids]

                    payload = {
                        "impr_index": impression_id,
                        "pred_rank": pred_rank,
                    }
                    _write_prediction(handle, scored_handle, payload, pred_rank)


def generate_pop_baseline(
    val_files: Sequence[Path],
    output_path: Path,
    scored_output_path: Path,
    *,
    popularity: Counter,
) -> None:
    """Write popularity-based baseline rankings for the provided validation files."""
    impression_id = 0

    with output_path.open("w", encoding="utf-8") as handle, \
        scored_output_path.open("w", encoding="utf-8") as scored_handle:
        for path in val_files:
            with path.open("r", encoding="utf-8") as src:
                reader = csv.reader(src, delimiter="\t")
                for row in reader:
                    if len(row) < 5:
                        continue
                    history_ids = _normalise_tokens(row[3])
                    candidate_tokens = _normalise_tokens(row[4])
                    if not history_ids or not candidate_tokens:
                        continue

                    impression_id += 1
                    candidate_ids = _candidate_ids(candidate_tokens)
                    sorted_ids = sorted(
                        candidate_ids,
                        key=lambda article_id: (-popularity.get(article_id, 0), article_id),
                    )
                    rank_map = _rank_from_order(sorted_ids)
                    pred_rank = [rank_map[candidate_id] for candidate_id in candidate_ids]

                    payload = {
                        "impr_index": impression_id,
                        "pred_rank": pred_rank,
                    }
                    _write_prediction(handle, scored_handle, payload, pred_rank)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Adressa baseline recommendations.")
    parser.add_argument("--val-pattern", type=str, default=DEFAULT_VAL_PATTERN)
    parser.add_argument("--train-path", type=Path, default=Path(DEFAULT_TRAIN_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=13, help="Random seed for the random baseline.")
    parser.add_argument(
        "--skip-random",
        action="store_true",
        help="Skip generation of the random baseline.",
    )
    parser.add_argument(
        "--skip-pop",
        action="store_true",
        help="Skip generation of the popularity baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_files = sorted(Path().glob(args.val_pattern))
    if not val_files:
        raise FileNotFoundError(f"No validation files found for pattern: {args.val_pattern}")
    if not args.train_path.exists():
        raise FileNotFoundError(f"Training behaviours file not found: {args.train_path}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    popularity: Counter | None = None
    if not args.skip_pop:
        popularity = accumulate_popularity(args.train_path, val_files=val_files)

    if not args.skip_random:
        generate_random_baseline(
            val_files,
            output_dir / "random_prediction.json",
            output_dir / "random_prediction_with_rel.json",
            seed=args.seed,
        )

    if not args.skip_pop and popularity is not None:
        generate_pop_baseline(
            val_files,
            output_dir / "pop_prediction.json",
            output_dir / "pop_prediction_with_rel.json",
            popularity=popularity,
        )


if __name__ == "__main__":
    main()
