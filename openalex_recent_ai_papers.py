import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pyalex import Concepts, Works, config


@dataclass(frozen=True)
class RunConfig:
    days: int
    max_results: int
    per_page: int
    min_concept_score: float
    concept_query: str
    output_dir: str
    openalex_email: str | None
    openalex_api_key: str | None


def _parse_args() -> RunConfig:
    p = argparse.ArgumentParser(
        description="Fetch recent OpenAlex works for the Artificial Intelligence concept."
    )
    p.add_argument(
        "--days",
        type=int,
        default=3,
        help="How many days back to include (default: 3).",
    )
    p.add_argument(
        "--max-results",
        type=int,
        default=500,
        help="Maximum works to fetch (default: 500).",
    )
    p.add_argument(
        "--per-page",
        type=int,
        default=200,
        help="Results per page for API pagination (1-200, default: 200).",
    )
    p.add_argument(
        "--min-concept-score",
        type=float,
        default=0.5,
        help="Keep works only if the AI concept score is >= this threshold (0-1, default: 0.5).",
    )
    p.add_argument(
        "--concept-query",
        default="artificial intelligence",
        help="Concept search query (default: 'artificial intelligence').",
    )
    p.add_argument(
        "--output-dir",
        default="temp",
        help="Directory to write output JSON (default: temp).",
    )
    p.add_argument(
        "--email",
        default=os.getenv("OPENALEX_EMAIL"),
        help="OpenAlex 'From' email. Defaults to OPENALEX_EMAIL env var.",
    )
    p.add_argument(
        "--api-key",
        default=os.getenv("OPENALEX_API_KEY"),
        help="OpenAlex API key. Defaults to OPENALEX_API_KEY env var.",
    )

    args = p.parse_args()
    if args.days < 1:
        raise SystemExit("--days must be >= 1")
    if args.max_results < 1:
        raise SystemExit("--max-results must be >= 1")
    if args.per_page < 1 or args.per_page > 200:
        raise SystemExit("--per-page must be between 1 and 200")
    if args.min_concept_score < 0 or args.min_concept_score > 1:
        raise SystemExit("--min-concept-score must be between 0 and 1")

    return RunConfig(
        days=args.days,
        max_results=args.max_results,
        per_page=args.per_page,
        min_concept_score=args.min_concept_score,
        concept_query=args.concept_query,
        output_dir=args.output_dir,
        openalex_email=args.email,
        openalex_api_key=args.api_key,
    )


def _configure_pyalex(cfg: RunConfig) -> None:
    if cfg.openalex_email:
        config.email = cfg.openalex_email
    if cfg.openalex_api_key:
        config.api_key = cfg.openalex_api_key

    config.max_retries = 5
    config.retry_backoff_factor = 0.4


def _pick_ai_concept(concepts: list[dict[str, Any]], query: str) -> dict[str, Any]:
    if not concepts:
        raise RuntimeError(f"No concepts found for query: {query!r}")

    normalized_query = query.strip().casefold()

    def score(c: dict[str, Any]) -> tuple[int, float]:
        name = str(c.get("display_name") or "").casefold()
        exact = 1 if name == normalized_query else 0
        level = float(c.get("level") or 999)
        works_count = float(c.get("works_count") or 0)
        # Prefer exact name match, then broader concepts (lower level), then more works.
        return (exact, -level, works_count)

    return sorted(concepts, key=score, reverse=True)[0]


def find_concept_id(concept_query: str) -> tuple[str, dict[str, Any]]:
    # Concepts is deprecated in OpenAlex, but it still exists and is supported by pyalex.
    raw = Concepts().search(concept_query).get(per_page=25)
    best = _pick_ai_concept(list(raw), concept_query)
    concept_id = best.get("id")
    if not isinstance(concept_id, str) or not concept_id:
        raise RuntimeError("OpenAlex concept search returned a concept without an 'id'")
    return concept_id, best


def fetch_recent_ai_works(
    concept_id: str,
    from_date: date,
    to_date: date,
    target_results: int,
    per_page: int,
    min_concept_score: float,
) -> list[dict[str, Any]]:
    works: list[dict[str, Any]] = []

    query = (
        Works()
        .filter(
            concepts={"id": concept_id},
            from_publication_date=from_date.isoformat(),
            to_publication_date=to_date.isoformat(),
        )
        .sort(publication_date="desc")
        .select(
            ",".join(
                [
                    "id",
                    "doi",
                    "display_name",
                    "publication_date",
                    "type",
                    "primary_location",
                    "authorships",
                    "open_access",
                    "cited_by_count",
                    "concepts",
                    "biblio",
                ]
            )
        )
    )

    def concept_score_fraction(w: dict[str, Any]) -> float | None:
        concepts = w.get("concepts")
        if not isinstance(concepts, list):
            return None
        for c in concepts:
            if not isinstance(c, dict):
                continue
            if c.get("id") != concept_id:
                continue
            score = c.get("score")
            if isinstance(score, (int, float)):
                # OpenAlex concept scores are typically 0-100.
                return float(score) / 100.0 if score > 1 else float(score)
        return None

    # Since the API can't filter by concept score, fetch extra and filter client-side.
    n_max = min(max(target_results * 20, per_page), 10000)

    for page in query.paginate(method="cursor", per_page=per_page, n_max=n_max):
        for w in page:
            wd = dict(w)
            score_frac = concept_score_fraction(wd)
            if score_frac is None or score_frac < min_concept_score:
                continue

            works.append(wd)
            if len(works) >= target_results:
                return works

    return works


def _configure_console_output() -> None:
    """
    Ensure printing never crashes due to Windows console encodings.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        # Fallback: keep default stdout; we avoid printing titles anyway.
        pass


def main() -> int:
    _configure_console_output()
    cfg = _parse_args()
    _configure_pyalex(cfg)

    concept_id, concept_obj = find_concept_id(cfg.concept_query)

    today = date.today()
    from_date = today - timedelta(days=cfg.days)
    to_date = today

    works = fetch_recent_ai_works(
        concept_id=concept_id,
        from_date=from_date,
        to_date=to_date,
        target_results=cfg.max_results,
        per_page=cfg.per_page,
        min_concept_score=cfg.min_concept_score,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"openalex_ai_works_last_{cfg.days}_days_{run_ts}.json"

    payload = {
        "run_config": asdict(cfg),
        "run_utc": run_ts,
        "from_date": from_date.isoformat(),
        "to_date": to_date.isoformat(),
        "concept_id": concept_id,
        "concept": concept_obj,
        "count": len(works),
        "results": works,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("=== OpenAlex summary ===")
    print(f"Concept ID: {concept_id}")
    print(f"Dates:   {from_date.isoformat()} .. {to_date.isoformat()}")
    print(f"Works:   {len(works)}")
    print(f"Output:  {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
