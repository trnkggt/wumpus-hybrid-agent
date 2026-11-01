from dataclasses import dataclass


@dataclass
class EpisodeMetrics:
    outcome: str
    score: int
    steps: int
    survived: bool
    escaped: bool
    escaped_with_gold: bool
    used_arrow: bool
    risky_entries: int


def summarize_episode(result: dict, used_arrow: bool, risky_entries: int) -> EpisodeMetrics:
    outcome = result["outcome"]
    return EpisodeMetrics(
        outcome=outcome,
        score=int(result["score"]),
        steps=int(result["steps"]),
        survived=outcome != "dead",
        escaped=outcome in {"escaped_with_gold", "escaped_without_gold"},
        escaped_with_gold=outcome == "escaped_with_gold",
        used_arrow=used_arrow,
        risky_entries=risky_entries,
    )
