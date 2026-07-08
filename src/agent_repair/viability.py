from __future__ import annotations

from dataclasses import asdict, dataclass

from agent_repair.models import JSONObject

# Classifications
TOO_EASY = "TOO_EASY"
VIABLE_FOCUSED_REPAIR = "VIABLE_FOCUSED_REPAIR"
BROADLY_BROKEN = "BROADLY_BROKEN"
TARGET_FAILURE_ABSENT = "TARGET_FAILURE_ABSENT"


@dataclass(frozen=True)
class ViabilityThresholds:
    target_failure_slice_tsa_high: float = 0.90
    minimum_non_target_tsa: float = 0.75
    minimum_overall_score: float = 0.50

    def to_dict(self) -> JSONObject:
        return asdict(self)


@dataclass(frozen=True)
class ViabilityAssessment:
    classification: str
    reason: str
    overall_score: float
    target_slice: str
    target_slice_tsa: float | None
    target_slice_cases: int
    non_target_tsa: float | None
    non_target_cases: int
    thresholds: ViabilityThresholds

    def to_dict(self) -> JSONObject:
        data = asdict(self)
        data["thresholds"] = self.thresholds.to_dict()
        return data


def classify(
    *,
    overall_score: float,
    target_slice: str,
    target_slice_tsa: float | None,
    target_slice_cases: int,
    non_target_tsa: float | None,
    non_target_cases: int,
    thresholds: ViabilityThresholds,
) -> ViabilityAssessment:
    """Predeclared, development-only scenario viability heuristic.

    This is a project heuristic for deciding whether a scenario poses a meaningful
    focused repair problem. It never depends on optimizer outcomes -- only on the
    baseline's development behavior.
    """

    def build(classification: str, reason: str) -> ViabilityAssessment:
        return ViabilityAssessment(
            classification=classification,
            reason=reason,
            overall_score=overall_score,
            target_slice=target_slice,
            target_slice_tsa=target_slice_tsa,
            target_slice_cases=target_slice_cases,
            non_target_tsa=non_target_tsa,
            non_target_cases=non_target_cases,
            thresholds=thresholds,
        )

    if overall_score < thresholds.minimum_overall_score:
        return build(
            BROADLY_BROKEN,
            f"overall score {overall_score:.3f} < minimum "
            f"{thresholds.minimum_overall_score:.2f}; not a scoped repair problem",
        )
    if target_slice_cases == 0:
        return build(
            TARGET_FAILURE_ABSENT,
            f"target slice '{target_slice}' has no cases in the development splits",
        )
    if target_slice_tsa is None:
        return build(
            TARGET_FAILURE_ABSENT,
            f"target slice '{target_slice}' produced no measurable tool-selection score",
        )
    if target_slice_tsa >= 1.0:
        return build(
            TARGET_FAILURE_ABSENT,
            f"target slice TSA {target_slice_tsa:.3f} shows no seeded failures for this task model",
        )
    if target_slice_tsa > thresholds.target_failure_slice_tsa_high:
        return build(
            TOO_EASY,
            f"target slice TSA {target_slice_tsa:.3f} > "
            f"{thresholds.target_failure_slice_tsa_high:.2f}; insufficient repair headroom",
        )
    if non_target_tsa is not None and non_target_tsa < thresholds.minimum_non_target_tsa:
        return build(
            BROADLY_BROKEN,
            f"target fails (TSA {target_slice_tsa:.3f}) but non-target competence is also low "
            f"(TSA {non_target_tsa:.3f} < {thresholds.minimum_non_target_tsa:.2f})",
        )
    return build(
        VIABLE_FOCUSED_REPAIR,
        f"target slice TSA {target_slice_tsa:.3f} shows meaningful failures while non-target "
        f"competence remains adequate; focused repair headroom exists",
    )
