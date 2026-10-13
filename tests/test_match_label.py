import json
from pathlib import Path

from labels import OTHER, VALID_LABELS, match_label

REPO = Path(__file__).resolve().parents[1]


def test_vocabulary_matches_eval_report():
    report = json.loads((REPO / "outputs" / "eval_report.json").read_text())["classification_report"]
    reported = {k for k in report if k not in ("accuracy", "macro avg", "weighted avg")}
    assert reported == set(VALID_LABELS)
    assert len(VALID_LABELS) == 27


def test_exact_label():
    assert match_label("get_refund") == "get_refund"


def test_case_and_whitespace_are_ignored():
    assert match_label("  Get_Refund \n") == "get_refund"


def test_label_embedded_in_a_sentence():
    assert match_label("The intent is get_refund.") == "get_refund"


def test_hallucinated_labels_map_to_other():
    # Both were real generations recorded in outputs/noisy_eval.json
    assert match_label("track_package") == OTHER
    assert match_label("enter_password") == OTHER


def test_empty_generation_maps_to_other():
    assert match_label("") == OTHER
    assert match_label("   ") == OTHER
    assert match_label(None) == OTHER
