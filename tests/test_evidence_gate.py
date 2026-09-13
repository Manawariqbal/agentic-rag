
import pytest

from app.rag.evidence_gate import EvidenceGate


def make_result(score):
    return {
        "text": "Employees receive 20 days of annual leave.",
        "score": 0.70,
        "rerank_score": score,
        "metadata": {
            "source": "leave_and_attendance_policy.pdf",
            "section": "Entitlement",
        },
    }


def test_evidence_gate_rejects_empty_results():
    gate = EvidenceGate(threshold=0.48)

    result = gate.evaluate([])

    assert result.sufficient is False
    assert result.score == 0.0
    assert result.results == []
    assert "No relevant evidence" in result.reason


def test_evidence_gate_rejects_low_score():
    gate = EvidenceGate(threshold=0.48)

    results = [make_result(0.30)]

    result = gate.evaluate(results)

    assert result.sufficient is False
    assert result.score == pytest.approx(0.30)
    assert result.results == results
    assert "below" in result.reason.lower()


def test_evidence_gate_accepts_high_score():
    gate = EvidenceGate(threshold=0.48)

    results = [make_result(0.60)]

    result = gate.evaluate(results)

    assert result.sufficient is True
    assert result.score == pytest.approx(0.60)
    assert result.results == results


def test_evidence_gate_accepts_exact_threshold():
    gate = EvidenceGate(threshold=0.48)

    results = [make_result(0.48)]

    result = gate.evaluate(results)

    assert result.sufficient is True
    assert result.score == pytest.approx(0.48)


def test_evidence_gate_uses_top_result():
    gate = EvidenceGate(threshold=0.48)

    results = [
        make_result(0.70),
        make_result(0.20),
        make_result(0.10),
    ]

    result = gate.evaluate(results)

    assert result.sufficient is True
    assert result.score == pytest.approx(0.70)


def test_evidence_gate_preserves_results():
    gate = EvidenceGate(threshold=0.48)

    results = [
        make_result(0.65),
        make_result(0.55),
    ]

    result = gate.evaluate(results)

    assert result.results == results
    assert len(result.results) == 2


def test_evidence_gate_supports_custom_threshold():
    gate = EvidenceGate(threshold=0.70)

    results = [make_result(0.60)]

    result = gate.evaluate(results)

    assert result.sufficient is False
    assert result.score == pytest.approx(0.60)
