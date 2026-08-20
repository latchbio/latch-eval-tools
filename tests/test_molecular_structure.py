from __future__ import annotations

import json

import pytest

from latch_eval_tools import MolecularStructureGrader
from latch_eval_tools.graders import GRADER_REGISTRY, get_grader
from latch_eval_tools.graders import molecular_structure as molecular_module
from latch_eval_tools.graders.base import GraderResult, serialize_grader_result
from latch_eval_tools.graders.molecular_structure import (
    CANONICALIZATION_REVISION,
    MORGAN_FINGERPRINT_REVISION,
    RDKIT_VERSION,
)

CONFIG = {
    "answer_field": "product_smiles",
    "expected_smiles": "CCO",
    "connectivity_only": True,
    "require_single_fragment": True,
    "similarity_threshold": 0.8,
}


def test_grader_is_registered_and_public() -> None:
    assert GRADER_REGISTRY["molecular_structure"] is MolecularStructureGrader
    assert isinstance(get_grader("molecular_structure"), MolecularStructureGrader)


def test_canonically_equivalent_smiles_pass_with_frozen_diagnostics() -> None:
    result = MolecularStructureGrader().evaluate_answer(
        {"product_smiles": "[H]OC([H])([H])C([H])([H])[H]"},
        CONFIG,
    )

    assert result.passed is True
    assert result.score == result.score_max == 1.0
    assert result.field_scores == {"product_smiles": 1.0}
    assert result.metrics["actual_canonical"] == "CCO"
    assert result.metrics["expected_canonical"] == "CCO"
    assert result.metrics["exact_match"] is True
    assert result.metrics["tanimoto"] == 1.0
    assert result.metrics["threshold_match"] is True
    assert result.metrics["tanimoto_scoring_role"] == "diagnostic_only"
    assert result.metrics["rdkit_version"] == RDKIT_VERSION
    assert result.metrics["canonicalization_revision"] == CANONICALIZATION_REVISION
    assert result.metrics["fingerprint_revision"] == MORGAN_FINGERPRINT_REVISION
    assert result.metrics["fingerprint_radius"] == 2
    assert result.metrics["fingerprint_size"] == 2048


def test_connectivity_only_ignores_stereo_isotopes_and_atom_maps() -> None:
    config = {**CONFIG, "expected_smiles": "F[C@@H]([13CH3:7])Cl"}
    result = MolecularStructureGrader().evaluate_answer(
        {"product_smiles": "F[C@H](C)Cl"},
        config,
    )

    assert result.passed is True
    assert result.metrics["actual_canonical"] == result.metrics["expected_canonical"]
    assert result.metrics["fingerprint_include_chirality"] is False


def test_stereo_is_preserved_when_connectivity_only_is_disabled() -> None:
    config = {
        **CONFIG,
        "expected_smiles": "F[C@@H](Cl)Br",
        "connectivity_only": False,
    }
    result = MolecularStructureGrader().evaluate_answer(
        {"product_smiles": "F[C@H](Cl)Br"},
        config,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics["exact_match"] is False
    assert result.metrics["fingerprint_include_chirality"] is True


def test_tanimoto_threshold_is_diagnostic_only_for_constitutional_mismatch() -> None:
    config = {**CONFIG, "expected_smiles": "CCCCCCC"}
    result = MolecularStructureGrader().evaluate_answer(
        {"product_smiles": "CCCCCC"},
        config,
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.field_scores == {"product_smiles": 0.0}
    assert result.metrics["exact_match"] is False
    assert result.metrics["tanimoto"] == pytest.approx(0.875)
    assert result.metrics["threshold_match"] is True


@pytest.mark.parametrize(
    ("answer", "error_fragment"),
    [
        ({}, "missing required field"),
        ({"product_smiles": None}, "non-empty string"),
        ({"product_smiles": ""}, "non-empty string"),
        ({"product_smiles": "not-smiles"}, "invalid SMILES"),
        ({"product_smiles": "CCO.[Na+]"}, "exactly one connected"),
    ],
)
def test_invalid_agent_smiles_fail_as_answer_errors(
    answer: dict, error_fragment: str
) -> None:
    result = MolecularStructureGrader().evaluate_answer(answer, CONFIG)

    assert result.passed is False
    assert result.score == 0.0
    assert result.field_scores == {"product_smiles": 0.0}
    assert error_fragment in result.metrics["answer_error"]
    assert "configuration_error" not in result.metrics
    if "product_smiles" in answer:
        assert error_fragment in result.metrics["parse_errors"]["actual"]


@pytest.mark.parametrize(
    "config",
    [
        {},
        {**CONFIG, "answer_field": ""},
        {**CONFIG, "answer_field": " product_smiles"},
        {**CONFIG, "expected_smiles": ""},
        {**CONFIG, "connectivity_only": "yes"},
        {**CONFIG, "require_single_fragment": 1},
        {**CONFIG, "similarity_threshold": -0.1},
        {**CONFIG, "similarity_threshold": 1.1},
        {**CONFIG, "similarity_treshold": 0.8},
    ],
)
def test_malformed_configuration_fails_as_configuration_error(config: dict) -> None:
    result = MolecularStructureGrader().evaluate_answer(
        {"product_smiles": "CCO"}, config
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics["configuration_error"]


def test_invalid_expected_smiles_is_a_configuration_error() -> None:
    result = MolecularStructureGrader().evaluate_answer(
        {"product_smiles": "CCO"},
        {**CONFIG, "expected_smiles": "not-smiles"},
    )

    assert result.passed is False
    assert result.score == 0.0
    assert "could not be canonicalized" in result.metrics["configuration_error"]
    assert result.metrics["parse_errors"] == {"expected": "invalid SMILES"}


def test_missing_chemistry_extra_fails_closed_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        raise molecular_module._ChemistryDependencyError(
            "molecular_structure requires the optional RDKit dependency. "
            "Install chemistry support with: pip install 'latch-eval-tools[chemistry]'"
        )

    monkeypatch.setattr(molecular_module, "_load_rdkit", unavailable)
    result = MolecularStructureGrader().evaluate_answer(
        {"product_smiles": "CCO"}, CONFIG
    )

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics["grader_system_error"] is True
    assert result.metrics["dependency"] == "rdkit"
    assert "latch-eval-tools[chemistry]" in result.metrics["grader_error"]


def test_grader_result_serialization_retains_score_contract() -> None:
    result = GraderResult(
        passed=True,
        score=2.0,
        score_max=3.0,
        field_scores={"product_smiles": 2.0},
        metrics={"exact_match": True},
        reasoning="ok",
        agent_answer={"product_smiles": "CCO"},
    )

    payload = serialize_grader_result(result)
    assert payload == {
        "passed": True,
        "score": 2.0,
        "score_max": 3.0,
        "field_scores": {"product_smiles": 2.0},
        "metrics": {"exact_match": True},
        "reasoning": "ok",
        "agent_answer": {"product_smiles": "CCO"},
    }
    assert json.loads(json.dumps(payload)) == payload
