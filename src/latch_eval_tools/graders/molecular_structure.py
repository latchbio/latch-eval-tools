from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .base import BinaryGrader, GraderResult, configuration_error_result
from .number_contract import is_finite_number

RDKIT_VERSION = "2025.09.6"
CANONICALIZATION_REVISION = "rdkit-sanitize-remove-hs-isomeric-v1"
MORGAN_FINGERPRINT_REVISION = "rdkit-2025.09.6-morgan-r2-2048-v1"
MORGAN_RADIUS = 2
MORGAN_FP_SIZE = 2048

_INSTALL_HINT = "Install chemistry support with: pip install 'latch-eval-tools[chemistry]'"


class _ChemistryDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class _RDKitRuntime:
    version: str
    Chem: Any
    DataStructs: Any
    rdFingerprintGenerator: Any


def _normalized_version(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in version.split("."))
    except (AttributeError, TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _load_rdkit() -> _RDKitRuntime:
    try:
        import rdkit
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as error:
        raise _ChemistryDependencyError(
            f"molecular_structure requires the optional RDKit dependency. {_INSTALL_HINT}"
        ) from error

    if _normalized_version(rdkit.__version__) != _normalized_version(RDKIT_VERSION):
        raise _ChemistryDependencyError(
            "molecular_structure requires RDKit "
            f"{RDKIT_VERSION}, found {rdkit.__version__}. {_INSTALL_HINT}"
        )

    return _RDKitRuntime(
        version=rdkit.__version__,
        Chem=Chem,
        DataStructs=DataStructs,
        rdFingerprintGenerator=rdFingerprintGenerator,
    )


def _canonicalize_smiles(
    runtime: _RDKitRuntime,
    smiles: object,
    *,
    connectivity_only: bool,
    require_single_fragment: bool,
) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")

    molecule = runtime.Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise ValueError("invalid SMILES")

    molecule = runtime.Chem.RemoveHs(molecule, sanitize=True)
    if require_single_fragment and len(runtime.Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("SMILES must contain exactly one connected molecular fragment")

    if connectivity_only:
        runtime.Chem.RemoveStereochemistry(molecule)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)

    canonical = runtime.Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=not connectivity_only,
    )
    reparsed = runtime.Chem.MolFromSmiles(canonical, sanitize=True)
    if reparsed is None:
        raise ValueError("canonical SMILES could not be reparsed")
    if require_single_fragment and len(runtime.Chem.GetMolFrags(reparsed)) != 1:
        raise ValueError("SMILES must contain exactly one connected molecular fragment")

    return runtime.Chem.MolToSmiles(
        reparsed,
        canonical=True,
        isomericSmiles=not connectivity_only,
    )


def _morgan_tanimoto(
    runtime: _RDKitRuntime,
    actual_canonical: str,
    expected_canonical: str,
    *,
    include_chirality: bool,
) -> float:
    actual = runtime.Chem.MolFromSmiles(actual_canonical, sanitize=True)
    expected = runtime.Chem.MolFromSmiles(expected_canonical, sanitize=True)
    if actual is None or expected is None:
        raise ValueError("canonical SMILES could not be parsed")

    generator = runtime.rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_FP_SIZE,
        includeChirality=include_chirality,
    )
    return float(
        runtime.DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(actual),
            generator.GetFingerprint(expected),
        )
    )


def _parse_config(agent_answer: object, config: object) -> dict | GraderResult:
    if not isinstance(config, dict):
        return configuration_error_result(
            agent_answer,
            "Molecular Structure",
            "config must be an object",
        )

    allowed_keys = {
        "answer_field",
        "expected_smiles",
        "connectivity_only",
        "require_single_fragment",
        "similarity_threshold",
    }
    unknown_keys = sorted(str(key) for key in config.keys() - allowed_keys)
    if unknown_keys:
        return configuration_error_result(
            agent_answer,
            "Molecular Structure",
            f"unknown config field(s): {', '.join(unknown_keys)}",
        )

    answer_field = config.get("answer_field")
    if (
        not isinstance(answer_field, str)
        or not answer_field
        or answer_field != answer_field.strip()
    ):
        return configuration_error_result(
            agent_answer,
            "Molecular Structure",
            "answer_field must be a non-empty string without surrounding whitespace",
        )

    expected_smiles = config.get("expected_smiles")
    if not isinstance(expected_smiles, str) or not expected_smiles.strip():
        return configuration_error_result(
            agent_answer,
            "Molecular Structure",
            "expected_smiles must be a non-empty string",
        )

    connectivity_only = config.get("connectivity_only", False)
    if not isinstance(connectivity_only, bool):
        return configuration_error_result(
            agent_answer,
            "Molecular Structure",
            "connectivity_only must be a boolean",
        )

    require_single_fragment = config.get("require_single_fragment", False)
    if not isinstance(require_single_fragment, bool):
        return configuration_error_result(
            agent_answer,
            "Molecular Structure",
            "require_single_fragment must be a boolean",
        )

    similarity_threshold = config.get("similarity_threshold", 0.8)
    if not is_finite_number(similarity_threshold):
        return configuration_error_result(
            agent_answer,
            "Molecular Structure",
            "similarity_threshold must be a finite number in [0, 1]",
        )
    similarity_threshold = float(similarity_threshold)
    if similarity_threshold < 0.0 or similarity_threshold > 1.0:
        return configuration_error_result(
            agent_answer,
            "Molecular Structure",
            "similarity_threshold must be a finite number in [0, 1]",
        )

    return {
        "answer_field": answer_field,
        "expected_smiles": expected_smiles,
        "connectivity_only": connectivity_only,
        "require_single_fragment": require_single_fragment,
        "similarity_threshold": similarity_threshold,
    }


def _diagnostic_metrics(
    *,
    answer_field: str,
    actual_raw: object,
    expected_raw: str,
    actual_canonical: str | None,
    expected_canonical: str | None,
    exact_match: bool,
    tanimoto: float | None,
    similarity_threshold: float,
    connectivity_only: bool,
    require_single_fragment: bool,
    parse_errors: dict[str, str] | None,
    rdkit_version: str,
) -> dict:
    return {
        "answer_field": answer_field,
        "actual_raw": actual_raw,
        "expected_raw": expected_raw,
        "actual_canonical": actual_canonical,
        "expected_canonical": expected_canonical,
        "parse_errors": parse_errors,
        "exact_match": exact_match,
        "tanimoto": tanimoto,
        "tanimoto_scoring_role": "diagnostic_only",
        "similarity_threshold": similarity_threshold,
        "threshold_match": tanimoto is not None and tanimoto >= similarity_threshold,
        "connectivity_only": connectivity_only,
        "fingerprint_include_chirality": not connectivity_only,
        "require_single_fragment": require_single_fragment,
        "rdkit_version": rdkit_version,
        "canonicalization_revision": CANONICALIZATION_REVISION,
        "fingerprint_revision": MORGAN_FINGERPRINT_REVISION,
        "fingerprint_radius": MORGAN_RADIUS,
        "fingerprint_size": MORGAN_FP_SIZE,
    }


class MolecularStructureGrader(BinaryGrader):
    """Grade canonical molecular identity and report similarity diagnostically."""

    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        parsed_config = _parse_config(agent_answer, config)
        if isinstance(parsed_config, GraderResult):
            return parsed_config

        answer_field = parsed_config["answer_field"]
        expected_raw = parsed_config["expected_smiles"]
        connectivity_only = parsed_config["connectivity_only"]
        require_single_fragment = parsed_config["require_single_fragment"]
        similarity_threshold = parsed_config["similarity_threshold"]

        try:
            runtime = _load_rdkit()
        except _ChemistryDependencyError as error:
            return GraderResult(
                passed=False,
                metrics={
                    "grader_system_error": True,
                    "grader_error": str(error),
                    "dependency": "rdkit",
                },
                reasoning=f"Molecular Structure: SYSTEM ERROR\n\n  x {error}",
                agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
                score=0.0,
                field_scores={},
            )

        try:
            expected_canonical = _canonicalize_smiles(
                runtime,
                expected_raw,
                connectivity_only=connectivity_only,
                require_single_fragment=require_single_fragment,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            reason = f"expected_smiles could not be canonicalized: {error}"
            result = configuration_error_result(
                agent_answer,
                "Molecular Structure",
                reason,
            )
            result.metrics.update(
                _diagnostic_metrics(
                    answer_field=answer_field,
                    actual_raw=None,
                    expected_raw=expected_raw,
                    actual_canonical=None,
                    expected_canonical=None,
                    exact_match=False,
                    tanimoto=None,
                    similarity_threshold=similarity_threshold,
                    connectivity_only=connectivity_only,
                    require_single_fragment=require_single_fragment,
                    parse_errors={"expected": str(error)},
                    rdkit_version=runtime.version,
                )
            )
            return result

        if not isinstance(agent_answer, dict):
            actual_raw = None
            answer_error = "agent answer must be an object"
        elif answer_field not in agent_answer:
            actual_raw = None
            answer_error = f"agent answer missing required field: {answer_field}"
        else:
            actual_raw = agent_answer[answer_field]
            answer_error = None

        actual_canonical: str | None = None
        parse_errors: dict[str, str] | None = None
        if answer_error is None:
            try:
                actual_canonical = _canonicalize_smiles(
                    runtime,
                    actual_raw,
                    connectivity_only=connectivity_only,
                    require_single_fragment=require_single_fragment,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                answer_error = f"agent answer could not be canonicalized: {error}"
                parse_errors = {"actual": str(error)}

        exact_match = (
            actual_canonical is not None and actual_canonical == expected_canonical
        )
        tanimoto = None
        if actual_canonical is not None:
            tanimoto = _morgan_tanimoto(
                runtime,
                actual_canonical,
                expected_canonical,
                include_chirality=not connectivity_only,
            )
        score = 1.0 if exact_match else 0.0
        metrics = _diagnostic_metrics(
            answer_field=answer_field,
            actual_raw=actual_raw,
            expected_raw=expected_raw,
            actual_canonical=actual_canonical,
            expected_canonical=expected_canonical,
            exact_match=exact_match,
            tanimoto=tanimoto,
            similarity_threshold=similarity_threshold,
            connectivity_only=connectivity_only,
            require_single_fragment=require_single_fragment,
            parse_errors=parse_errors,
            rdkit_version=runtime.version,
        )
        if answer_error is not None:
            metrics["answer_error"] = answer_error

        if exact_match:
            reasoning = (
                "Molecular Structure: PASS\n\n"
                f"  + {answer_field} is an exact canonical molecular match"
            )
        elif answer_error is not None:
            reasoning = f"Molecular Structure: FAIL\n\n  x {answer_error}"
        else:
            reasoning = (
                "Molecular Structure: FAIL\n\n"
                "  x Canonical molecular connectivity does not match exactly\n"
                f"    Morgan Tanimoto (diagnostic only): {tanimoto:.4f}"
            )

        return GraderResult(
            passed=exact_match,
            metrics=metrics,
            reasoning=reasoning,
            agent_answer=agent_answer if isinstance(agent_answer, dict) else None,
            score=score,
            score_max=1.0,
            field_scores={answer_field: score},
        )


__all__ = [
    "CANONICALIZATION_REVISION",
    "MORGAN_FINGERPRINT_REVISION",
    "RDKIT_VERSION",
    "MolecularStructureGrader",
]
