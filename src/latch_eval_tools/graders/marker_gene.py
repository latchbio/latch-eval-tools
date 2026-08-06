from .base import BinaryGrader, GraderResult
from .list_contract import ListCardinality, check_list_cardinality


class MarkerGenePrecisionRecallGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        canonical_markers = config.get(
            "canonical_markers", config.get("ground_truth_labels", [])
        )
        scoring = config.get("scoring", {})
        if not isinstance(scoring, dict):
            return GraderResult(
                passed=False,
                metrics={"configuration_error": "scoring must be an object"},
                reasoning=f"scoring must be an object, got {type(scoring).__name__}",
                agent_answer=agent_answer,
                score=0.0,
            )
        thresholds = scoring.get("pass_thresholds", {})
        if not isinstance(thresholds, dict):
            return GraderResult(
                passed=False,
                metrics={"configuration_error": "pass_thresholds must be an object"},
                reasoning=(
                    "pass_thresholds must be an object, got "
                    f"{type(thresholds).__name__}"
                ),
                agent_answer=agent_answer,
                score=0.0,
            )

        configured_cardinality = check_list_cardinality(
            [], config.get("expected_count")
        )
        if configured_cardinality.configuration_error is not None:
            return GraderResult(
                passed=False,
                metrics={
                    "configuration_error": configured_cardinality.configuration_error
                },
                reasoning=configured_cardinality.configuration_error,
                agent_answer=agent_answer,
                score=0.0,
            )
        expected_count = configured_cardinality.expected_count
        answer_field = config.get("answer_field", "top_marker_genes")

        if not isinstance(answer_field, str) or answer_field.strip() == "":
            return GraderResult(
                passed=False,
                metrics={
                    "configuration_error": "answer_field must be a non-empty string"
                },
                reasoning=f"answer_field must be a non-empty string, got {answer_field!r}",
                agent_answer=agent_answer,
                score=0.0,
            )

        if answer_field not in agent_answer:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning=f"Agent answer missing required field: {answer_field}",
                agent_answer=agent_answer,
                score=0.0,
            )

        predicted = agent_answer[answer_field]

        if isinstance(canonical_markers, dict) and isinstance(predicted, dict):
            return self._evaluate_per_celltype(
                predicted,
                canonical_markers,
                thresholds,
                answer_field,
                agent_answer,
                expected_count,
            )

        if isinstance(canonical_markers, dict) and answer_field in canonical_markers:
            canonical_markers = canonical_markers[answer_field]

        if not isinstance(predicted, list):
            return GraderResult(
                passed=False,
                metrics={},
                reasoning=f"{answer_field} must be a list, got {type(predicted).__name__}",
                agent_answer=agent_answer,
                score=0.0,
            )

        if not isinstance(canonical_markers, list):
            return GraderResult(
                passed=False,
                metrics={},
                reasoning=f"canonical_markers must be a list for flat evaluation, got {type(canonical_markers).__name__}",
                agent_answer=agent_answer,
                score=0.0,
            )

        return self._evaluate_flat_list(
            predicted,
            canonical_markers,
            thresholds,
            answer_field,
            agent_answer,
            expected_count,
        )

    def _evaluate_per_celltype(
        self,
        predicted: dict,
        canonical_markers: dict,
        thresholds: dict,
        answer_field: str,
        agent_answer: dict,
        expected_count: int | None,
    ) -> GraderResult:
        if len(canonical_markers) == 0:
            configuration_error = (
                "canonical_markers must contain at least one cell type for "
                "per-cell-type evaluation"
            )
            return GraderResult(
                passed=False,
                metrics={"configuration_error": configuration_error},
                reasoning=configuration_error,
                agent_answer=agent_answer,
                score=0.0,
            )

        for celltype, canonical_genes in canonical_markers.items():
            if not isinstance(canonical_genes, list):
                configuration_error = (
                    f"canonical_markers[{celltype!r}] must be a non-empty list, "
                    f"got {type(canonical_genes).__name__}"
                )
                return GraderResult(
                    passed=False,
                    metrics={"configuration_error": configuration_error},
                    reasoning=configuration_error,
                    agent_answer=agent_answer,
                    score=0.0,
                )
            if len(canonical_genes) == 0:
                configuration_error = (
                    f"canonical_markers[{celltype!r}] must be a non-empty list"
                )
                return GraderResult(
                    passed=False,
                    metrics={"configuration_error": configuration_error},
                    reasoning=configuration_error,
                    agent_answer=agent_answer,
                    score=0.0,
                )

        min_recall = thresholds.get(
            "min_recall_per_celltype", thresholds.get("recall_at_k", 0.50)
        )
        min_celltypes_passing = thresholds.get(
            "min_celltypes_passing", len(canonical_markers)
        )

        celltype_results = {}
        celltypes_passing = 0
        total_celltypes = len(canonical_markers)

        for celltype, canonical_genes in canonical_markers.items():
            predicted_genes = predicted.get(celltype, [])
            if not isinstance(predicted_genes, list):
                celltype_results[celltype] = {
                    "pass": False,
                    "recall": 0.0,
                    "error": f"Expected list, got {type(predicted_genes).__name__}",
                }
                continue

            predicted_genes = [str(g) for g in predicted_genes]
            normalized_predicted = [gene.lower() for gene in predicted_genes]
            cardinality = check_list_cardinality(normalized_predicted, expected_count)
            if cardinality.configuration_error is not None:
                return GraderResult(
                    passed=False,
                    metrics={"configuration_error": cardinality.configuration_error},
                    reasoning=cardinality.configuration_error,
                    agent_answer=agent_answer,
                    score=0.0,
                )

            if cardinality.expected_count is None:
                scored_genes = normalized_predicted
            else:
                scored_genes = normalized_predicted[: cardinality.expected_count]
            canonical_set = {str(gene).lower() for gene in canonical_genes}
            predicted_set = set(scored_genes)

            true_positives = canonical_set & predicted_set
            false_negatives = canonical_set - predicted_set

            recall = (
                len(true_positives) / len(canonical_set)
                if len(canonical_set) > 0
                else 1.0
            )
            celltype_pass = recall >= min_recall and cardinality.passed

            if celltype_pass:
                celltypes_passing += 1

            celltype_results[celltype] = {
                "pass": celltype_pass,
                "recall": recall,
                "num_predicted": len(predicted_genes),
                "num_unique_predicted": cardinality.unique_count,
                "expected_count": cardinality.expected_count,
                "cardinality_pass": cardinality.passed,
                "num_canonical": len(canonical_set),
                "true_positives": sorted(true_positives),
                "false_negatives": sorted(false_negatives),
            }

        passed = celltypes_passing >= min_celltypes_passing

        metrics = {
            "celltypes_passing": celltypes_passing,
            "total_celltypes": total_celltypes,
            "min_celltypes_passing": min_celltypes_passing,
            "min_recall_per_celltype": min_recall,
            "expected_count": expected_count,
            "per_celltype": celltype_results,
            "answer_field_used": answer_field,
        }

        lines = [
            f"Marker Gene Per-Celltype: {'PASS' if passed else 'FAIL'}",
            f"Celltypes passing: {celltypes_passing}/{total_celltypes} (required: {min_celltypes_passing})",
            "",
        ]
        for celltype, result in celltype_results.items():
            check = "+" if result["pass"] else "x"
            lines.append(
                f"  {check} {celltype}: recall={result['recall']:.2f} (threshold: {min_recall:.2f})"
            )

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning="\n".join(lines),
            agent_answer=agent_answer,
            score=1.0 if passed else 0.0,
        )

    def _evaluate_flat_list(
        self,
        predicted_genes: list,
        canonical_markers: list,
        thresholds: dict,
        answer_field: str,
        agent_answer: dict,
        expected_count: int | None,
    ) -> GraderResult:
        precision_threshold = thresholds.get("precision_at_k", 0.60)
        recall_threshold = thresholds.get("recall_at_k", 0.50)

        predicted_genes = [str(g) for g in predicted_genes]
        normalized_predicted = [gene.lower() for gene in predicted_genes]
        cardinality = check_list_cardinality(normalized_predicted, expected_count)
        if cardinality.configuration_error is not None:
            return GraderResult(
                passed=False,
                metrics={"configuration_error": cardinality.configuration_error},
                reasoning=cardinality.configuration_error,
                agent_answer=agent_answer,
                score=0.0,
            )

        k = (
            cardinality.expected_count
            if cardinality.expected_count is not None
            else len(predicted_genes)
        )
        scored_predicted = (
            normalized_predicted[:k]
            if cardinality.expected_count is not None
            else normalized_predicted
        )

        canonical_set = {str(gene).lower() for gene in canonical_markers}
        predicted_set = set(scored_predicted)

        true_positives = canonical_set & predicted_set
        false_positives = predicted_set - canonical_set
        false_negatives = canonical_set - predicted_set

        precision_at_k = len(true_positives) / k if k > 0 else 0.0
        recall_at_k = (
            len(true_positives) / len(canonical_set) if len(canonical_set) > 0 else 0.0
        )

        precision_pass = precision_at_k >= precision_threshold
        recall_pass = recall_at_k >= recall_threshold
        passed = precision_pass and recall_pass and cardinality.passed

        original_case_map = {gene.lower(): gene for gene in predicted_genes}
        canonical_case_map = {
            str(gene).lower(): str(gene) for gene in canonical_markers
        }

        true_positive_genes = [
            original_case_map.get(g, canonical_case_map.get(g, g))
            for g in true_positives
        ]
        false_positive_genes = [original_case_map.get(g, g) for g in false_positives]
        false_negative_genes = [canonical_case_map.get(g, g) for g in false_negatives]

        metrics = {
            "k": k,
            "precision_at_k": precision_at_k,
            "recall_at_k": recall_at_k,
            "precision_threshold": precision_threshold,
            "recall_threshold": recall_threshold,
            "true_positives": sorted(true_positive_genes),
            "false_positives": sorted(false_positive_genes),
            "false_negatives": sorted(false_negative_genes),
            "num_true_positives": len(true_positives),
            "num_false_positives": len(false_positives),
            "num_false_negatives": len(false_negatives),
            "num_canonical_markers": len(canonical_set),
            "submitted_count": cardinality.submitted_count,
            "unique_count": cardinality.unique_count,
            "expected_count": cardinality.expected_count,
            "cardinality_pass": cardinality.passed,
            "precision_pass": precision_pass,
            "recall_pass": recall_pass,
            "answer_field_used": answer_field,
        }

        reasoning = self._format_reasoning(
            k,
            precision_at_k,
            recall_at_k,
            precision_threshold,
            recall_threshold,
            true_positive_genes,
            false_positive_genes,
            false_negative_genes,
            precision_pass,
            recall_pass,
            passed,
            answer_field,
            cardinality,
        )

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning=reasoning,
            agent_answer=agent_answer,
            score=1.0 if passed else 0.0,
        )

    def _format_reasoning(
        self,
        k,
        precision,
        recall,
        precision_threshold,
        recall_threshold,
        true_positives,
        false_positives,
        false_negatives,
        precision_pass,
        recall_pass,
        passed,
        answer_field,
        cardinality: ListCardinality,
    ):
        lines = [
            f"Marker Gene Precision/Recall: {'PASS' if passed else 'FAIL'}",
            f"Answer field: {answer_field}",
            "",
            f"  {'+' if precision_pass else 'x'} Precision@{k}: {precision:.3f} (threshold: {precision_threshold:.3f})",
            f"  {'+' if recall_pass else 'x'} Recall@{k}: {recall:.3f} (threshold: {recall_threshold:.3f})",
        ]
        if cardinality.expected_count is not None:
            lines.append(
                f"  {'+' if cardinality.passed else 'x'} Exact unique count: "
                f"submitted={cardinality.submitted_count}, "
                f"unique={cardinality.unique_count}, "
                f"expected={cardinality.expected_count}"
            )
        lines.extend(["", f"True Positives ({len(true_positives)}):"])

        if true_positives:
            for gene in sorted(true_positives):
                lines.append(f"  + {gene}")
        else:
            lines.append("  None")

        lines.extend(["", f"False Negatives ({len(false_negatives)}):"])
        if false_negatives:
            for gene in sorted(false_negatives):
                lines.append(f"  - {gene}")
        else:
            lines.append("  None")

        if not passed:
            lines.append("")
            failures = []
            if not precision_pass:
                failures.append(
                    f"Precision {precision:.3f} < {precision_threshold:.3f}"
                )
            if not recall_pass:
                failures.append(f"Recall {recall:.3f} < {recall_threshold:.3f}")
            if not cardinality.passed:
                failures.append(
                    "Response must contain exactly "
                    f"{cardinality.expected_count} unique items"
                )
            lines.append(f"Failure: {'; '.join(failures)}")

        return "\n".join(lines)


class MarkerGeneSeparationGrader(BinaryGrader):
    def evaluate_answer(self, agent_answer: dict, config: dict) -> GraderResult:
        scoring = config.get("scoring", {})
        thresholds = scoring.get("pass_thresholds", {})
        mean_auroc_threshold = thresholds.get("mean_auroc", 0.85)
        fraction_high_threshold = thresholds.get("fraction_high", 0.70)
        per_gene_cutoff = thresholds.get("per_gene_cutoff", 0.80)

        if "per_gene_stats" not in agent_answer:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="Agent answer missing required field: per_gene_stats",
                agent_answer=agent_answer,
                score=0.0,
            )

        if "mean_auroc" not in agent_answer:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="Agent answer missing required field: mean_auroc",
                agent_answer=agent_answer,
                score=0.0,
            )

        per_gene_stats = agent_answer["per_gene_stats"]
        agent_mean_auroc = agent_answer["mean_auroc"]

        if not isinstance(per_gene_stats, list):
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="per_gene_stats must be a list",
                agent_answer=agent_answer,
                score=0.0,
            )

        num_genes = len(per_gene_stats)
        if num_genes == 0:
            return GraderResult(
                passed=False,
                metrics={},
                reasoning="per_gene_stats is empty",
                agent_answer=agent_answer,
                score=0.0,
            )

        gene_aurocs = {}
        for stat in per_gene_stats:
            if not isinstance(stat, dict) or "gene" not in stat or "auroc" not in stat:
                return GraderResult(
                    passed=False,
                    metrics={},
                    reasoning="Each element in per_gene_stats must have 'gene' and 'auroc' fields",
                    agent_answer=agent_answer,
                    score=0.0,
                )
            gene_aurocs[stat["gene"]] = stat["auroc"]

        computed_mean_auroc = sum(gene_aurocs.values()) / len(gene_aurocs)

        high_auroc_genes = [
            gene for gene, auroc in gene_aurocs.items() if auroc >= per_gene_cutoff
        ]
        low_auroc_genes = [
            gene for gene, auroc in gene_aurocs.items() if auroc < per_gene_cutoff
        ]
        fraction_high = len(high_auroc_genes) / num_genes

        mean_auroc_pass = agent_mean_auroc >= mean_auroc_threshold
        fraction_high_pass = fraction_high >= fraction_high_threshold
        passed = mean_auroc_pass and fraction_high_pass

        metrics = {
            "num_genes": num_genes,
            "mean_auroc_agent": agent_mean_auroc,
            "mean_auroc_computed": computed_mean_auroc,
            "mean_auroc_threshold": mean_auroc_threshold,
            "fraction_high": fraction_high,
            "fraction_high_threshold": fraction_high_threshold,
            "per_gene_cutoff": per_gene_cutoff,
            "num_high_auroc_genes": len(high_auroc_genes),
            "num_low_auroc_genes": len(low_auroc_genes),
            "high_auroc_genes": sorted(high_auroc_genes),
            "low_auroc_genes": sorted(low_auroc_genes),
            "mean_auroc_pass": mean_auroc_pass,
            "fraction_high_pass": fraction_high_pass,
            "per_gene_aurocs": gene_aurocs,
        }

        lines = [
            f"Marker Gene Separation: {'PASS' if passed else 'FAIL'}",
            "",
            f"  {'+' if mean_auroc_pass else 'x'} Mean AUROC: {agent_mean_auroc:.3f} (threshold: {mean_auroc_threshold:.3f})",
            f"  {'+' if fraction_high_pass else 'x'} Fraction High (>={per_gene_cutoff:.2f}): {fraction_high:.3f} ({len(high_auroc_genes)}/{num_genes})",
        ]

        if not passed:
            failures = []
            if not mean_auroc_pass:
                failures.append(
                    f"Mean AUROC {agent_mean_auroc:.3f} < {mean_auroc_threshold:.3f}"
                )
            if not fraction_high_pass:
                failures.append(
                    f"Fraction high {fraction_high:.3f} < {fraction_high_threshold:.3f}"
                )
            lines.append(f"\nFailure: {'; '.join(failures)}")

        return GraderResult(
            passed=passed,
            metrics=metrics,
            reasoning="\n".join(lines),
            agent_answer=agent_answer,
            score=1.0 if passed else 0.0,
        )
