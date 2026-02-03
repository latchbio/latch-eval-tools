from dataclasses import dataclass


@dataclass
class ErrorExplanation:
    code: str
    title: str
    explanation: str
    example_before: str
    example_after: str
    doc_link: str | None = None


EXPLANATIONS: dict[str, ErrorExplanation] = {
    "E000": ErrorExplanation(
        code="E000",
        title="File not found",
        explanation="The specified file does not exist at the given path.",
        example_before="evals/missing_file.json",
        example_after="evals/my_eval.json  # Use correct path",
        doc_link=None,
    ),
    "E001": ErrorExplanation(
        code="E001",
        title="Invalid JSON / Missing 'id' field",
        explanation="The file contains malformed JSON or is missing the required 'id' field. Every eval must have a unique identifier.",
        example_before='{ "task": "..." }',
        example_after='{ "id": "my_eval_001", "task": "..." }',
        doc_link=None,
    ),
    "E002": ErrorExplanation(
        code="E002",
        title="Invalid root type / Invalid 'id' field",
        explanation="The root must be a JSON object, or the 'id' field must be a non-empty string.",
        example_before='{ "id": "" }',
        example_after='{ "id": "clustering_exp_01" }',
        doc_link=None,
    ),
    "E003": ErrorExplanation(
        code="E003",
        title="Missing 'task' field",
        explanation="Every eval must have a 'task' field containing the prompt/question for the agent.",
        example_before='{ "id": "eval_01" }',
        example_after='{ "id": "eval_01", "task": "Perform clustering on the provided dataset..." }',
        doc_link=None,
    ),
    "E004": ErrorExplanation(
        code="E004",
        title="Invalid 'task' field",
        explanation="The 'task' field must be a non-empty string describing what the agent should do.",
        example_before='{ "task": "" }',
        example_after='{ "task": "Calculate the number of clusters in the dataset..." }',
        doc_link=None,
    ),
    "E005": ErrorExplanation(
        code="E005",
        title="Missing 'metadata' field",
        explanation="Every eval must have a 'metadata' object containing category, kit, time_horizon, etc.",
        example_before='{ "id": "eval_01", "task": "..." }',
        example_after='{ "id": "eval_01", "task": "...", "metadata": { "task": "clustering", "kit": "xenium", "time_horizon": "small" } }',
        doc_link=None,
    ),
    "E006": ErrorExplanation(
        code="E006",
        title="Invalid 'metadata' field",
        explanation="The 'metadata' field must be a JSON object, not a string or array.",
        example_before='"metadata": "clustering"',
        example_after='"metadata": { "task": "clustering" }',
        doc_link=None,
    ),
    "E010": ErrorExplanation(
        code="E010",
        title="Missing 'metadata.task'",
        explanation="The metadata must specify a task category (e.g., 'clustering', 'normalization').",
        example_before='"metadata": { "kit": "xenium" }',
        example_after='"metadata": { "task": "clustering", "kit": "xenium" }',
        doc_link=None,
    ),
    "E011": ErrorExplanation(
        code="E011",
        title="Invalid 'metadata.task'",
        explanation="The task category must be one of: qc, normalization, dimensionality_reduction, clustering, cell_typing, differential_expression, spatial_analysis.",
        example_before='"task": "cluster_analysis"',
        example_after='"task": "clustering"',
        doc_link=None,
    ),
    "E012": ErrorExplanation(
        code="E012",
        title="Missing 'metadata.kit'",
        explanation="The metadata must specify which spatial platform kit was used.",
        example_before='"metadata": { "task": "clustering" }',
        example_after='"metadata": { "task": "clustering", "kit": "xenium" }',
        doc_link=None,
    ),
    "E013": ErrorExplanation(
        code="E013",
        title="Invalid 'metadata.kit'",
        explanation="The kit must be one of: xenium, visium, merfish, vizgen, cosmx, seeker, takara, atlasxomics, curio.",
        example_before='"kit": "10x"',
        example_after='"kit": "xenium"',
        doc_link=None,
    ),
    "E014": ErrorExplanation(
        code="E014",
        title="Missing 'metadata.time_horizon'",
        explanation="The metadata must specify the expected time horizon for the task.",
        example_before='"metadata": { "task": "clustering", "kit": "xenium" }',
        example_after='"metadata": { "task": "clustering", "kit": "xenium", "time_horizon": "small" }',
        doc_link=None,
    ),
    "E015": ErrorExplanation(
        code="E015",
        title="Invalid 'metadata.time_horizon'",
        explanation="The time horizon must be one of: small, medium, large.",
        example_before='"time_horizon": "quick"',
        example_after='"time_horizon": "small"',
        doc_link=None,
    ),
    "E016": ErrorExplanation(
        code="E016",
        title="Invalid 'metadata.eval_type'",
        explanation="The eval_type must be one of: scientific, procedural, observational. Note: 'benchmark' is NOT valid.",
        example_before='"eval_type": "benchmark"',
        example_after='"eval_type": "observational"',
        doc_link=None,
    ),
    "E020": ErrorExplanation(
        code="E020",
        title="Invalid data_node type",
        explanation="The data_node field must be a string (Latch URI).",
        example_before='"data_node": 12345',
        example_after='"data_node": "latch://40248.account/path/to/data"',
        doc_link=None,
    ),
    "E021": ErrorExplanation(
        code="E021",
        title="Invalid data_node format",
        explanation="The data_node must be a valid Latch URI: latch://<id>.(account|node)/<path>",
        example_before='"data_node": "s3://bucket/data"',
        example_after='"data_node": "latch://40248.account/spatialbench/data/GSE123"',
        doc_link=None,
    ),
    "E022": ErrorExplanation(
        code="E022",
        title="Invalid data_node type",
        explanation="The data_node must be a string or array of strings, not an object.",
        example_before='"data_node": { "path": "..." }',
        example_after='"data_node": "latch://40248.account/path/to/data"',
        doc_link=None,
    ),
    "E030": ErrorExplanation(
        code="E030",
        title="Invalid grader type",
        explanation="The grader field must be a JSON object.",
        example_before='"grader": "numeric_tolerance"',
        example_after='"grader": { "type": "numeric_tolerance", "config": { ... } }',
        doc_link=None,
    ),
    "E031": ErrorExplanation(
        code="E031",
        title="Missing 'grader.type'",
        explanation="The grader must specify a type (e.g., 'numeric_tolerance', 'multiple_choice').",
        example_before='"grader": { "config": { ... } }',
        example_after='"grader": { "type": "numeric_tolerance", "config": { ... } }',
        doc_link=None,
    ),
    "E032": ErrorExplanation(
        code="E032",
        title="Invalid 'grader.type'",
        explanation="The grader type must be one of: numeric_tolerance, multiple_choice, distribution_comparison, marker_gene_precision_recall, label_set_jaccard, jaccard_label_set, marker_gene_separation, spatial_adjacency.",
        example_before='"type": "exact_match"',
        example_after='"type": "numeric_tolerance"',
        doc_link=None,
    ),
    "E033": ErrorExplanation(
        code="E033",
        title="Missing 'grader.config'",
        explanation="The grader must have a config object with grader-specific settings.",
        example_before='"grader": { "type": "numeric_tolerance" }',
        example_after='"grader": { "type": "numeric_tolerance", "config": { "ground_truth": { "n_clusters": 5 }, "tolerances": { ... } } }',
        doc_link=None,
    ),
    "E034": ErrorExplanation(
        code="E034",
        title="Invalid 'grader.config'",
        explanation="The grader config must be a JSON object.",
        example_before='"config": "default"',
        example_after='"config": { "ground_truth": { ... } }',
        doc_link=None,
    ),
    "E035": ErrorExplanation(
        code="E035",
        title="Missing required config field",
        explanation="The grader config is missing a required field for this grader type.",
        example_before='"config": { "ground_truth": { "n_clusters": 5 } }',
        example_after='"config": { "ground_truth": { "n_clusters": 5 }, "tolerances": { "n_clusters": { "type": "absolute", "value": 1 } } }',
        doc_link=None,
    ),
    "E036": ErrorExplanation(
        code="E036",
        title="Missing required config field (one of)",
        explanation="The grader config must have at least one of the specified fields.",
        example_before='"config": { }',
        example_after='"config": { "ground_truth_labels": ["A", "B", "C"] }',
        doc_link=None,
    ),
    "E037": ErrorExplanation(
        code="E037",
        title="Missing 'answer_field' in marker_gene_precision_recall",
        explanation="The marker_gene_precision_recall grader requires an 'answer_field' specifying which JSON field in the agent's response contains the gene list.",
        example_before='"config": { "canonical_markers": ["Epcam"], "scoring": { ... } }',
        example_after='"config": { "canonical_markers": ["Epcam"], "answer_field": "housekeeping_genes", "scoring": { ... } }',
        doc_link=None,
    ),
    "E040": ErrorExplanation(
        code="E040",
        title="Invalid tolerances type",
        explanation="The tolerances field must be a JSON object mapping field names to tolerance configs.",
        example_before='"tolerances": 0.1',
        example_after='"tolerances": { "n_clusters": { "type": "absolute", "value": 1 } }',
        doc_link=None,
    ),
    "E041": ErrorExplanation(
        code="E041",
        title="Invalid tolerance config",
        explanation="Each tolerance config must be a JSON object with 'type' and 'value'.",
        example_before='"n_clusters": 1',
        example_after='"n_clusters": { "type": "absolute", "value": 1 }',
        doc_link=None,
    ),
    "E042": ErrorExplanation(
        code="E042",
        title="Missing tolerance type",
        explanation="Each tolerance config must specify a type.",
        example_before='"n_clusters": { "value": 1 }',
        example_after='"n_clusters": { "type": "absolute", "value": 1 }',
        doc_link=None,
    ),
    "E043": ErrorExplanation(
        code="E043",
        title="Invalid tolerance type",
        explanation="The tolerance type must be one of: absolute, relative, min, max. Note: 'percentage' is NOT valid.",
        example_before='"type": "percentage"',
        example_after='"type": "relative"',
        doc_link=None,
    ),
    "E044": ErrorExplanation(
        code="E044",
        title="Missing tolerance value",
        explanation="Each tolerance config must specify a numeric value.",
        example_before='"n_clusters": { "type": "absolute" }',
        example_after='"n_clusters": { "type": "absolute", "value": 1 }',
        doc_link=None,
    ),
    "E045": ErrorExplanation(
        code="E045",
        title="Invalid tolerance value",
        explanation="The tolerance value must be a number (int or float).",
        example_before='"value": "one"',
        example_after='"value": 1',
        doc_link=None,
    ),
    "W000": ErrorExplanation(
        code="W000",
        title="Non-JSON file extension",
        explanation="The file does not have a .json extension. While it may still be valid JSON, consider renaming for clarity.",
        example_before="my_eval.txt",
        example_after="my_eval.json",
        doc_link=None,
    ),
    "W001": ErrorExplanation(
        code="W001",
        title="Missing 'metadata.eval_type'",
        explanation="Consider adding an eval_type to classify this eval. Valid types: scientific, procedural, observational.",
        example_before='"metadata": { "task": "clustering" }',
        example_after='"metadata": { "task": "clustering", "eval_type": "observational" }',
        doc_link=None,
    ),
    "W010": ErrorExplanation(
        code="W010",
        title="Missing <EVAL_ANSWER> block",
        explanation="The task description should include an <EVAL_ANSWER> block to specify the expected output format for the agent.",
        example_before='"task": "Count the clusters in the dataset."',
        example_after='"task": "Count the clusters in the dataset.\\n\\n<EVAL_ANSWER>\\n{\\\"n_clusters\\\": <integer>}\\n</EVAL_ANSWER>"',
        doc_link=None,
    ),
    "W011": ErrorExplanation(
        code="W011",
        title="Missing </EVAL_ANSWER> closing tag",
        explanation="The task has an <EVAL_ANSWER> tag but is missing the closing </EVAL_ANSWER> tag.",
        example_before='"task": "...\\n<EVAL_ANSWER>\\n..."',
        example_after='"task": "...\\n<EVAL_ANSWER>\\n...\\n</EVAL_ANSWER>"',
        doc_link=None,
    ),
    "W012": ErrorExplanation(
        code="W012",
        title="Missing 'Return EXACTLY:' instruction",
        explanation="Tasks with <EVAL_ANSWER> blocks should include 'Return EXACTLY:' before the block to clearly indicate the agent must output the exact format shown, including the tags.",
        example_before='"task": "Count clusters.\\n\\n<EVAL_ANSWER>\\n{\\\"n_clusters\\\": <int>}\\n</EVAL_ANSWER>"',
        example_after='"task": "Count clusters.\\n\\nReturn EXACTLY:\\n\\n<EVAL_ANSWER>\\n{\\\"n_clusters\\\": <int>}\\n</EVAL_ANSWER>"',
        doc_link=None,
    ),
}


def get_explanation(code: str) -> ErrorExplanation | None:
    return EXPLANATIONS.get(code)


def format_rich_error(code: str, message: str, location: str = "") -> str:
    explanation = get_explanation(code)
    if not explanation:
        loc_str = f" at {location}" if location else ""
        return f"{code}: {message}{loc_str}"

    lines = [
        f"{code}: {explanation.title}",
        f"  {message}",
        "",
        f"  How to fix:",
        f"    Before: {explanation.example_before}",
        f"    After:  {explanation.example_after}",
    ]

    if explanation.doc_link:
        lines.append(f"  Docs: {explanation.doc_link}")

    if location:
        lines.append(f"  Location: {location}")

    return "\n".join(lines)
