import re

from .schema import (
    VALID_TASKS,
    VALID_KITS,
    VALID_TIME_HORIZONS,
    VALID_EVAL_TYPES,
    VALID_GRADER_TYPES,
    VALID_TOLERANCE_TYPES,
    GRADER_CONFIGS,
    DATA_NODE_PATTERN,
    ALLOWED_TOP_LEVEL_FIELDS,
    ALLOWED_METADATA_FIELDS,
    ALLOWED_GRADER_FIELDS,
    MULTIPLE_CHOICE_PLACEHOLDER,
    LintIssue,
)


def validate_required_fields(data: dict) -> list[LintIssue]:
    issues = []

    if "id" not in data:
        issues.append(LintIssue("error", "E001", "Missing required field: id"))
    elif not isinstance(data["id"], str) or not data["id"].strip():
        issues.append(LintIssue("error", "E002", "Field 'id' must be a non-empty string"))

    if "task" not in data:
        issues.append(LintIssue("error", "E003", "Missing required field: task"))
    elif not isinstance(data["task"], str) or not data["task"].strip():
        issues.append(LintIssue("error", "E004", "Field 'task' must be a non-empty string"))

    if "metadata" not in data:
        issues.append(LintIssue("error", "E005", "Missing required field: metadata"))
    elif not isinstance(data["metadata"], dict):
        issues.append(LintIssue("error", "E006", "Field 'metadata' must be an object"))

    return issues


def validate_metadata(data: dict) -> list[LintIssue]:
    issues = []
    metadata = data.get("metadata")

    if not isinstance(metadata, dict):
        return issues

    if "task" not in metadata:
        issues.append(LintIssue("error", "E010", "Missing required field: metadata.task"))
    elif metadata["task"] not in VALID_TASKS:
        issues.append(LintIssue(
            "error", "E011",
            f"Invalid metadata.task: '{metadata['task']}'. Must be one of: {VALID_TASKS}"
        ))

    if "kit" not in metadata:
        issues.append(LintIssue("error", "E012", "Missing required field: metadata.kit"))
    elif metadata["kit"] not in VALID_KITS:
        issues.append(LintIssue(
            "error", "E013",
            f"Invalid metadata.kit: '{metadata['kit']}'. Must be one of: {VALID_KITS}"
        ))

    if "time_horizon" not in metadata:
        issues.append(LintIssue("error", "E014", "Missing required field: metadata.time_horizon"))
    elif metadata["time_horizon"] not in VALID_TIME_HORIZONS:
        issues.append(LintIssue(
            "error", "E015",
            f"Invalid metadata.time_horizon: '{metadata['time_horizon']}'. Must be one of: {VALID_TIME_HORIZONS}"
        ))

    if "eval_type" not in metadata:
        issues.append(LintIssue(
            "warning", "W001",
            f"Missing metadata.eval_type. Consider adding one of: {VALID_EVAL_TYPES}"
        ))
    elif metadata["eval_type"] not in VALID_EVAL_TYPES:
        issues.append(LintIssue(
            "error", "E016",
            f"Invalid metadata.eval_type: '{metadata['eval_type']}'. Must be one of: {VALID_EVAL_TYPES}"
        ))

    return issues


def validate_data_node(data: dict) -> list[LintIssue]:
    issues = []
    data_node = data.get("data_node")

    if data_node is None:
        return issues

    def check_node(node: str, location: str) -> list[LintIssue]:
        if not isinstance(node, str):
            return [LintIssue("error", "E020", f"data_node must be string, got {type(node).__name__}", location)]
        if not DATA_NODE_PATTERN.match(node):
            return [LintIssue(
                "error", "E021",
                f"Invalid data_node format: '{node}'. Expected: latch://<id>.(account|node)/<path>",
                location
            )]
        return []

    if isinstance(data_node, str):
        issues.extend(check_node(data_node, "data_node"))
    elif isinstance(data_node, list):
        for i, node in enumerate(data_node):
            issues.extend(check_node(node, f"data_node[{i}]"))
    else:
        issues.append(LintIssue(
            "error", "E022",
            f"data_node must be string or list, got {type(data_node).__name__}"
        ))

    return issues


def validate_task_answer_format(data: dict) -> list[LintIssue]:
    issues = []
    task = data.get("task", "")
    grader_type = data.get("grader", {}).get("type")

    if "<EVAL_ANSWER>" not in task:
        issues.append(LintIssue(
            "warning", "W010",
            "Task description does not contain <EVAL_ANSWER> format specification"
        ))
    elif "</EVAL_ANSWER>" not in task:
        issues.append(LintIssue(
            "warning", "W011",
            "Task description has <EVAL_ANSWER> but missing closing </EVAL_ANSWER> tag"
        ))
    else:
        task_lower = task.lower()
        has_return_exactly = "return exactly" in task_lower or "respond exactly" in task_lower
        if not has_return_exactly:
            issues.append(LintIssue(
                "warning", "W012",
                "Task has <EVAL_ANSWER> but missing 'Return EXACTLY:' instruction before it"
            ))

        if grader_type == "multiple_choice":
            answer_pattern = re.search(r'"answer"\s*:\s*"([^"]*)"', task)
            if answer_pattern:
                placeholder = answer_pattern.group(1)
                if placeholder != MULTIPLE_CHOICE_PLACEHOLDER:
                    issues.append(LintIssue(
                        "warning", "W013",
                        f"Multiple choice answer placeholder should be '{MULTIPLE_CHOICE_PLACEHOLDER}', "
                        f"found '{placeholder}'",
                        "task"
                    ))

    return issues


def validate_grader(data: dict) -> list[LintIssue]:
    issues = []
    grader = data.get("grader")

    if grader is None:
        return issues

    if not isinstance(grader, dict):
        issues.append(LintIssue("error", "E030", f"grader must be object, got {type(grader).__name__}"))
        return issues

    grader_type = grader.get("type")
    if grader_type is None:
        issues.append(LintIssue("error", "E031", "Missing required field: grader.type"))
        return issues

    if grader_type not in VALID_GRADER_TYPES:
        issues.append(LintIssue(
            "error", "E032",
            f"Invalid grader.type: '{grader_type}'. Must be one of: {VALID_GRADER_TYPES}"
        ))
        return issues

    config = grader.get("config")
    if config is None:
        issues.append(LintIssue("error", "E033", "Missing required field: grader.config"))
        return issues

    if not isinstance(config, dict):
        issues.append(LintIssue("error", "E034", f"grader.config must be object, got {type(config).__name__}"))
        return issues

    grader_spec = GRADER_CONFIGS.get(grader_type, {})

    for req_field in grader_spec.get("required", []):
        if req_field not in config:
            if grader_type == "marker_gene_precision_recall" and req_field == "answer_field":
                issues.append(LintIssue(
                    "error", "E037",
                    f"Missing 'answer_field' - specify which JSON field contains the gene list",
                    f"grader.config.{req_field}"
                ))
            else:
                issues.append(LintIssue(
                    "error", "E035",
                    f"Missing required config field for {grader_type}: {req_field}",
                    f"grader.config.{req_field}"
                ))

    for req_any_group in grader_spec.get("required_any", []):
        if not any(f in config for f in req_any_group):
            issues.append(LintIssue(
                "error", "E036",
                f"Missing required config field for {grader_type}: one of {req_any_group}",
                "grader.config"
            ))

    issues.extend(_validate_tolerances(config))
    issues.extend(_validate_unrecognized_config_fields(grader_type, config))
    issues.extend(_validate_config_types(grader_type, config))
    issues.extend(_validate_config_semantics(grader_type, config))
    issues.extend(_validate_config_edge_cases(grader_type, config))

    return issues


def _validate_unrecognized_config_fields(grader_type: str, config: dict) -> list[LintIssue]:
    issues = []
    grader_spec = GRADER_CONFIGS.get(grader_type, {})
    recognized = grader_spec.get("recognized", set())

    if not recognized:
        return issues

    for field in config.keys():
        if field not in recognized:
            issues.append(LintIssue(
                "warning", "W030",
                f"Config field '{field}' is not recognized by {grader_type} grader and will be ignored",
                f"grader.config.{field}"
            ))

    return issues


def _validate_config_types(grader_type: str, config: dict) -> list[LintIssue]:
    issues = []

    if grader_type in ("numeric_tolerance", "distribution_comparison"):
        ground_truth = config.get("ground_truth")
        if ground_truth is not None and not isinstance(ground_truth, dict):
            issues.append(LintIssue(
                "error", "E060",
                f"ground_truth must be object, got {type(ground_truth).__name__}",
                "grader.config.ground_truth"
            ))

    if grader_type in ("label_set_jaccard", "jaccard_label_set", "marker_gene_precision_recall"):
        ground_truth_labels = config.get("ground_truth_labels")
        if ground_truth_labels is not None and not isinstance(ground_truth_labels, list):
            issues.append(LintIssue(
                "error", "E062",
                f"ground_truth_labels must be list, got {type(ground_truth_labels).__name__}",
                "grader.config.ground_truth_labels"
            ))

    if grader_type in ("label_set_jaccard", "jaccard_label_set", "spatial_adjacency",
                       "marker_gene_separation", "marker_gene_precision_recall"):
        scoring = config.get("scoring")
        if scoring is not None and not isinstance(scoring, dict):
            issues.append(LintIssue(
                "error", "E065",
                f"scoring must be object, got {type(scoring).__name__}",
                "grader.config.scoring"
            ))

    return issues


def _validate_config_semantics(grader_type: str, config: dict) -> list[LintIssue]:
    issues = []

    if grader_type == "numeric_tolerance":
        ground_truth = config.get("ground_truth", {})
        tolerances = config.get("tolerances", {})
        if isinstance(ground_truth, dict) and isinstance(tolerances, dict):
            for field_name in ground_truth.keys():
                if field_name not in tolerances:
                    issues.append(LintIssue(
                        "warning", "W070",
                        f"ground_truth field '{field_name}' has no tolerance specified (defaults to 0)",
                        f"grader.config.ground_truth.{field_name}"
                    ))

    issues.extend(_validate_tolerance_values(config))
    issues.extend(_validate_threshold_ranges(grader_type, config))

    return issues


def _validate_tolerance_values(config: dict) -> list[LintIssue]:
    issues = []
    tolerances = config.get("tolerances", {})

    if not isinstance(tolerances, dict):
        return issues

    for field_name, tol_config in tolerances.items():
        if not isinstance(tol_config, dict):
            continue

        value = tol_config.get("value")
        if isinstance(value, (int, float)) and value < 0:
            issues.append(LintIssue(
                "error", "E080",
                f"Tolerance value must be non-negative, got {value}",
                f"grader.config.tolerances.{field_name}.value"
            ))

        lower = tol_config.get("lower")
        if isinstance(lower, (int, float)) and lower < 0:
            issues.append(LintIssue(
                "error", "E080",
                f"Tolerance lower bound must be non-negative, got {lower}",
                f"grader.config.tolerances.{field_name}.lower"
            ))

        upper = tol_config.get("upper")
        if isinstance(upper, (int, float)) and upper < 0:
            issues.append(LintIssue(
                "error", "E080",
                f"Tolerance upper bound must be non-negative, got {upper}",
                f"grader.config.tolerances.{field_name}.upper"
            ))

    return issues


def _validate_threshold_ranges(grader_type: str, config: dict) -> list[LintIssue]:
    issues = []
    scoring = config.get("scoring", {})

    if not isinstance(scoring, dict):
        return issues

    if grader_type in ("label_set_jaccard", "jaccard_label_set"):
        pass_threshold = scoring.get("pass_threshold")
        if isinstance(pass_threshold, (int, float)):
            if pass_threshold < 0 or pass_threshold > 1:
                issues.append(LintIssue(
                    "error", "E081",
                    f"Jaccard pass_threshold must be in [0, 1], got {pass_threshold}",
                    "grader.config.scoring.pass_threshold"
                ))

    if grader_type == "marker_gene_precision_recall":
        pass_thresholds = scoring.get("pass_thresholds", {})
        if isinstance(pass_thresholds, dict):
            for key in ("precision_at_k", "recall_at_k"):
                val = pass_thresholds.get(key)
                if isinstance(val, (int, float)) and (val < 0 or val > 1):
                    issues.append(LintIssue(
                        "error", "E082",
                        f"Precision/recall threshold must be in [0, 1], got {val}",
                        f"grader.config.scoring.pass_thresholds.{key}"
                    ))

    return issues


def _validate_config_edge_cases(grader_type: str, config: dict) -> list[LintIssue]:
    issues = []

    if grader_type == "numeric_tolerance":
        has_tolerance = "tolerance" in config
        has_tolerances = "tolerances" in config
        if has_tolerance and has_tolerances:
            issues.append(LintIssue(
                "warning", "W085",
                "Both 'tolerance' and 'tolerances' present; 'tolerances' will be used",
                "grader.config"
            ))

    if grader_type == "marker_gene_precision_recall":
        has_canonical = "canonical_markers" in config
        has_ground_truth_labels = "ground_truth_labels" in config
        if not has_canonical and has_ground_truth_labels:
            issues.append(LintIssue(
                "warning", "W086",
                "Using 'ground_truth_labels' as fallback for 'canonical_markers'",
                "grader.config"
            ))

    if grader_type == "distribution_comparison":
        ground_truth = config.get("ground_truth", {})
        if isinstance(ground_truth, dict):
            distribution = ground_truth.get("cell_type_distribution", ground_truth)
            if isinstance(distribution, dict):
                percentages = [v for v in distribution.values() if isinstance(v, (int, float))]
                if percentages:
                    total = sum(percentages)
                    if abs(total - 100) > 5:
                        issues.append(LintIssue(
                            "warning", "W080",
                            f"Distribution percentages sum to {total}, expected ~100%",
                            "grader.config.ground_truth"
                        ))

    return issues


def _validate_tolerances(config: dict) -> list[LintIssue]:
    issues = []
    tolerances = config.get("tolerances")

    if tolerances is None:
        return issues

    if not isinstance(tolerances, dict):
        issues.append(LintIssue(
            "error", "E040",
            f"tolerances must be object, got {type(tolerances).__name__}",
            "grader.config.tolerances"
        ))
        return issues

    for field_name, tol_config in tolerances.items():
        if not isinstance(tol_config, dict):
            issues.append(LintIssue(
                "error", "E041",
                f"tolerance config must be object, got {type(tol_config).__name__}",
                f"grader.config.tolerances.{field_name}"
            ))
            continue

        tol_type = tol_config.get("type")
        if tol_type is None:
            issues.append(LintIssue(
                "error", "E042",
                "Missing tolerance type",
                f"grader.config.tolerances.{field_name}.type"
            ))
        elif tol_type not in VALID_TOLERANCE_TYPES:
            issues.append(LintIssue(
                "error", "E043",
                f"Invalid tolerance type: '{tol_type}'. Must be one of: {VALID_TOLERANCE_TYPES}",
                f"grader.config.tolerances.{field_name}.type"
            ))

        has_value = "value" in tol_config
        has_lower = "lower" in tol_config
        has_upper = "upper" in tol_config

        if not has_value and not has_lower and not has_upper:
            issues.append(LintIssue(
                "error", "E044",
                "Missing tolerance: need 'value' or 'lower'/'upper' for asymmetric",
                f"grader.config.tolerances.{field_name}"
            ))
        elif has_value:
            tol_value = tol_config["value"]
            if not isinstance(tol_value, (int, float)):
                issues.append(LintIssue(
                    "error", "E045",
                    f"Tolerance value must be numeric, got {type(tol_value).__name__}",
                    f"grader.config.tolerances.{field_name}.value"
                ))
        if has_lower and not isinstance(tol_config["lower"], (int, float)):
            issues.append(LintIssue(
                "error", "E046",
                f"Tolerance lower must be numeric, got {type(tol_config['lower']).__name__}",
                f"grader.config.tolerances.{field_name}.lower"
            ))
        if has_upper and not isinstance(tol_config["upper"], (int, float)):
            issues.append(LintIssue(
                "error", "E047",
                f"Tolerance upper must be numeric, got {type(tol_config['upper']).__name__}",
                f"grader.config.tolerances.{field_name}.upper"
            ))

    return issues


def validate_answer_fields_match(data: dict) -> list[LintIssue]:
    issues = []
    task = data.get("task", "")
    grader = data.get("grader", {})
    grader_type = grader.get("type")
    config = grader.get("config", {})

    if not grader_type or grader_type not in GRADER_CONFIGS:
        return issues

    grader_spec = GRADER_CONFIGS.get(grader_type, {})
    expected_fields = _get_expected_answer_fields(grader_spec, config)

    if not expected_fields:
        return issues

    task_fields = _extract_answer_fields_from_task(task)

    if not task_fields:
        return issues

    missing_in_task = set(expected_fields) - task_fields
    extra_in_task = task_fields - set(expected_fields)

    optional_fields = set(grader_spec.get("answer_fields_optional", []))
    missing_in_task -= optional_fields

    for field in missing_in_task:
        issues.append(LintIssue(
            "error", "E050",
            f"Grader expects answer field '{field}' but task <EVAL_ANSWER> does not include it",
            "task"
        ))

    for field in extra_in_task:
        issues.append(LintIssue(
            "warning", "W031",
            f"Task <EVAL_ANSWER> has field '{field}' not expected by {grader_type} grader",
            "task"
        ))

    return issues


def _get_expected_answer_fields(grader_spec: dict, config: dict) -> list[str]:
    if "answer_fields" in grader_spec:
        return grader_spec["answer_fields"]

    if "answer_fields_from" in grader_spec:
        source_field = grader_spec["answer_fields_from"]
        source_data = config.get(source_field, {})
        if isinstance(source_data, dict):
            return list(source_data.keys())

    if "answer_field_from_config" in grader_spec:
        config_key = grader_spec["answer_field_from_config"]
        default = grader_spec.get("answer_field_default", "value")
        field_name = config.get(config_key, default)
        return [field_name]

    return []


def _extract_answer_fields_from_task(task: str) -> set[str]:
    match = re.search(r"<EVAL_ANSWER>\s*(\{[^}]+\})\s*</EVAL_ANSWER>", task, re.DOTALL)
    if not match:
        return set()

    json_template = match.group(1)
    field_matches = re.findall(r'"([^"]+)"\s*:', json_template)
    return set(field_matches)


def validate_unknown_fields(data: dict) -> list[LintIssue]:
    issues = []

    for field in data.keys():
        if field not in ALLOWED_TOP_LEVEL_FIELDS:
            issues.append(LintIssue(
                "warning", "W020",
                f"Unknown top-level field: '{field}'",
                field
            ))

    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        for field in metadata.keys():
            if field not in ALLOWED_METADATA_FIELDS:
                issues.append(LintIssue(
                    "warning", "W021",
                    f"Unknown metadata field: '{field}'",
                    f"metadata.{field}"
                ))

    grader = data.get("grader")
    if isinstance(grader, dict):
        for field in grader.keys():
            if field not in ALLOWED_GRADER_FIELDS:
                issues.append(LintIssue(
                    "warning", "W022",
                    f"Unknown grader field: '{field}'",
                    f"grader.{field}"
                ))

    return issues


ALL_VALIDATORS = [
    validate_required_fields,
    validate_metadata,
    validate_data_node,
    validate_task_answer_format,
    validate_grader,
    validate_answer_fields_match,
    validate_unknown_fields,
]
