import re
from dataclasses import dataclass, field

VALID_TASKS = [
    "qc",
    "normalization",
    "dimensionality_reduction",
    "clustering",
    "cell_typing",
    "differential_expression",
    "spatial_analysis",
]

VALID_KITS = [
    "xenium",
    "visium",
    "merfish",
    "vizgen",
    "cosmx",
    "seeker",
    "takara",
    "atlasxomics",
    "curio",
]

VALID_TIME_HORIZONS = ["small"]

VALID_EVAL_TYPES = ["scientific", "procedural", "observational"]

VALID_TOLERANCE_TYPES = ["absolute", "relative", "min", "max"]

DATA_NODE_PATTERN = re.compile(r"^latch://\d+\.(account|node)(/.*)?$")

MULTIPLE_CHOICE_PLACEHOLDER = "<letter>"
NUMERIC_PLACEHOLDER = "<number>"

GRADER_CONFIGS: dict[str, dict] = {
    "numeric_tolerance": {
        "required": ["ground_truth", "tolerances"],
        "recognized": {"ground_truth", "tolerances", "tolerance"},
        "answer_fields_from": "ground_truth",
    },
    "multiple_choice": {
        "required_any": [["correct_answer", "correct_answers"]],
        "recognized": {"correct_answer", "correct_answers"},
        "answer_fields": ["answer"],
    },
    "distribution_comparison": {
        "required": ["ground_truth", "tolerances"],
        "recognized": {"ground_truth", "tolerances"},
        "answer_fields": ["cell_type_distribution"],
        "answer_fields_optional": ["total_cells"],
    },
    "marker_gene_precision_recall": {
        "required": ["canonical_markers", "scoring", "answer_field"],
        "recognized": {"canonical_markers", "ground_truth_labels", "scoring", "answer_field"},
        "answer_field_from_config": "answer_field",
        "answer_field_default": "top_marker_genes",
    },
    "label_set_jaccard": {
        "required": ["ground_truth_labels"],
        "recognized": {"ground_truth_labels", "scoring", "answer_field"},
        "answer_field_from_config": "answer_field",
        "answer_field_default": "cell_types_predicted",
    },
    "jaccard_label_set": {
        "required": ["ground_truth_labels"],
        "recognized": {"ground_truth_labels", "scoring", "answer_field"},
        "answer_field_from_config": "answer_field",
        "answer_field_default": "cell_types_predicted",
    },
    "marker_gene_separation": {
        "required": ["scoring"],
        "recognized": {"scoring"},
        "answer_fields": ["per_gene_stats", "mean_auroc"],
    },
    "spatial_adjacency": {
        "required": ["scoring"],
        "recognized": {"scoring"},
        "answer_fields": [
            "median_ic_to_pc_um",
            "p90_ic_to_pc_um",
            "pct_ic_within_15um",
            "pct_ic_mixed_within_55um",
            "adjacency_pass",
        ],
    },
}

VALID_GRADER_TYPES = list(GRADER_CONFIGS.keys())

ALLOWED_TOP_LEVEL_FIELDS = {"id", "task", "data_node", "grader", "notes", "metadata"}

ALLOWED_METADATA_FIELDS = {"task", "kit", "time_horizon", "eval_type", "timeout_s"}

ALLOWED_GRADER_FIELDS = {"type", "config"}


@dataclass
class LintIssue:
    level: str  # "error", "warning", "info"
    code: str
    message: str
    location: str = ""

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.level.upper()}] {self.code}: {self.message}{loc}"


@dataclass
class LintResult:
    file_path: str
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "warning")
