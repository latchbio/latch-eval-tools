from typing import TypeAlias

from . import (
    all_of,
    distribution_comparison,
    label_set,
    marker_gene_precision_recall,
    marker_gene_separation,
    multiple_choice,
    numeric_range,
    numeric_tolerance,
    spatial_adjacency,
)

AgentAnswer: TypeAlias = (
    all_of.AgentAnswer
    | distribution_comparison.AgentAnswer
    | label_set.AgentAnswer
    | marker_gene_precision_recall.AgentAnswer
    | marker_gene_separation.AgentAnswer
    | multiple_choice.AgentAnswer
    | numeric_range.AgentAnswer
    | numeric_tolerance.AgentAnswer
    | spatial_adjacency.AgentAnswer
)

Config: TypeAlias = (
    all_of.Config
    | distribution_comparison.Config
    | label_set.Config
    | marker_gene_precision_recall.Config
    | marker_gene_separation.Config
    | multiple_choice.Config
    | numeric_range.Config
    | numeric_tolerance.Config
    | spatial_adjacency.Config
)

Spec: TypeAlias = (
    all_of.Spec
    | distribution_comparison.Spec
    | label_set.Spec
    | marker_gene_precision_recall.Spec
    | marker_gene_separation.Spec
    | multiple_choice.Spec
    | numeric_range.Spec
    | numeric_tolerance.Spec
    | spatial_adjacency.Spec
)
