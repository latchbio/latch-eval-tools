from dataclasses import dataclass


@dataclass(frozen=True)
class ListCardinality:
    expected_count: int | None
    submitted_count: int
    unique_count: int
    passed: bool
    configuration_error: str | None = None


def check_list_cardinality(
    items: list[str], expected_count_value: object
) -> ListCardinality:
    """Validate an optional exact-length contract for a normalized string list.

    When ``expected_count`` is configured, both the submitted length and the
    number of unique items must match. This prevents duplicates from satisfying
    a prompt such as "return exactly 10 genes" while leaving existing
    variable-length graders unchanged when the field is omitted.
    """

    submitted_count = len(items)
    unique_count = len(set(items))

    if expected_count_value is None:
        return ListCardinality(
            expected_count=None,
            submitted_count=submitted_count,
            unique_count=unique_count,
            passed=True,
        )

    if (
        not isinstance(expected_count_value, int)
        or isinstance(expected_count_value, bool)
        or expected_count_value < 0
    ):
        return ListCardinality(
            expected_count=None,
            submitted_count=submitted_count,
            unique_count=unique_count,
            passed=False,
            configuration_error=(
                "expected_count must be a non-negative integer, got "
                f"{expected_count_value!r}"
            ),
        )

    return ListCardinality(
        expected_count=expected_count_value,
        submitted_count=submitted_count,
        unique_count=unique_count,
        passed=(
            submitted_count == expected_count_value
            and unique_count == expected_count_value
        ),
    )
