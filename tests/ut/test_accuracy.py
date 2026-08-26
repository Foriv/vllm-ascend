# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path
from unittest import mock

import pytest
import torch

from tests.accuracy import (
    DEFAULT_DTYPE_TOLERANCES,
    AccuracyTolerance,
    assert_close,
    assert_nrmse_close,
    get_default_tolerance,
)

TESTS_ROOT = Path(__file__).resolve().parents[1]
TRITON_TEST_DIR = TESTS_ROOT / "e2e/nightly/single_node/ops/singlecard_ops/triton"
EXTRA_TRITON_ACCURACY_TESTS = (
    TESTS_ROOT / "ut/ops/a2/test_gdn_chunk_meta.py",
    TESTS_ROOT / "ut/sample/a2/test_gumbel_sampling.py",
)
_LEGACY_ASSERT_FUNCTIONS = {
    "numpy.testing.assert_allclose",
    "torch.testing.assert_close",
}
_LEGACY_BOOLEAN_FUNCTIONS = {
    "numpy.allclose",
    "numpy.isclose",
    "torch.allclose",
    "torch.equal",
    "torch.isclose",
}
_SHARED_ASSERT_CLOSE = "tests.accuracy.assert_close"
_SHARED_ASSERT_NRMSE_CLOSE = "tests.accuracy.assert_nrmse_close"


def _qualified_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                local_name = imported.asname or imported.name.split(".", maxsplit=1)[0]
                aliases[local_name] = imported.name if imported.asname else local_name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                if imported.name == "*":
                    continue
                aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"
    return aliases


def _resolved_name(node: ast.expr, aliases: dict[str, str]) -> str | None:
    qualified_name = _qualified_name(node)
    if not qualified_name:
        return None
    root, separator, remainder = qualified_name.partition(".")
    resolved_root = aliases.get(root, root)
    return f"{resolved_root}.{remainder}" if separator else resolved_root


def _assert_context(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> tuple[bool, bool]:
    current: ast.AST = call
    negated = False
    while current in parents:
        current = parents[current]
        if isinstance(current, ast.UnaryOp) and isinstance(current.op, ast.Not):
            negated = not negated
        if isinstance(current, ast.Assert):
            return True, negated
    return False, False


def _reduced_tensor_comparison(call: ast.Call, call_name: str | None) -> tuple[str, ast.Compare] | None:
    reduction = None
    comparison = None
    if call_name in {"torch.all", "torch.any"} and call.args:
        reduction = call_name.rsplit(".", maxsplit=1)[-1]
        comparison = call.args[0]
    elif isinstance(call.func, ast.Attribute) and call.func.attr in {"all", "any"}:
        reduction = call.func.attr
        comparison = call.func.value
    if reduction is not None and isinstance(comparison, ast.Compare):
        return reduction, comparison
    return None


def _legacy_accuracy_violations(tree: ast.AST, filename: str) -> list[str]:
    aliases = _import_aliases(tree)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _resolved_name(node.func, aliases)
        if call_name == _SHARED_ASSERT_NRMSE_CLOSE:
            keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
            if not {"name", "reason"} <= keywords:
                violations.append(f"{filename}:{node.lineno} uses an undocumented NRMSE comparison")
            continue
        if call_name == _SHARED_ASSERT_CLOSE:
            keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
            if "tolerance" in keywords and not {"name", "reason"} <= keywords:
                violations.append(f"{filename}:{node.lineno} uses an undocumented tolerance override")
            continue
        if call_name and call_name.rsplit(".", maxsplit=1)[-1] in {"assert_close", "assert_allclose"}:
            violations.append(f"{filename}:{node.lineno} uses an unapproved accuracy assertion")
            continue
        if call_name in _LEGACY_ASSERT_FUNCTIONS:
            violations.append(f"{filename}:{node.lineno} uses {call_name}")
            continue
        is_legacy_boolean = call_name in _LEGACY_BOOLEAN_FUNCTIONS
        is_tensor_comparison_method = call_name is not None and call_name.rsplit(".", maxsplit=1)[-1] in {
            "allclose",
            "equal",
            "isclose",
        }
        reduced_comparison = _reduced_tensor_comparison(node, call_name)
        if not (is_legacy_boolean or is_tensor_comparison_method or reduced_comparison):
            continue
        in_assert, negated = _assert_context(node, parents)
        if reduced_comparison:
            reduction, comparison = reduced_comparison
            has_equal = any(isinstance(operator, ast.Eq) for operator in comparison.ops)
            has_not_equal = any(isinstance(operator, ast.NotEq) for operator in comparison.ops)
            asserts_equality = (reduction == "all" and has_equal and not negated) or (
                reduction == "any" and has_not_equal and negated
            )
            if asserts_equality:
                violations.append(f"{filename}:{node.lineno} bypasses the shared accuracy assertion")
            continue
        if not in_assert or not negated:
            violations.append(f"{filename}:{node.lineno} bypasses the shared accuracy assertion")
    return violations


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float16, AccuracyTolerance(rtol=1e-3, atol=1e-5)),
        (torch.bfloat16, AccuracyTolerance(rtol=1.6e-2, atol=1e-5)),
        (torch.float32, AccuracyTolerance(rtol=1.3e-6, atol=1e-5)),
        (torch.float64, AccuracyTolerance(rtol=1e-7, atol=1e-7)),
    ],
)
def test_get_default_tolerance_uses_frozen_dtype_policy(dtype, expected):
    assert get_default_tolerance(dtype) == expected


@pytest.mark.parametrize("dtype", [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
def test_get_default_tolerance_requires_exact_integer_comparison(dtype):
    assert get_default_tolerance(dtype) == AccuracyTolerance(rtol=0.0, atol=0.0)


def test_default_triton_tolerances_are_immutable():
    with pytest.raises(TypeError):
        DEFAULT_DTYPE_TOLERANCES[torch.float32] = AccuracyTolerance(rtol=1.0, atol=1.0)  # type: ignore[index]


@pytest.mark.parametrize(
    ("rtol", "atol"),
    [
        (-1.0, 0.0),
        (0.0, -1.0),
        (float("nan"), 0.0),
        (0.0, float("inf")),
    ],
)
def test_tolerance_rejects_invalid_values(rtol, atol):
    with pytest.raises(ValueError):
        AccuracyTolerance(rtol=rtol, atol=atol)


def test_get_default_tolerance_rejects_unsupported_float_dtype():
    with pytest.raises(ValueError, match="No default accuracy tolerance"):
        get_default_tolerance(torch.complex64)


def test_assert_close_uses_policy_dtype_after_promotion():
    actual = torch.ones(2, dtype=torch.float32)
    expected = torch.ones(2, dtype=torch.float32)

    with mock.patch("tests.accuracy.torch.testing.assert_close") as torch_assert_close:
        assert_close(actual, expected, policy_dtype=torch.bfloat16)

    torch_assert_close.assert_called_once()
    assert_close_kwargs = torch_assert_close.call_args.kwargs
    assert assert_close_kwargs["rtol"] == 1.6e-2
    assert assert_close_kwargs["atol"] == 1e-5


def test_assert_close_accepts_algorithm_override():
    actual = torch.ones(2)
    expected = torch.ones(2)
    tolerance = AccuracyTolerance(rtol=2e-2, atol=5e-2)

    with mock.patch("tests.accuracy.torch.testing.assert_close") as torch_assert_close:
        assert_close(
            actual,
            expected,
            tolerance=tolerance,
            name="example",
            reason="validated accumulation error",
        )

    assert_close_kwargs = torch_assert_close.call_args.kwargs
    assert assert_close_kwargs["rtol"] == tolerance.rtol
    assert assert_close_kwargs["atol"] == tolerance.atol
    assert "example" in assert_close_kwargs["msg"]("details")
    assert "validated accumulation error" in assert_close_kwargs["msg"]("details")


def test_assert_close_exact_mode_uses_zero_tolerances():
    actual = torch.tensor([1], dtype=torch.int32)
    expected = actual.clone()

    with mock.patch("tests.accuracy.torch.testing.assert_close") as torch_assert_close:
        assert_close(actual, expected, exact=True)

    assert_close_kwargs = torch_assert_close.call_args.kwargs
    assert assert_close_kwargs["rtol"] == 0.0
    assert assert_close_kwargs["atol"] == 0.0


def test_assert_close_uses_implicit_exact_policy_for_integer_tensors():
    actual = torch.tensor([1], dtype=torch.int32)
    assert_close(actual, actual.clone())

    with pytest.raises(AssertionError):
        assert_close(actual, torch.tensor([2], dtype=torch.int32))


def test_assert_close_rejects_conflicting_exact_override():
    actual = torch.ones(2)

    with pytest.raises(ValueError, match="mutually exclusive"):
        assert_close(
            actual,
            actual,
            tolerance=AccuracyTolerance(rtol=0.0, atol=0.0),
            exact=True,
        )


@pytest.mark.parametrize(
    ("name", "reason", "missing_field"),
    [
        (None, "validated accumulation error", "name"),
        ("example", None, "reason"),
    ],
)
def test_assert_close_requires_documented_override(name, reason, missing_field):
    actual = torch.ones(2)

    with pytest.raises(ValueError, match=missing_field):
        assert_close(
            actual,
            actual,
            tolerance=AccuracyTolerance(rtol=2e-2, atol=5e-2),
            name=name,
            reason=reason,
        )


def test_assert_close_uses_expected_as_relative_error_denominator():
    actual = torch.tensor([9.0], dtype=torch.float64)
    expected = torch.tensor([10.0], dtype=torch.float64)
    tolerance = AccuracyTolerance(rtol=0.1, atol=0.0)

    assert_close(
        actual,
        expected,
        tolerance=tolerance,
        name="asymmetric-boundary",
        reason="locks the actual-expected argument contract",
    )
    with pytest.raises(AssertionError):
        assert_close(
            expected,
            actual,
            tolerance=tolerance,
            name="reversed-asymmetric-boundary",
            reason="proves the relative-error denominator is the expected tensor",
        )


def test_assert_close_rejects_paired_nan_by_default():
    actual = torch.tensor([float("nan")])

    with pytest.raises(AssertionError):
        assert_close(actual, actual.clone())

    assert_close(actual, actual.clone(), equal_nan=True)


def test_assert_nrmse_close_supports_absolute_guard_and_relative_bound():
    expected = torch.tensor([1.0, 1.0])

    assert_nrmse_close(
        expected + 5e-7,
        expected,
        max_nrmse=0.0,
        name="absolute-guard",
        reason="exercise the absolute-error fast path",
    )
    assert_nrmse_close(
        expected + 1e-2,
        expected,
        max_nrmse=2e-2,
        name="nrmse-bound",
        reason="exercise the normalized-RMSE path",
    )
    with pytest.raises(AssertionError, match="nrmse"):
        assert_nrmse_close(
            expected + 2e-1,
            expected,
            max_nrmse=1e-1,
            name="nrmse-failure",
            reason="exercise a normalized-RMSE failure",
        )


def test_assert_nrmse_close_rejects_invalid_inputs():
    expected = torch.ones(2)
    with pytest.raises(AssertionError, match="shape mismatch"):
        assert_nrmse_close(
            torch.ones(3),
            expected,
            max_nrmse=1e-2,
            name="shape",
            reason="validate input contracts",
        )
    with pytest.raises(AssertionError, match="NaN detected in actual"):
        assert_nrmse_close(
            torch.tensor([float("nan"), 1.0]),
            expected,
            max_nrmse=1e-2,
            name="nan",
            reason="validate input contracts",
        )
    with pytest.raises(ValueError, match="max_nrmse"):
        assert_nrmse_close(
            expected,
            expected,
            max_nrmse=float("nan"),
            name="invalid-threshold",
            reason="validate threshold contracts",
        )
    with pytest.raises(ValueError, match="abs_guard"):
        assert_nrmse_close(
            expected,
            expected,
            max_nrmse=1e-2,
            abs_guard=-1.0,
            name="invalid-guard",
            reason="validate threshold contracts",
        )
    with pytest.raises(ValueError, match="non-empty name"):
        assert_nrmse_close(
            expected,
            expected,
            max_nrmse=1e-2,
            name="",
            reason="validate diagnostic contracts",
        )
    with pytest.raises(TypeError, match="reason"):
        assert_nrmse_close(expected, expected, max_nrmse=1e-2, name="missing-reason")


def test_assert_nrmse_close_promotes_low_precision_reductions():
    expected = torch.full((1024,), 300.0, dtype=torch.float16)
    actual = torch.full((1024,), 330.0, dtype=torch.float16)

    with pytest.raises(AssertionError, match="nrmse"):
        assert_nrmse_close(
            actual,
            expected,
            max_nrmse=5e-2,
            abs_guard=0.0,
            name="fp16-square-overflow",
            reason="prove NRMSE is accumulated in FP32",
        )


def test_assert_nrmse_close_handles_empty_and_rejects_mixed_dtype():
    empty = torch.empty(0)
    assert_nrmse_close(
        empty,
        empty.clone(),
        max_nrmse=0.0,
        name="empty",
        reason="empty tensors with the same contract are equal",
    )

    with pytest.raises(AssertionError, match="dtype mismatch"):
        assert_nrmse_close(
            torch.ones(2, dtype=torch.float16),
            torch.ones(2, dtype=torch.float32),
            max_nrmse=1e-2,
            name="mixed-dtype",
            reason="require an explicit comparison dtype",
        )


def test_assert_nrmse_close_preserves_float64_reduction_precision():
    expected = torch.tensor([1.0], dtype=torch.float64)
    actual = torch.tensor([1.0 + 1e-10], dtype=torch.float64)

    with pytest.raises(AssertionError, match="nrmse"):
        assert_nrmse_close(
            actual,
            expected,
            max_nrmse=0.0,
            abs_guard=0.0,
            name="float64-reduction",
            reason="do not discard differences representable in FP64",
        )


@pytest.mark.parametrize(
    ("actual", "expected", "message"),
    [
        (torch.ones(2), torch.tensor([float("nan"), 1.0]), "NaN detected in expected"),
        (torch.tensor([float("inf"), 1.0]), torch.ones(2), "infinity detected in actual"),
        (torch.ones(2), torch.tensor([float("inf"), 1.0]), "infinity detected in expected"),
    ],
)
def test_assert_nrmse_close_rejects_non_finite_inputs(actual, expected, message):
    with pytest.raises(AssertionError, match=message):
        assert_nrmse_close(
            actual,
            expected,
            max_nrmse=1e-2,
            name="non-finite",
            reason="NRMSE is undefined for non-finite values",
        )


def test_assert_nrmse_close_rejects_device_and_non_floating_inputs():
    with pytest.raises(AssertionError, match="device mismatch"):
        assert_nrmse_close(
            torch.ones(2),
            torch.ones(2, device="meta"),
            max_nrmse=1e-2,
            name="mixed-device",
            reason="require an explicit comparison device",
        )

    integers = torch.ones(2, dtype=torch.int32)
    with pytest.raises(TypeError, match="floating-point"):
        assert_nrmse_close(
            integers,
            integers.clone(),
            max_nrmse=1e-2,
            name="integer-nrmse",
            reason="NRMSE is only defined for floating-point accuracy",
        )


@pytest.mark.parametrize(
    "source",
    [
        "import torch as t\nt.testing.assert_close(actual, expected)",
        "from torch import equal as same\nassert same(actual, expected)",
        "import numpy.testing as npt\nnpt.assert_allclose(actual, expected)",
        "assert (actual == expected).all()",
        "import torch\nassert torch.isclose(actual, expected).all()",
        "import torch\nassert torch.all(actual == expected)",
        "import torch\nassert not torch.any(actual != expected)",
        "def assert_close(actual, expected): pass\nassert_close(actual, expected)",
    ],
)
def test_accuracy_guard_detects_aliases_and_exact_reductions(source):
    tree = ast.parse(source)
    assert _legacy_accuracy_violations(tree, "example.py")


def test_accuracy_guard_allows_negative_behavior_checks():
    tree = ast.parse("import torch as t\nassert not t.equal(actual, expected)\nassert not t.allclose(actual, expected)")
    assert not _legacy_accuracy_violations(tree, "example.py")


def test_accuracy_guard_recognizes_the_shared_backend_independent_api():
    tree = ast.parse(
        "from tests.accuracy import assert_close as check\n"
        "check(actual, expected)\n"
        "from tests.accuracy import assert_nrmse_close\n"
        "assert_nrmse_close(actual, expected, max_nrmse=0.005, name='op', reason='accumulation')"
    )
    assert not _legacy_accuracy_violations(tree, "example.py")


def test_triton_tests_use_the_shared_accuracy_assertion():
    assert TRITON_TEST_DIR.is_dir(), f"Triton test directory does not exist: {TRITON_TEST_DIR}"
    nightly_tests = sorted(TRITON_TEST_DIR.rglob("test_*.py"))
    assert nightly_tests, f"No Triton accuracy tests found under {TRITON_TEST_DIR}"
    missing_extra_tests = [path for path in EXTRA_TRITON_ACCURACY_TESTS if not path.is_file()]
    assert not missing_extra_tests, f"Missing Triton accuracy tests: {missing_extra_tests}"

    violations = []
    for path in [*nightly_tests, *EXTRA_TRITON_ACCURACY_TESTS]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_legacy_accuracy_violations(tree, path.name))

    assert not violations, "Use assert_close for Triton accuracy checks:\n" + "\n".join(violations)
