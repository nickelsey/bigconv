"""Tests for bigconv.ops.conv_transpose3d."""

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from bigconv.ops.conv_transpose3d import (
    _reference_conv_transpose3d,
    _validate_conv_transpose_args,
    conv_transpose3d,
)
from bigconv.testing import distributed, run_distributed


def _good_x():
    return torch.randn(3, 4, 4, 4)


def _good_w():
    return torch.randn(3, 6, 3, 3, 3)


def _validate(
    *,
    x=None,
    w=None,
    bias=None,
    stride=(2, 2, 2),
    padding=(1, 1, 1),
    output_padding=(1, 1, 1),
    mesh=None,
):
    _validate_conv_transpose_args(
        x if x is not None else _good_x(),
        w if w is not None else _good_w(),
        bias,
        stride,
        padding,
        output_padding,
        mesh,
    )


def test_validator_accepts_good_args():
    _validate()


def test_validator_rejects_wrong_x_ndim():
    with pytest.raises(ValueError, match="x must be 4D"):
        _validate(x=torch.randn(3, 4, 4, 4, 1))


def test_validator_rejects_wrong_weight_ndim():
    with pytest.raises(ValueError, match="weight must be 5D"):
        _validate(w=torch.randn(3, 6, 3, 3))


def test_validator_rejects_mismatched_c_in():
    with pytest.raises(ValueError, match="C_in=4 doesn't match"):
        _validate(x=torch.randn(4, 4, 4, 4))


def test_validator_rejects_bad_bias_shape():
    with pytest.raises(ValueError, match=r"bias must have shape \(6,\)"):
        _validate(bias=torch.zeros(5))


def test_validator_rejects_bad_stride():
    with pytest.raises(ValueError, match="stride_x=3"):
        _validate(stride=(3, 2, 2))


def test_validator_rejects_non_same_padding_x():
    with pytest.raises(ValueError, match=r"padding_x=0 must equal \(K_x - 1\) // 2 = 1"):
        _validate(padding=(0, 1, 1))


def test_validator_rejects_bad_output_padding_x():
    with pytest.raises(ValueError, match="output_padding_x=0 must equal stride_x - 1 = 1"):
        _validate(output_padding=(0, 1, 1))


@distributed(world_size=4)
def test_validator_rejects_multi_dim_mesh(rank, world_size):
    mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "sp"))
    with pytest.raises(ValueError, match="1D DeviceMesh"):
        _validate(mesh=mesh_2d)


@distributed(world_size=2)
def test_validator_rejects_non_uniform_x_local(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    x_local = 4 if rank == 0 else 5
    with pytest.raises(ValueError, match="X_local must be uniform"):
        _validate(x=torch.randn(3, x_local, 4, 4), mesh=mesh)


@pytest.mark.parametrize("stride", [1, 2])
def test_reference_matches_torch_conv_transpose3d(stride):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 4, 4)
    w = torch.randn(3, 6, 3, 3, 3)
    b = torch.randn(6)
    output_padding = stride - 1

    got = _reference_conv_transpose3d(
        x,
        w,
        b,
        stride=stride,
        padding=1,
        output_padding=output_padding,
    )
    expected = F.conv_transpose3d(
        x.unsqueeze(0),
        w,
        b,
        stride=stride,
        padding=1,
        output_padding=output_padding,
    ).squeeze(0)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("stride", [1, 2])
def test_conv_transpose3d_mesh_none_matches_reference(stride):
    torch.manual_seed(0)
    x = torch.randn(3, 4, 4, 4)
    w = torch.randn(3, 6, 3, 3, 3)
    b = torch.randn(6)
    output_padding = stride - 1

    got = conv_transpose3d(x, w, b, stride=stride, padding=1, output_padding=output_padding)
    expected = _reference_conv_transpose3d(
        x,
        w,
        b,
        stride=stride,
        padding=1,
        output_padding=output_padding,
    )
    torch.testing.assert_close(got, expected)


def test_conv_transpose3d_mesh_none_asymmetric_kernel():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5, 4)
    w = torch.randn(3, 6, 3, 5, 1)
    b = torch.randn(6)

    got = conv_transpose3d(
        x,
        w,
        b,
        stride=(2, 1, 1),
        padding=(1, 2, 0),
        output_padding=(1, 0, 0),
    )
    expected = _reference_conv_transpose3d(
        x,
        w,
        b,
        stride=(2, 1, 1),
        padding=(1, 2, 0),
        output_padding=(1, 0, 0),
    )
    torch.testing.assert_close(got, expected)


def _conv_transpose_distributed_worker(
    rank, world_size, global_x, weight, bias, stride, padding, output_padding
):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous()
    out = conv_transpose3d(
        x_local,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        mesh=mesh,
    )
    return out.detach()


@pytest.mark.parametrize("stride", [1, 2])
def test_conv_transpose3d_distributed_matches_single_process_reference(stride):
    torch.manual_seed(0)
    world_size = 2
    X_total = 8
    global_x = torch.randn(3, X_total, 4, 4)
    weight = torch.randn(3, 6, 3, 3, 3)
    bias = torch.randn(6)
    output_padding = stride - 1

    ref_out = _reference_conv_transpose3d(
        global_x,
        weight,
        bias,
        stride=stride,
        padding=1,
        output_padding=output_padding,
    )
    results = run_distributed(
        _conv_transpose_distributed_worker,
        world_size=world_size,
        global_x=global_x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=1,
        output_padding=output_padding,
    )

    X_local_out = (X_total // world_size) * stride
    for r in range(world_size):
        expected = ref_out[:, r * X_local_out : (r + 1) * X_local_out]
        torch.testing.assert_close(results[r], expected, msg=lambda m, r=r: f"rank {r}: {m}")


def _conv_transpose_distributed_backward_worker(rank, world_size, global_x, weight, bias, stride):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous().requires_grad_(True)
    w_local = weight.clone().requires_grad_(True)
    b_local = bias.clone().requires_grad_(True) if bias is not None else None

    out = conv_transpose3d(
        x_local,
        w_local,
        b_local,
        stride=stride,
        padding=1,
        output_padding=stride - 1,
        mesh=mesh,
    )
    (out * out).sum().backward()
    assert x_local.grad is not None
    assert w_local.grad is not None
    if b_local is not None:
        assert b_local.grad is not None

    return {
        "grad_x": x_local.grad.detach().clone(),
        "grad_w": w_local.grad.detach().clone(),
        "grad_b": b_local.grad.detach().clone() if b_local is not None else None,
    }


@pytest.mark.parametrize("stride", [1, 2])
def test_conv_transpose3d_distributed_backward_matches_reference(stride):
    torch.manual_seed(0)
    world_size = 2
    X_total = 8
    global_x = torch.randn(3, X_total, 4, 4, dtype=torch.float64)
    weight = torch.randn(3, 6, 3, 3, 3, dtype=torch.float64)
    bias = torch.randn(6, dtype=torch.float64)

    x_ref = global_x.clone().requires_grad_(True)
    w_ref = weight.clone().requires_grad_(True)
    b_ref = bias.clone().requires_grad_(True)
    ref_out = _reference_conv_transpose3d(
        x_ref,
        w_ref,
        b_ref,
        stride=stride,
        padding=1,
        output_padding=stride - 1,
    )
    (ref_out * ref_out).sum().backward()
    assert x_ref.grad is not None
    assert w_ref.grad is not None
    assert b_ref.grad is not None

    results = run_distributed(
        _conv_transpose_distributed_backward_worker,
        world_size=world_size,
        global_x=global_x,
        weight=weight,
        bias=bias,
        stride=stride,
    )

    X_local = X_total // world_size
    for r in range(world_size):
        expected = x_ref.grad[:, r * X_local : (r + 1) * X_local]
        torch.testing.assert_close(
            results[r]["grad_x"],
            expected,
            msg=lambda m, r=r: f"grad_x rank {r}: {m}",
        )

    sum_grad_w = sum(r["grad_w"] for r in results)
    torch.testing.assert_close(sum_grad_w, w_ref.grad)

    sum_grad_b = sum(r["grad_b"] for r in results)
    torch.testing.assert_close(sum_grad_b, b_ref.grad)


def test_conv_transpose3d_mesh_none_backward_no_bias():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 6, 3, 3, 3, dtype=torch.float64, requires_grad=True)

    out = conv_transpose3d(x, w, None, stride=2, padding=1, output_padding=1)
    out.sum().backward()
    assert x.grad is not None
    assert w.grad is not None


@distributed(world_size=2)
def test_conv_transpose3d_opcheck(rank, world_size):
    torch.manual_seed(rank)
    x = torch.randn(3, 4, 4, 4, requires_grad=True)
    w = torch.randn(3, 6, 3, 3, 3, requires_grad=True)
    b = torch.randn(6, requires_grad=True)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    group_name = mesh.get_group().group_name

    torch.library.opcheck(
        torch.ops.bigconv.conv_transpose3d.default,
        (x, w, b, [2, 2, 2], [1, 1, 1], [1, 1, 1], group_name),
        test_utils=("test_autograd_registration",),
    )
