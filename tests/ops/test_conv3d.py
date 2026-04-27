"""Tests for bigconv.ops.conv3d — validator and CPU reference impl."""

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from bigconv.ops.conv3d import (
    _halo_exchange,
    _normalize_tuple,
    _reference_conv3d,
    _validate_conv_args,
    conv3d,
)
from bigconv.testing import distributed, run_distributed


# ---------------------------------------------------------------------------
# _normalize_tuple
# ---------------------------------------------------------------------------


def test_normalize_tuple_scalar():
    assert _normalize_tuple(3, "stride") == (3, 3, 3)


def test_normalize_tuple_tuple():
    assert _normalize_tuple((1, 2, 3), "stride") == (1, 2, 3)


def test_normalize_tuple_rejects_wrong_length():
    with pytest.raises(TypeError, match="3-tuple"):
        _normalize_tuple((1, 2), "stride")  # pyrefly: ignore[bad-argument-type]


def test_normalize_tuple_rejects_non_ints():
    with pytest.raises(TypeError, match="3-tuple"):
        _normalize_tuple((1.0, 2.0, 3.0), "stride")  # pyrefly: ignore[bad-argument-type]


# ---------------------------------------------------------------------------
# Validator (single-process, no mesh)
# ---------------------------------------------------------------------------


def _good_x():
    return torch.randn(3, 8, 4, 4)


def _good_w():
    return torch.randn(6, 3, 3, 3, 3)


def _validate(
    *,
    x=None,
    w=None,
    bias=None,
    stride=(1, 1, 1),
    padding=(1, 1, 1),  # same-padding for default K=3 kernel
    padding_mode="zeros",
    mesh=None,
):
    _validate_conv_args(
        x if x is not None else _good_x(),
        w if w is not None else _good_w(),
        bias,
        stride,
        padding,
        padding_mode,
        mesh,
    )


def test_validator_accepts_good_args():
    _validate()


def test_validator_rejects_wrong_x_ndim():
    with pytest.raises(ValueError, match="x must be 4D"):
        _validate(x=torch.randn(8, 4, 4, 3, 3))


def test_validator_rejects_wrong_weight_ndim():
    with pytest.raises(ValueError, match="weight must be 5D"):
        _validate(w=torch.randn(6, 3, 3, 3))


def test_validator_rejects_mismatched_c_in():
    with pytest.raises(ValueError, match="C_in=5 doesn't match"):
        _validate(x=torch.randn(5, 8, 4, 4))


def test_validator_rejects_even_kernel():
    with pytest.raises(ValueError, match="K_x=2 must be odd"):
        _validate(w=torch.randn(6, 3, 2, 3, 3))


def test_validator_rejects_oversized_kernel():
    with pytest.raises(ValueError, match=r"K_x=9 must be in \[1, 7\]"):
        _validate(w=torch.randn(6, 3, 9, 3, 3))


def test_validator_rejects_bad_stride():
    with pytest.raises(ValueError, match="stride_x=3"):
        _validate(stride=(3, 1, 1))


def test_validator_rejects_negative_padding():
    with pytest.raises(ValueError, match="padding_x=-1"):
        _validate(padding=(-1, 0, 0))


def test_validator_rejects_oversized_padding():
    # K=3 -> max padding = 1
    with pytest.raises(ValueError, match="padding_x=2 exceeds"):
        _validate(padding=(2, 0, 0))


def test_validator_rejects_bad_padding_mode():
    with pytest.raises(ValueError, match="padding_mode='circular'"):
        _validate(padding_mode="circular")


def test_validator_rejects_bad_bias_shape():
    with pytest.raises(ValueError, match=r"bias must have shape \(6,\)"):
        _validate(bias=torch.zeros(5))


def test_validator_rejects_bad_bias_ndim():
    with pytest.raises(ValueError, match="bias must have shape"):
        _validate(bias=torch.zeros(6, 1))


def test_validator_rejects_non_divisible_x_local():
    # Must supply a valid padding (same-padding P_x=1 for K_x=3) so the
    # stride-divisibility check is what actually fires.
    with pytest.raises(ValueError, match="X_local=9 must be divisible"):
        _validate(x=torch.randn(3, 9, 4, 4), stride=(2, 1, 1), padding=(1, 1, 1))


def test_validator_rejects_non_same_padding_x():
    # K_x=3 → required P_x = 1. P_x=0 must be rejected (would produce
    # non-uniform output sharding).
    with pytest.raises(ValueError, match=r"padding_x=0 must equal \(K_x - 1\) // 2 = 1"):
        _validate(padding=(0, 1, 1))


# ---------------------------------------------------------------------------
# Distributed validator cases
# ---------------------------------------------------------------------------


@distributed(world_size=4)
def test_validator_rejects_multi_dim_mesh(rank, world_size):
    mesh_2d = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "sp"))
    with pytest.raises(ValueError, match="1D DeviceMesh"):
        _validate_conv_args(
            _good_x(),
            _good_w(),
            None,
            (1, 1, 1),
            (1, 1, 1),
            "zeros",
            mesh=mesh_2d,
        )


@distributed(world_size=2)
def test_validator_rejects_non_uniform_x_local(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    x_local = 8 if rank == 0 else 4
    x = torch.randn(3, x_local, 4, 4)
    with pytest.raises(ValueError, match="X_local must be uniform"):
        _validate_conv_args(
            x,
            _good_w(),
            None,
            (1, 1, 1),
            (1, 1, 1),
            "zeros",
            mesh=mesh,
        )


@distributed(world_size=2)
def test_validator_accepts_uniform_x_local_with_mesh(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    _validate_conv_args(
        _good_x(),
        _good_w(),
        None,
        (1, 1, 1),
        (1, 1, 1),
        "zeros",
        mesh=mesh,
    )


# ---------------------------------------------------------------------------
# Reference impl — should match nn.functional.conv3d bit-for-bit
# ---------------------------------------------------------------------------


def _conv3d_nchw_reference(x, w, b, stride, padding, padding_mode):
    """Expected result computed the plain way for comparison."""
    x_nchw = x.unsqueeze(0).contiguous()
    if padding_mode == "zeros":
        out = F.conv3d(x_nchw, w, b, stride=stride, padding=padding)
    else:
        pad_amounts = [
            padding[2],
            padding[2],
            padding[1],
            padding[1],
            padding[0],
            padding[0],
        ]
        x_padded = F.pad(x_nchw, pad_amounts, mode=padding_mode)
        out = F.conv3d(x_padded, w, b, stride=stride, padding=0)
    return out.squeeze(0)


@pytest.mark.parametrize("stride", [1, 2])
def test_reference_matches_nn_conv3d(stride):
    # Required P_x for K_x=3 is 1.
    torch.manual_seed(42)
    x = torch.randn(3, 8, 4, 4)
    w = torch.randn(6, 3, 3, 3, 3)
    b = torch.randn(6)

    got = _reference_conv3d(x, w, b, stride=stride, padding=1)
    expected = _conv3d_nchw_reference(x, w, b, (stride,) * 3, (1,) * 3, "zeros")
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("padding_mode", ["reflect", "replicate"])
def test_reference_non_zero_padding_modes(padding_mode):
    torch.manual_seed(42)
    x = torch.randn(3, 8, 4, 4)
    w = torch.randn(6, 3, 3, 3, 3)

    got = _reference_conv3d(x, w, stride=1, padding=1, padding_mode=padding_mode)
    expected = _conv3d_nchw_reference(x, w, None, (1, 1, 1), (1, 1, 1), padding_mode)
    torch.testing.assert_close(got, expected)


def test_reference_output_shape_same_padding():
    x = torch.randn(3, 8, 6, 4)
    w = torch.randn(6, 3, 3, 3, 3)
    out = _reference_conv3d(x, w, stride=1, padding=1)
    assert out.shape == (6, 8, 6, 4)


def test_reference_output_shape_stride_2():
    x = torch.randn(3, 8, 6, 4)
    w = torch.randn(6, 3, 3, 3, 3)
    out = _reference_conv3d(x, w, stride=2, padding=1)
    # floor((8+2-3)/2)+1 = 4, floor((6+2-3)/2)+1 = 3, floor((4+2-3)/2)+1 = 2
    assert out.shape == (6, 4, 3, 2)


def test_reference_output_shape_asymmetric_kernel():
    x = torch.randn(3, 8, 6, 4)
    w = torch.randn(6, 3, 3, 5, 1)  # K_y=5, K_z=1
    out = _reference_conv3d(x, w, stride=1, padding=(1, 2, 0))
    assert out.shape == (6, 8, 6, 4)


def test_reference_with_bias_adds_bias():
    torch.manual_seed(0)
    x = torch.zeros(3, 4, 4, 4)
    w = torch.zeros(6, 3, 3, 3, 3)
    b = torch.arange(6, dtype=torch.float32)
    out = _reference_conv3d(x, w, b, stride=1, padding=1)
    # Conv of zeros + bias -> every voxel == b
    for c in range(6):
        assert torch.all(out[c] == float(c))


def test_validator_rejects_reflect_with_too_small_axis():
    # For K_x=3 the halo is 1 on each side; reflect needs X > halo.
    x = torch.randn(3, 1, 4, 4)
    w = torch.randn(6, 3, 3, 3, 3)
    with pytest.raises(ValueError, match="reflect padding on X"):
        _validate_conv_args(x, w, None, (1, 1, 1), (1, 1, 1), "reflect", mesh=None)


# ---------------------------------------------------------------------------
# _halo_exchange — single-rank (mesh=None): halos are all global edges
# ---------------------------------------------------------------------------


def _expected_edge_padded(x, halo_left, halo_right, padding_mode):
    """Single-rank expected output via F.pad on an NCDHW view."""
    x_nchw = x.unsqueeze(0).contiguous()
    # F.pad on NCDHW: (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi). X is the D dim.
    pads = [0, 0, 0, 0, halo_left, halo_right]
    if padding_mode == "zeros":
        padded = F.pad(x_nchw, pads, mode="constant", value=0)
    else:
        padded = F.pad(x_nchw, pads, mode=padding_mode)
    return padded.squeeze(0)


def test_halo_exchange_noop_when_both_halos_zero():
    x = torch.randn(3, 4, 2, 2)
    out = _halo_exchange(x, 0, 0, "zeros", mesh=None)
    assert out.shape == x.shape
    torch.testing.assert_close(out, x)


@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate"])
@pytest.mark.parametrize("halos", [(1, 1), (2, 2), (1, 2), (0, 3), (3, 0)])
def test_halo_exchange_single_rank_matches_fpad(halos, padding_mode):
    halo_left, halo_right = halos
    torch.manual_seed(0)
    x = torch.randn(3, 6, 2, 2)
    got = _halo_exchange(x, halo_left, halo_right, padding_mode, mesh=None)
    expected = _expected_edge_padded(x, halo_left, halo_right, padding_mode)
    torch.testing.assert_close(got, expected)


# ---------------------------------------------------------------------------
# _halo_exchange — multi-rank: interior from neighbors, edges from padding_mode
# ---------------------------------------------------------------------------


@distributed(world_size=4)
def test_halo_exchange_zeros_multi_rank(rank, world_size):
    """Rank-valued tensors; confirm halos carry the neighbor's value, edges are zero."""
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_local = 4
    x = torch.full((3, X_local, 2, 2), float(rank))

    out = _halo_exchange(x, 1, 1, "zeros", mesh=mesh)

    # Interior is this rank's data
    assert torch.all(out[:, 1 : 1 + X_local] == float(rank))

    # Left halo
    expected_left = float(rank - 1) if rank > 0 else 0.0
    assert torch.all(out[:, :1] == expected_left), f"rank {rank} left halo: {out[:, :1]}"

    # Right halo
    expected_right = float(rank + 1) if rank < world_size - 1 else 0.0
    assert torch.all(out[:, -1:] == expected_right), f"rank {rank} right halo: {out[:, -1:]}"


@distributed(world_size=3)
def test_halo_exchange_asymmetric_halos(rank, world_size):
    """halo_left=2, halo_right=1 — verifies we request/send the right slice widths."""
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_local = 4
    # Each rank's data: rank*10 + index along X, broadcast across C, Y, Z
    x = (
        (torch.arange(X_local, dtype=torch.float32) + rank * 10)
        .view(1, X_local, 1, 1)
        .expand(3, X_local, 2, 2)
        .contiguous()
    )

    halo_left, halo_right = 2, 1
    out = _halo_exchange(x, halo_left, halo_right, "zeros", mesh=mesh)

    # Interior
    torch.testing.assert_close(out[:, halo_left : halo_left + X_local], x)

    # Left halo from rank-1 (its last 2 voxels), or zeros on rank 0
    if rank == 0:
        assert torch.all(out[:, :halo_left] == 0.0)
    else:
        # rank-1's last 2 X values: (rank-1)*10 + [X_local-2, X_local-1]
        prev = rank - 1
        expected_left_vals = torch.tensor(
            [prev * 10 + (X_local - 2), prev * 10 + (X_local - 1)], dtype=torch.float32
        )
        got_left_vals = out[0, :halo_left, 0, 0]
        torch.testing.assert_close(got_left_vals, expected_left_vals)

    # Right halo from rank+1 (its first 1 voxel), or zeros on last rank
    if rank == world_size - 1:
        assert torch.all(out[:, -halo_right:] == 0.0)
    else:
        nxt = rank + 1
        expected_right_val = torch.tensor([nxt * 10 + 0], dtype=torch.float32)
        got_right_val = out[0, -halo_right:, 0, 0]
        torch.testing.assert_close(got_right_val, expected_right_val)


@distributed(world_size=2)
def test_halo_exchange_reflect_edge_ranks(rank, world_size):
    """Reflect mode on boundary ranks; interior halos still come from neighbors."""
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_local = 4
    x = (
        torch.arange(X_local * rank, X_local * (rank + 1), dtype=torch.float32)
        .view(1, X_local, 1, 1)
        .expand(1, X_local, 1, 1)
        .contiguous()
    )

    out = _halo_exchange(x, 1, 1, "reflect", mesh=mesh)

    # Interior
    torch.testing.assert_close(out[:, 1 : 1 + X_local], x)

    if rank == 0:
        # Left halo: reflect of own data -> x[1] = 1
        assert out[0, 0, 0, 0].item() == 1.0
        # Right halo: from rank 1's x[0] = 4
        assert out[0, -1, 0, 0].item() == 4.0
    else:
        # Left halo: from rank 0's x[-1] = 3
        assert out[0, 0, 0, 0].item() == 3.0
        # Right halo: reflect of own data -> x[-2] = 6
        assert out[0, -1, 0, 0].item() == 6.0


@distributed(world_size=2)
def test_halo_exchange_replicate_edge_ranks(rank, world_size):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_local = 4
    x = (
        torch.arange(X_local * rank, X_local * (rank + 1), dtype=torch.float32)
        .view(1, X_local, 1, 1)
        .expand(1, X_local, 1, 1)
        .contiguous()
    )

    out = _halo_exchange(x, 1, 1, "replicate", mesh=mesh)

    if rank == 0:
        assert out[0, 0, 0, 0].item() == 0.0  # x[0] replicated
        assert out[0, -1, 0, 0].item() == 4.0  # from rank 1's x[0]
    else:
        assert out[0, 0, 0, 0].item() == 3.0  # from rank 0's x[-1]
        assert out[0, -1, 0, 0].item() == 7.0  # x[-1] replicated


@distributed(world_size=1)
def test_halo_exchange_world_size_1_treats_all_as_edge(rank, world_size):
    """world_size=1: no neighbors, both halos are global edges."""
    mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("x",))
    x = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1)
    out = _halo_exchange(x, 1, 1, "zeros", mesh=mesh)
    assert out.shape == (1, 6, 1, 1)
    assert out[0, 0].item() == 0.0
    assert out[0, -1].item() == 0.0
    torch.testing.assert_close(out[:, 1:5], x)


# ---------------------------------------------------------------------------
# conv3d (forward-only distributed conv)
# ---------------------------------------------------------------------------


def test_conv3d_mesh_none_matches_reference():
    """Single-process mesh=None should be identical to _reference_conv3d."""
    torch.manual_seed(0)
    x = torch.randn(3, 8, 4, 4)
    w = torch.randn(6, 3, 3, 3, 3)
    b = torch.randn(6)

    got = conv3d(x, w, b, stride=1, padding=1)
    expected = _reference_conv3d(x, w, b, stride=1, padding=1)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate"])
def test_conv3d_mesh_none_parametrized(stride, padding_mode):
    torch.manual_seed(0)
    x = torch.randn(3, 8, 4, 4)
    w = torch.randn(6, 3, 3, 3, 3)
    b = torch.randn(6)

    got = conv3d(x, w, b, stride=stride, padding=1, padding_mode=padding_mode)
    expected = _reference_conv3d(x, w, b, stride=stride, padding=1, padding_mode=padding_mode)
    torch.testing.assert_close(got, expected)


def test_conv3d_mesh_none_asymmetric_stride_padding():
    """Per-axis stride/padding tuples round-trip correctly."""
    torch.manual_seed(0)
    x = torch.randn(3, 8, 6, 4)
    w = torch.randn(6, 3, 3, 5, 1)  # K_y=5, K_z=1
    b = torch.randn(6)

    got = conv3d(x, w, b, stride=(2, 1, 1), padding=(1, 2, 0))
    expected = _reference_conv3d(x, w, b, stride=(2, 1, 1), padding=(1, 2, 0))
    torch.testing.assert_close(got, expected)


# --- Distributed: each rank runs on its X-slice; gather and compare to reference. ---


def _conv_distributed_worker(
    rank, world_size, global_x, weight, bias, stride, padding, padding_mode
):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous()
    out = conv3d(
        x_local,
        weight,
        bias,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        mesh=mesh,
    )
    return out.detach()


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate"])
def test_conv3d_distributed_matches_single_process_reference(stride, padding_mode):
    torch.manual_seed(0)
    world_size = 2
    X_total = 8
    global_x = torch.randn(3, X_total, 4, 4)
    weight = torch.randn(6, 3, 3, 3, 3)
    bias = torch.randn(6)

    ref_out = _reference_conv3d(
        global_x,
        weight,
        bias,
        stride=stride,
        padding=1,
        padding_mode=padding_mode,
    )

    results = run_distributed(
        _conv_distributed_worker,
        world_size=world_size,
        global_x=global_x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=1,
        padding_mode=padding_mode,
    )

    X_local_out = (X_total // world_size) // stride
    for r in range(world_size):
        expected_slice = ref_out[:, r * X_local_out : (r + 1) * X_local_out]
        torch.testing.assert_close(
            results[r],
            expected_slice,
            msg=lambda m, r=r: f"rank {r}: {m}",
        )


def test_conv3d_distributed_world_size_4():
    """Larger world size to exercise more interior ranks."""
    torch.manual_seed(0)
    world_size = 4
    X_total = 16
    global_x = torch.randn(3, X_total, 4, 4)
    weight = torch.randn(6, 3, 3, 3, 3)
    bias = torch.randn(6)

    ref_out = _reference_conv3d(global_x, weight, bias, stride=1, padding=1)

    results = run_distributed(
        _conv_distributed_worker,
        world_size=world_size,
        global_x=global_x,
        weight=weight,
        bias=bias,
        stride=1,
        padding=1,
        padding_mode="zeros",
    )

    X_local_out = X_total // world_size
    for r in range(world_size):
        expected_slice = ref_out[:, r * X_local_out : (r + 1) * X_local_out]
        torch.testing.assert_close(results[r], expected_slice)


# ---------------------------------------------------------------------------
# Backward — autograd correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate"])
def test_conv3d_mesh_none_backward_matches_fconv3d(stride, padding_mode):
    """Single-process conv3d backward should match F.conv3d autograd."""
    torch.manual_seed(0)
    x = torch.randn(3, 8, 4, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(6, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)
    b = torch.randn(6, dtype=torch.float64, requires_grad=True)

    # Forward + backward through our op
    out = conv3d(x, w, b, stride=stride, padding=1, padding_mode=padding_mode)
    loss = (out * out).sum()
    loss.backward()
    assert x.grad is not None
    assert w.grad is not None
    assert b.grad is not None
    got_gx, got_gw, got_gb = x.grad.clone(), w.grad.clone(), b.grad.clone()

    # Reference: F.conv3d with the same padding handling.
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    ref_out = _reference_conv3d(
        x_ref, w_ref, b_ref, stride=stride, padding=1, padding_mode=padding_mode
    )
    (ref_out * ref_out).sum().backward()
    assert x_ref.grad is not None
    assert w_ref.grad is not None
    assert b_ref.grad is not None

    torch.testing.assert_close(got_gx, x_ref.grad)
    torch.testing.assert_close(got_gw, w_ref.grad)
    torch.testing.assert_close(got_gb, b_ref.grad)


def test_conv3d_mesh_none_backward_no_bias():
    """Verify backward handles bias=None correctly (grad_bias should not exist)."""
    torch.manual_seed(0)
    x = torch.randn(3, 8, 4, 4, dtype=torch.float64, requires_grad=True)
    w = torch.randn(6, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)

    out = conv3d(x, w, None, stride=1, padding=1)
    out.sum().backward()
    assert x.grad is not None
    assert w.grad is not None


def _conv_distributed_backward_worker(
    rank, world_size, global_x, weight, bias, stride, padding_mode
):
    """Each rank computes a local forward+backward on its slice, returns grads."""
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous().requires_grad_(True)
    w_local = weight.clone().requires_grad_(True)
    b_local = bias.clone().requires_grad_(True) if bias is not None else None

    out = conv3d(
        x_local,
        w_local,
        b_local,
        stride=stride,
        padding=1,
        padding_mode=padding_mode,
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
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate"])
def test_conv3d_distributed_backward_matches_reference(stride, padding_mode):
    """Distributed backward: gather per-rank grads, compare to single-process reference."""
    torch.manual_seed(0)
    world_size = 2
    X_total = 8
    global_x = torch.randn(3, X_total, 4, 4, dtype=torch.float64)
    weight = torch.randn(6, 3, 3, 3, 3, dtype=torch.float64)
    bias = torch.randn(6, dtype=torch.float64)

    # Single-process reference: same loss on the full tensor.
    x_ref = global_x.clone().requires_grad_(True)
    w_ref = weight.clone().requires_grad_(True)
    b_ref = bias.clone().requires_grad_(True)
    ref_out = _reference_conv3d(
        x_ref, w_ref, b_ref, stride=stride, padding=1, padding_mode=padding_mode
    )
    (ref_out * ref_out).sum().backward()
    assert x_ref.grad is not None
    assert w_ref.grad is not None
    assert b_ref.grad is not None

    results = run_distributed(
        _conv_distributed_backward_worker,
        world_size=world_size,
        global_x=global_x,
        weight=weight,
        bias=bias,
        stride=stride,
        padding_mode=padding_mode,
    )

    # grad_x: concatenate per-rank slices, compare to the full reference grad.
    X_local = X_total // world_size
    for r in range(world_size):
        expected = x_ref.grad[:, r * X_local : (r + 1) * X_local]
        torch.testing.assert_close(
            results[r]["grad_x"],
            expected,
            msg=lambda m, r=r: f"grad_x rank {r}: {m}",
        )

    # grad_w / grad_b: each rank's gradient is a partial sum — the full ref
    # gradient equals sum across ranks.
    sum_grad_w = sum(r["grad_w"] for r in results)
    torch.testing.assert_close(sum_grad_w, w_ref.grad)

    sum_grad_b = sum(r["grad_b"] for r in results)
    torch.testing.assert_close(sum_grad_b, b_ref.grad)


@distributed(world_size=2)
def test_conv3d_opcheck(rank, world_size):
    """opcheck the registered custom op. Only test_autograd_registration is
    reliable for distributed ops; see the encoder tests for details."""
    torch.manual_seed(rank)
    x = torch.randn(3, 4, 4, 4, requires_grad=True)
    w = torch.randn(6, 3, 3, 3, 3, requires_grad=True)
    b = torch.randn(6, requires_grad=True)

    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    group_name = mesh.get_group().group_name

    torch.library.opcheck(
        torch.ops.bigconv.conv3d.default,
        (x, w, b, [1, 1, 1], [1, 1, 1], "zeros", group_name),
        test_utils=("test_autograd_registration",),
    )


def test_conv3d_distributed_asymmetric_kernel():
    """Kernel (3, 5, 1) with per-axis padding."""
    torch.manual_seed(0)
    world_size = 2
    X_total = 8
    global_x = torch.randn(3, X_total, 6, 4)
    weight = torch.randn(6, 3, 3, 5, 1)
    bias = torch.randn(6)

    ref_out = _reference_conv3d(
        global_x, weight, bias, stride=1, padding=(1, 2, 0), padding_mode="reflect"
    )

    results = run_distributed(
        _conv_distributed_worker,
        world_size=world_size,
        global_x=global_x,
        weight=weight,
        bias=bias,
        stride=1,
        padding=(1, 2, 0),
        padding_mode="reflect",
    )

    X_local_out = X_total // world_size
    for r in range(world_size):
        expected_slice = ref_out[:, r * X_local_out : (r + 1) * X_local_out]
        torch.testing.assert_close(results[r], expected_slice)
