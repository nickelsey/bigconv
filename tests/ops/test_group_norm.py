import pytest
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from bigconv.ops.group_norm import _reference_group_norm, _validate_group_norm_args, group_norm
from bigconv.testing import run_distributed


def _good_x():
    return torch.randn(8, 6, 4, 4)


def _validate(*, x=None, num_groups=4, weight=None, bias=None, eps=1e-5, mesh=None):
    x = _good_x() if x is None else x
    if weight is None:
        weight = torch.ones(x.shape[0], device=x.device)
    if bias is None:
        bias = torch.zeros(x.shape[0], device=x.device)
    _validate_group_norm_args(x, num_groups, weight, bias, eps, mesh)


def test_validator_accepts_good_args():
    _validate()


def test_validator_rejects_bad_x_ndim():
    with pytest.raises(ValueError, match="x must be 4D"):
        _validate(x=torch.randn(8, 6, 4, 4, 1))


def test_validator_rejects_bad_num_groups():
    with pytest.raises(ValueError, match="num_groups must be positive"):
        _validate(num_groups=0)

    with pytest.raises(ValueError, match="C=8 must be divisible"):
        _validate(num_groups=3)


def test_validator_rejects_bad_affine_shapes():
    with pytest.raises(ValueError, match=r"weight must have shape \(8,\)"):
        _validate(weight=torch.ones(7))

    with pytest.raises(ValueError, match=r"bias must have shape \(8,\)"):
        _validate(bias=torch.zeros(7))


def test_validator_rejects_bad_eps():
    with pytest.raises(ValueError, match="eps must be non-negative"):
        _validate(eps=-1e-5)


def test_reference_matches_torch_group_norm():
    torch.manual_seed(0)
    x = torch.randn(8, 6, 4, 4, dtype=torch.float64)
    weight = torch.randn(8, dtype=torch.float64)
    bias = torch.randn(8, dtype=torch.float64)

    got = _reference_group_norm(x, 4, weight, bias, eps=1e-5)
    expected = F.group_norm(x.unsqueeze(0), 4, weight, bias, eps=1e-5).squeeze(0)
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("affine", [False, True])
def test_group_norm_mesh_none_matches_reference(affine):
    torch.manual_seed(0)
    x = torch.randn(8, 6, 4, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(8, dtype=torch.float64, requires_grad=True) if affine else None
    bias = torch.randn(8, dtype=torch.float64, requires_grad=True) if affine else None

    got = group_norm(x, 4, weight, bias, eps=1e-5)
    expected = _reference_group_norm(x, 4, weight, bias, eps=1e-5)
    torch.testing.assert_close(got, expected)

    (got * got).sum().backward()
    assert x.grad is not None
    if weight is not None:
        assert weight.grad is not None
    if bias is not None:
        assert bias.grad is not None


def _group_norm_distributed_worker(rank, world_size, global_x, weight, bias, num_groups):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous()
    out = group_norm(x_local, num_groups, weight, bias, mesh=mesh)
    return out.detach()


def test_group_norm_distributed_forward_matches_reference():
    torch.manual_seed(0)
    world_size = 2
    X_total = 6
    global_x = torch.randn(8, X_total, 4, 4, dtype=torch.float64)
    weight = torch.randn(8, dtype=torch.float64)
    bias = torch.randn(8, dtype=torch.float64)

    ref = _reference_group_norm(global_x, 4, weight, bias)
    results = run_distributed(
        _group_norm_distributed_worker,
        world_size=world_size,
        global_x=global_x,
        weight=weight,
        bias=bias,
        num_groups=4,
    )

    X_local = X_total // world_size
    for r in range(world_size):
        expected = ref[:, r * X_local : (r + 1) * X_local]
        torch.testing.assert_close(results[r], expected)


def _group_norm_distributed_backward_worker(rank, world_size, global_x, weight, bias, num_groups):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous().requires_grad_(True)
    w_local = weight.clone().requires_grad_(True)
    b_local = bias.clone().requires_grad_(True)

    out = group_norm(x_local, num_groups, w_local, b_local, mesh=mesh)
    (out * out).sum().backward()
    assert x_local.grad is not None
    assert w_local.grad is not None
    assert b_local.grad is not None

    return {
        "grad_x": x_local.grad.detach().clone(),
        "grad_w": w_local.grad.detach().clone(),
        "grad_b": b_local.grad.detach().clone(),
    }


def test_group_norm_distributed_backward_matches_reference():
    torch.manual_seed(0)
    world_size = 2
    X_total = 6
    global_x = torch.randn(8, X_total, 4, 4, dtype=torch.float64)
    weight = torch.randn(8, dtype=torch.float64)
    bias = torch.randn(8, dtype=torch.float64)

    x_ref = global_x.clone().requires_grad_(True)
    w_ref = weight.clone().requires_grad_(True)
    b_ref = bias.clone().requires_grad_(True)
    ref = _reference_group_norm(x_ref, 4, w_ref, b_ref)
    (ref * ref).sum().backward()
    assert x_ref.grad is not None
    assert w_ref.grad is not None
    assert b_ref.grad is not None

    results = run_distributed(
        _group_norm_distributed_backward_worker,
        world_size=world_size,
        global_x=global_x,
        weight=weight,
        bias=bias,
        num_groups=4,
    )

    X_local = X_total // world_size
    for r in range(world_size):
        expected = x_ref.grad[:, r * X_local : (r + 1) * X_local]
        torch.testing.assert_close(results[r]["grad_x"], expected)

    torch.testing.assert_close(sum(r["grad_w"] for r in results), w_ref.grad)
    torch.testing.assert_close(sum(r["grad_b"] for r in results), b_ref.grad)


def _group_norm_no_affine_backward_worker(rank, world_size, global_x, num_groups):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous().requires_grad_(True)

    out = group_norm(x_local, num_groups, mesh=mesh)
    (out * out).sum().backward()
    assert x_local.grad is not None
    return x_local.grad.detach().clone()


def test_group_norm_distributed_backward_no_affine():
    torch.manual_seed(0)
    world_size = 2
    X_total = 6
    global_x = torch.randn(8, X_total, 4, 4, dtype=torch.float64)

    x_ref = global_x.clone().requires_grad_(True)
    ref = _reference_group_norm(x_ref, 4)
    (ref * ref).sum().backward()
    assert x_ref.grad is not None

    results = run_distributed(
        _group_norm_no_affine_backward_worker,
        world_size=world_size,
        global_x=global_x,
        num_groups=4,
    )

    X_local = X_total // world_size
    for r in range(world_size):
        expected = x_ref.grad[:, r * X_local : (r + 1) * X_local]
        torch.testing.assert_close(results[r], expected)


def _group_norm_opcheck_worker(rank, world_size):
    torch.manual_seed(rank)
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    group_name = mesh.get_group().group_name
    x = torch.randn(4, 3, 2, 2, requires_grad=True)
    weight = torch.randn(4, requires_grad=True)
    bias = torch.randn(4, requires_grad=True)

    torch.library.opcheck(
        torch.ops.bigconv.group_norm.default,
        (x, weight, bias, 2, 1e-5, group_name),
        test_utils=("test_autograd_registration",),
    )


def test_group_norm_opcheck():
    run_distributed(_group_norm_opcheck_worker, world_size=2)
