import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh

from bigconv.modules import UNet3d
from bigconv.testing import run_distributed


def test_unet3d_rejects_too_short_channel_schedule():
    with pytest.raises(ValueError, match="at least two entries"):
        UNet3d(1, 2, channels=(4,))


def test_unet3d_preserves_spatial_shape():
    torch.manual_seed(0)
    model = UNet3d(2, 3, channels=(4, 8, 16), num_groups=4)
    x = torch.randn(2, 8, 8, 8)

    out = model(x)

    assert out.shape == (3, 8, 8, 8)


def test_unet3d_backward_smoke():
    torch.manual_seed(0)
    model = UNet3d(2, 3, channels=(4, 8), num_groups=4)
    x = torch.randn(2, 8, 4, 4, requires_grad=True)

    out = model(x)
    out.square().sum().backward()

    assert x.grad is not None
    for parameter in model.parameters():
        assert parameter.grad is not None


def _unet3d_distributed_worker(rank, world_size, state_dict, global_x):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    model = UNet3d(2, 3, channels=(4, 8), num_groups=4)
    model.load_state_dict(state_dict)

    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous()
    return model(x_local, mesh=mesh).detach()


def test_unet3d_distributed_matches_full_reference():
    torch.manual_seed(0)
    world_size = 2
    global_x = torch.randn(2, 8, 4, 4)
    model = UNet3d(2, 3, channels=(4, 8), num_groups=4)
    reference = model(global_x).detach()

    results = run_distributed(
        _unet3d_distributed_worker,
        world_size=world_size,
        state_dict=model.state_dict(),
        global_x=global_x,
    )

    X_local = global_x.shape[1] // world_size
    for rank, result in enumerate(results):
        expected = reference[:, rank * X_local : (rank + 1) * X_local]
        torch.testing.assert_close(result, expected)
