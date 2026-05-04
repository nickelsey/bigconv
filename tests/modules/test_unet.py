"""Tests for U-Net module building blocks."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from bigconv.modules import (
    ConvNormAct3d,
    DistributedConv3d,
    DistributedConvTranspose3d,
    DistributedGroupNorm,
    UNetConvBlock3d,
    UNetUpBlock3d,
)
from bigconv.ops import conv3d, conv_transpose3d, group_norm
from bigconv.testing import run_distributed


def test_distributed_conv3d_module_matches_functional_op():
    torch.manual_seed(0)
    module = DistributedConv3d(3, 4)
    x = torch.randn(3, 6, 4, 4)

    got = module(x)
    expected = conv3d(
        x,
        module.weight,
        module.bias,
        stride=module.stride,
        padding=module.padding,
        padding_mode=module.padding_mode,
    )

    torch.testing.assert_close(got, expected)


def test_distributed_conv_transpose3d_module_matches_functional_op():
    torch.manual_seed(0)
    module = DistributedConvTranspose3d(3, 4)
    x = torch.randn(3, 3, 2, 2)

    got = module(x)
    expected = conv_transpose3d(
        x,
        module.weight,
        module.bias,
        stride=module.stride,
        padding=module.padding,
        output_padding=module.output_padding,
    )

    torch.testing.assert_close(got, expected)


def test_distributed_group_norm_module_matches_functional_op():
    torch.manual_seed(0)
    module = DistributedGroupNorm(2, 4)
    x = torch.randn(4, 6, 4, 4)

    got = module(x)
    expected = group_norm(x, module.num_groups, module.weight, module.bias, module.eps)

    torch.testing.assert_close(got, expected)


def test_conv_norm_act_matches_functional_ops():
    torch.manual_seed(0)
    module = ConvNormAct3d(
        3,
        4,
        num_groups=2,
        activation=nn.Identity(),
    )
    x = torch.randn(3, 6, 4, 4)

    got = module(x)
    expected = module.conv(x)
    expected = group_norm(
        expected,
        module.norm.num_groups,
        module.norm.weight,
        module.norm.bias,
        module.norm.eps,
    )
    torch.testing.assert_close(got, expected)


def test_unet_conv_block_downsamples_with_first_conv_stride():
    torch.manual_seed(0)
    module = UNetConvBlock3d(3, 4, num_groups=2, stride=2)
    x = torch.randn(3, 8, 6, 4)

    out = module(x)

    assert out.shape == (4, 4, 3, 2)


def test_unet_conv_block_backward_smoke():
    torch.manual_seed(0)
    module = UNetConvBlock3d(3, 4, num_groups=2)
    x = torch.randn(3, 6, 4, 4, requires_grad=True)

    out = module(x)
    out.square().sum().backward()

    assert x.grad is not None
    for parameter in module.parameters():
        assert parameter.grad is not None


def test_unet_up_block_matches_functional_upsample_and_block():
    torch.manual_seed(0)
    module = UNetUpBlock3d(6, skip_channels=4, out_channels=4, num_groups=2)
    x = torch.randn(6, 3, 2, 2)
    skip = torch.randn(4, 6, 4, 4)

    got = module(x, skip)
    up = module.up(x)
    expected = module.block(torch.cat((up, skip), dim=0))

    torch.testing.assert_close(got, expected)


def test_unet_up_block_rejects_mismatched_skip_shape():
    torch.manual_seed(0)
    module = UNetUpBlock3d(6, skip_channels=4, out_channels=4, num_groups=2)
    x = torch.randn(6, 3, 2, 2)
    skip = torch.randn(4, 5, 4, 4)

    with pytest.raises(ValueError, match="does not match skip spatial shape"):
        module(x, skip)


def _conv_block_distributed_worker(rank, world_size, state_dict, global_x):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    module = UNetConvBlock3d(3, 4, num_groups=2)
    module.load_state_dict(state_dict)

    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous()
    return module(x_local, mesh=mesh).detach()


def test_unet_conv_block_distributed_matches_full_reference():
    torch.manual_seed(0)
    world_size = 2
    global_x = torch.randn(3, 8, 4, 4)
    module = UNetConvBlock3d(3, 4, num_groups=2)
    reference = module(global_x).detach()

    results = run_distributed(
        _conv_block_distributed_worker,
        world_size=world_size,
        state_dict=module.state_dict(),
        global_x=global_x,
    )

    X_local = global_x.shape[1] // world_size
    for rank, result in enumerate(results):
        expected = reference[:, rank * X_local : (rank + 1) * X_local]
        torch.testing.assert_close(result, expected)


def _up_block_distributed_worker(rank, world_size, state_dict, global_x, global_skip):
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("x",))
    module = UNetUpBlock3d(6, skip_channels=4, out_channels=4, num_groups=2)
    module.load_state_dict(state_dict)

    X_total = global_x.shape[1]
    X_local = X_total // world_size
    x_local = global_x[:, rank * X_local : (rank + 1) * X_local].contiguous()

    X_skip_local = global_skip.shape[1] // world_size
    skip_local = global_skip[:, rank * X_skip_local : (rank + 1) * X_skip_local].contiguous()
    return module(x_local, skip_local, mesh=mesh).detach()


def test_unet_up_block_distributed_matches_full_reference():
    torch.manual_seed(0)
    world_size = 2
    global_x = torch.randn(6, 4, 2, 2)
    global_skip = torch.randn(4, 8, 4, 4)
    module = UNetUpBlock3d(6, skip_channels=4, out_channels=4, num_groups=2)
    reference = module(global_x, global_skip).detach()

    results = run_distributed(
        _up_block_distributed_worker,
        world_size=world_size,
        state_dict=module.state_dict(),
        global_x=global_x,
        global_skip=global_skip,
    )

    X_local = global_skip.shape[1] // world_size
    for rank, result in enumerate(results):
        expected = reference[:, rank * X_local : (rank + 1) * X_local]
        torch.testing.assert_close(result, expected)
