"""Model topology tests: config-driven architecture loads correctly."""

import torch

from src.models.pinn_architecture import PINN, build_pinn


def test_default_topology():
    """Default 3x128 Tanh network has 4 outputs and correct depth."""
    model = build_pinn(
        {
            "input_dim": 1,
            "hidden_layers": [128, 128, 128],
            "output_dim": 4,
            "activation": "tanh",
            "initialization": "xavier",
        }
    )
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4  # 3 hidden + 1 output
    assert linears[0].in_features == 1
    assert linears[-1].out_features == 4
    assert all(layer.out_features == 128 for layer in linears[:-1])


def test_forward_shapes_and_grad():
    """Forward (N,1)->(N,4) and 3rd-order autograd smoke test."""
    model = PINN(hidden_layers=[32, 32], activation="tanh")
    eta = torch.rand(8, 1, requires_grad=True)
    out = model(eta)
    assert out.shape == (8, 4)
    f = out[:, 0:1]
    fp = torch.autograd.grad(
        f, eta, grad_outputs=torch.ones_like(f), create_graph=True, retain_graph=True
    )[0]
    fpp = torch.autograd.grad(
        fp, eta, grad_outputs=torch.ones_like(fp), create_graph=True, retain_graph=True
    )[0]
    fppp = torch.autograd.grad(
        fpp, eta, grad_outputs=torch.ones_like(fpp), create_graph=False
    )[0]
    assert fp.shape == (8, 1)
    assert fpp.shape == (8, 1)
    assert fppp.shape == (8, 1)
    assert torch.all(torch.isfinite(fppp))


def test_config_layers_respected():
    """Custom layer list produces matching linear stack."""
    model = build_pinn(
        {
            "input_dim": 1,
            "hidden_layers": [64, 32],
            "output_dim": 4,
            "activation": "gelu",
            "initialization": "xavier",
        }
    )
    linears = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert [layer.out_features for layer in linears] == [64, 32, 4]
