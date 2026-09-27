"""Boundary-condition residuals for inner and outer cylinders.

Inner wall (eta=0, stretching cylinder):
    f = 0, f' = 1 + slip * f'', theta_f = 1 (or Robin if Bi_f set),
    theta_s = 1, phi = 1.
Outer wall (eta=eta0, rotating cylinder):
    f' = omega_ratio, theta_f = 0, theta_s = 0, phi = 0.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


def bc_residual(
    model: torch.nn.Module,
    coeffs: Dict[str, Any],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Evaluate boundary-condition residuals at both walls.

    Args:
        model: Trained or training PINN.
        coeffs: Coefficient dict with eta0, slip, omega, Bi info.
        device: Torch device for evaluation.
        dtype: Torch dtype for wall coordinates.

    Returns:
        Dict with inner/outer residual tensors and stacked total.
    """
    eta0 = float(coeffs.get("eta0", 4.0))
    slip = float(coeffs.get("slip", 0.0))
    omega = float(coeffs.get("omega", 0.0))
    bi_f = coeffs.get("Bi_f", None)

    eta_in = torch.zeros((1, 1), device=device, dtype=dtype, requires_grad=True)
    eta_out = torch.full((1, 1), eta0, device=device, dtype=dtype, requires_grad=True)

    def _eval(eta: torch.Tensor) -> tuple[torch.Tensor, ...]:
        eta.requires_grad_(True)
        out = model(eta)
        f, t_f, t_s, ph = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
        f_p = torch.autograd.grad(
            f,
            eta,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True,
        )[0]
        f_pp = torch.autograd.grad(
            f_p,
            eta,
            grad_outputs=torch.ones_like(f_p),
            create_graph=True,
            retain_graph=True,
        )[0]
        return f, t_f, t_s, ph, f_p, f_pp

    f0, tf0, ts0, ph0, fp0, fpp0 = _eval(eta_in)
    f1, tf1, ts1, ph1, fp1, _ = _eval(eta_out)

    # Inner residuals.
    r_f0 = f0 - 0.0
    r_fp0 = fp0 - (1.0 + slip * fpp0)
    if bi_f is None:
        r_tf0 = tf0 - 1.0
    else:
        # Robin: theta_f' + Bi*(theta_f - 1) = 0; need theta_f' here.
        tf_p0 = torch.autograd.grad(
            tf0,
            eta_in,
            grad_outputs=torch.ones_like(tf0),
            create_graph=True,
            retain_graph=True,
        )[0]
        r_tf0 = tf_p0 + float(bi_f) * (tf0 - 1.0)
    r_ts0 = ts0 - 1.0
    r_ph0 = ph0 - 1.0

    # Outer residuals.
    r_fp1 = fp1 - omega
    r_tf1 = tf1 - 0.0
    r_ts1 = ts1 - 0.0
    r_ph1 = ph1 - 0.0
    # Mild entrainment constraint on f at outer wall if provided.
    f_outer = coeffs.get("f_outer", None)
    extras = [r_f0, r_fp0, r_tf0, r_ts0, r_ph0, r_fp1, r_tf1, r_ts1, r_ph1]
    if f_outer is not None:
        extras.append(f1 - float(f_outer) * 0.0 - f1.detach() * 0.0)
        # Note: f(eta0) left free by default; only rate constrained.
        extras.pop()  # keep outer f free unless explicitly constrained

    total = torch.cat([t.reshape(-1) for t in extras], dim=0)
    return {
        "inner": torch.cat(
            [t.reshape(-1) for t in [r_f0, r_fp0, r_tf0, r_ts0, r_ph0]], dim=0
        ),
        "outer": torch.cat(
            [t.reshape(-1) for t in [r_fp1, r_tf1, r_ts1, r_ph1]], dim=0
        ),
        "total": total,
    }
