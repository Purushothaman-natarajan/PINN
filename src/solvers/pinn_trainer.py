"""PINN trainer with PyTorch autograd and weighted PDE+BC loss.

Loss = w_pde * mean(R_pde^2) + w_bc * mean(R_bc^2), weights from config.
Computes up to 3rd-order spatial derivatives with torch.autograd.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.physics.boundary_conditions import bc_residual
from src.physics.governing_equations import compute_derivatives, pde_residuals


class PINNTrainer:
    """Custom training loop for the trihybrid LTNE PINN.

    Args:
        model: PINN module to train.
        config: Full config dictionary.
        coeffs: Flat physics/nanofluid coefficient dictionary.
        device: Torch device override (defaults to config resolution).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        coeffs: Dict[str, Any],
        device: torch.device | None = None,
    ) -> None:
        from src.core.config_loader import resolve_device

        self.model = model
        self.config = config
        self.coeffs = coeffs
        self.equations = config["equations"]
        train_cfg = config["training"]
        domain = config["domain"]

        dev_name = str(train_cfg.get("device", "auto"))
        self.device = device or resolve_device(dev_name)
        self.model.to(self.device)

        self.epochs = int(train_cfg.get("epochs", 20000))
        self.lr = float(train_cfg.get("lr", 1e-3))
        self.pde_weight = float(train_cfg.get("pde_weight", 1.0))
        self.bc_weight = float(train_cfg.get("bc_weight", 10.0))
        self.n_col = int(domain.get("n_collocation", 512))
        self.eta0 = float(domain.get("eta0", 4.0))
        self.log_every = int(train_cfg.get("log_every", 500))
        self.save_every = int(train_cfg.get("save_every", 2000))
        self.ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        dtype_name = str(domain.get("dtype", "float32")).lower()
        self.dtype = torch.float64 if dtype_name == "float64" else torch.float32

        opt_name = str(train_cfg.get("optimizer", "adam")).lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                self.model.parameters(), lr=self.lr, max_iter=20
            )
        else:
            raise ValueError(f"Unsupported optimizer '{opt_name}'")

        sched_cfg = train_cfg.get("scheduler", None)
        self.scheduler = None
        if isinstance(sched_cfg, dict):
            if sched_cfg.get("type", "").lower() == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=int(sched_cfg.get("step_size", 5000)),
                    gamma=float(sched_cfg.get("gamma", 0.5)),
                )

        self.history: Dict[str, List[float]] = {
            "total": [],
            "pde": [],
            "bc": [],
        }

    def _sample_collocation(self) -> torch.Tensor:
        """Sample interior collocation points in (0, eta0)."""
        eta = torch.rand(
            (self.n_col, 1), device=self.device, dtype=self.dtype
        ) * self.eta0
        eta.requires_grad_(True)
        return eta

    def _pde_loss(self, eta: torch.Tensor) -> torch.Tensor:
        """Compute mean-squared PDE residual loss at collocation points."""
        out = compute_derivatives(self.model, eta)
        (f, t_f, t_s, ph, f_p, tf_p, ts_p, ph_p) = out[0:8]
        (f_pp, tf_pp, ts_pp, ph_pp, f_ppp) = out[8:13]
        comps = {"f": f, "theta_f": t_f, "theta_s": t_s, "phi": ph}
        derivs = {
            "fp": f_p,
            "fpp": f_pp,
            "fppp": f_ppp,
            "tfp": tf_p,
            "tfpp": tf_pp,
            "tsp": ts_p,
            "tspp": ts_pp,
            "php": ph_p,
            "phpp": ph_pp,
        }
        res = pde_residuals(eta, comps, derivs, self.coeffs, self.equations)
        stacked = torch.cat([v.reshape(-1) for v in res.values()], dim=0)
        return torch.mean(stacked**2)

    def _bc_loss(self) -> torch.Tensor:
        """Compute mean-squared boundary-condition loss."""
        bc = bc_residual(self.model, self.coeffs, self.device, self.dtype)
        return torch.mean(bc["total"] ** 2)

    def train_step(self) -> Dict[str, float]:
        """Single Adam optimisation step.

        Returns:
            Dict with total, pde, bc losses for this step.
        """
        self.model.train()
        self.optimizer.zero_grad()
        eta = self._sample_collocation()
        # Ensure model dtype consistency for float64 configs.
        loss_pde = self._pde_loss(eta)
        loss_bc = self._bc_loss()
        loss = self.pde_weight * loss_pde + self.bc_weight * loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if isinstance(self.optimizer, torch.optim.LBFGS):

            def _closure() -> torch.Tensor:
                self.optimizer.zero_grad()
                e2 = self._sample_collocation()
                l_p = self._pde_loss(e2)
                l_b = self._bc_loss()
                l_t = self.pde_weight * l_p + self.bc_weight * l_b
                l_t.backward()
                return l_t

            self.optimizer.step(_closure)
        else:
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {
            "total": float(loss.detach().cpu()),
            "pde": float(loss_pde.detach().cpu()),
            "bc": float(loss_bc.detach().cpu()),
        }

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop.

        Returns:
            History dict with total/pde/bc loss curves.
        """
        for epoch in range(1, self.epochs + 1):
            losses = self.train_step()
            for k, v in losses.items():
                self.history[k].append(v)
            if epoch % self.log_every == 0 or epoch == 1:
                print(
                    f"[epoch {epoch:06d}/{self.epochs}] "
                    f"total={losses['total']:.3e} "
                    f"pde={losses['pde']:.3e} bc={losses['bc']:.3e}"
                )
            if epoch % self.save_every == 0:
                self.save(self.ckpt_dir / f"pinn_epoch{epoch}.pt")
        self.save(self.ckpt_dir / "pinn_final.pt")
        return self.history

    @torch.no_grad()
    def predict(
        self, eta_vals: torch.Tensor | Any
    ) -> Dict[str, Any]:
        """Predict physical fields on a 1-D eta grid.

        Args:
            eta_vals: Array-like or tensor of shape (N,) or (N,1).

        Returns:
            Dict with numpy arrays for eta, f, theta_f, theta_s, phi.
        """
        import numpy as np

        self.model.eval()
        arr = np.asarray(eta_vals, dtype=float).reshape(-1, 1)
        eta_t = torch.tensor(arr, device=self.device, dtype=self.dtype)
        out = self.model(eta_t).detach().cpu().numpy()
        return {
            "eta": arr.reshape(-1),
            "f": out[:, 0],
            "theta_f": out[:, 1],
            "theta_s": out[:, 2],
            "phi": out[:, 3],
        }

    def save(self, path: str | Path) -> None:
        """Save model state dict to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path) -> None:
        """Load model state dict from disk."""
        self.model.load_state_dict(
            torch.load(Path(path), map_location=self.device)
        )
