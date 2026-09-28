"""Thermophysical properties for base, nano, and trihybrid nanofluids.

Computes stepwise mixture rules for up to three nanoparticle species and
exposes the A1-A7 style effective coefficients consumed by the PDEs and
the baseline BVP solver.

Conventions (documented so PINN and BVP agree):
    A1 = mu_ratio / rho_ratio   (kinematic-viscosity ratio)
    A2 = sigma_ratio / rho_ratio (magnetic scaling)
    A3 = rho_cp_ratio            (thermal-inertia ratio)
    A4 = k_ratio                 (conductivity ratio)
    A5 = mu_ratio                (viscous diffusion ratio)
    A6 = sigma_ratio             (electrical-conductivity ratio)
    A7 = rho_ratio               (density ratio)

Mixture rules:
    - Brinkman viscosity applied stepwise for each species.
    - Linear mixture for density and heat capacity.
    - Maxwell-Garnett conductivity applied stepwise.
    - Maxwell electrical conductivity applied stepwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class ParticleSpec:
    """Single nanoparticle species specification."""

    name: str
    rho: float
    k: float
    cp: float
    sigma: float
    phi: float


def _maxwell_step(k_f: float, k_p: float, phi: float) -> float:
    """Single Maxwell-Garnett conductivity step."""
    num = k_p + 2.0 * k_f - 2.0 * phi * (k_f - k_p)
    den = k_p + 2.0 * k_f + phi * (k_f - k_p)
    return k_f * (num / den) if den != 0.0 else k_f


def _sigma_step(s_f: float, s_p: float, phi: float) -> float:
    """Single Maxwell electrical-conductivity step."""
    num = s_p + 2.0 * s_f - 2.0 * phi * (s_f - s_p)
    den = s_p + 2.0 * s_f + phi * (s_f - s_p)
    return s_f * (num / den) if den != 0.0 else s_f


class TrihybridProperties:
    """Compute effective trihybrid nanofluid properties."""

    def __init__(self, nano_cfg: Dict[str, Any]) -> None:
        """Initialise from the ``nanofluid`` config block.

        Args:
            nano_cfg: Dictionary with base-fluid scalars and particle list.
        """
        self.rho_f = float(nano_cfg["rho_f"])
        self.mu_f = float(nano_cfg["mu_f"])
        self.k_f = float(nano_cfg["k_f"])
        self.cp_f = float(nano_cfg["cp_f"])
        self.sigma_f = float(nano_cfg.get("sigma_f", 0.05))
        self.particles: List[ParticleSpec] = [
            ParticleSpec(
                name=str(p.get("name", f"p{i}")),
                rho=float(p["rho"]),
                k=float(p["k"]),
                cp=float(p["cp"]),
                sigma=float(p.get("sigma", 1.0e6)),
                phi=float(p.get("phi", 0.0)),
            )
            for i, p in enumerate(nano_cfg.get("particles", []))
        ]

    @property
    def total_phi(self) -> float:
        """Total nanoparticle volume fraction."""
        return float(sum(p.phi for p in self.particles))

    def effective(self) -> Dict[str, float]:
        """Compute effective dimensional properties.

        Returns:
            Dictionary with rho, mu, k, cp, sigma and ratios.
        """
        phi_total = self.total_phi
        # Density and heat capacity: linear mixtures.
        rho = (1.0 - phi_total) * self.rho_f + sum(
            p.phi * p.rho for p in self.particles
        )
        rho_cp = (1.0 - phi_total) * self.rho_f * self.cp_f + sum(
            p.phi * p.rho * p.cp for p in self.particles
        )
        cp = rho_cp / rho if rho > 0 else self.cp_f
        # Brinkman viscosity stepwise.
        mu = self.mu_f
        for p in self.particles:
            mu = mu / ((1.0 - p.phi) ** 2.5)
        # Conductivity / electrical conductivity stepwise.
        k_eff = self.k_f
        sig_eff = self.sigma_f
        for p in self.particles:
            # Conductivity step uses running medium conductivity.
            k_eff = _maxwell_step(k_eff, p.k, p.phi)
            sig_eff = _sigma_step(sig_eff, p.sigma, p.phi)
        return {
            "rho": float(rho),
            "mu": float(mu),
            "k": float(k_eff),
            "cp": float(cp),
            "sigma": float(sig_eff),
            "rho_ratio": float(rho / self.rho_f),
            "mu_ratio": float(mu / self.mu_f),
            "k_ratio": float(k_eff / self.k_f),
            "cp_ratio": float(cp / self.cp_f),
            "sigma_ratio": float(sig_eff / self.sigma_f)
            if self.sigma_f != 0
            else 1.0,
            "rho_cp_ratio": float(rho_cp / (self.rho_f * self.cp_f)),
        }

    def a_coefficients(self) -> Dict[str, float]:
        """Return A1-A7 coefficients for the similarity ODEs."""
        e = self.effective()
        mu_r, rho_r = e["mu_ratio"], e["rho_ratio"]
        return {
            "A1": float(mu_r / rho_r) if rho_r != 0 else 1.0,
            "A2": float(e["sigma_ratio"] / rho_r) if rho_r != 0 else 1.0,
            "A3": float(e["rho_cp_ratio"]),
            "A4": float(e["k_ratio"]),
            "A5": float(mu_r),
            "A6": float(e["sigma_ratio"]),
            "A7": float(rho_r),
        }

    def combined_coefficients(
        self, physics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge nanofluid ratios with physics params into PDE coefficients.

        Args:
            physics: The ``physics`` config block.

        Returns:
            Flat coefficient dict consumed by PDEs and BVP solver. Wall
            parameters that admit ``null`` (``Bi_f``, ``f_outer``) are passed
            through unchanged so ``None`` keeps its "feature off" meaning.
        """
        a = self.a_coefficients()
        e = self.effective()
        coeffs: Dict[str, Any] = dict(a)
        coeffs.update(
            {
                "M": float(physics.get("M", 0.0)),
                "Fr": float(physics.get("Fr", 0.0)),
                "Kp": float(physics.get("Kp", 0.0)),
                "Rd": float(physics.get("Rd", 0.0)),
                "Pr": float(physics.get("Pr", 6.2)),
                "Ec": float(physics.get("Ec", 0.0)),
                "Hs": float(physics.get("Hs", 1.0)),
                "Hsg": float(physics.get("Hsg", 1.0)),
                "Sc": float(physics.get("Sc", 0.6)),
                "Kr": float(physics.get("Kr", 0.0)),
                "Sr": float(physics.get("Sr", 0.0)),
                "Q": float(physics.get("Q", 0.0)),
                "gamma": float(physics.get("gamma_curv", 0.0)),
                "slip": float(physics.get("slip", 0.0)),
                "omega": float(physics.get("omega_ratio", 0.0)),
                "Bi_f": physics.get("Bi_f", None),
                "f_outer": physics.get("f_outer", None),
                "k_ratio": float(e["k_ratio"]),
                "sigma_ratio": float(e["sigma_ratio"]),
                "rho_cp_ratio": float(e["rho_cp_ratio"]),
                "mu_ratio": float(e["mu_ratio"]),
            }
        )
        if coeffs["Bi_f"] is not None:
            coeffs["Bi_f"] = float(coeffs["Bi_f"])
        if coeffs["f_outer"] is not None:
            coeffs["f_outer"] = float(coeffs["f_outer"])
        return coeffs


def compute_coefficients(config: Dict[str, Any]) -> Dict[str, float]:
    """Convenience helper to compute coefficients from full config.

    Args:
        config: Full YAML config dictionary.

    Returns:
        Coefficient dictionary for physics modules.
    """
    props = TrihybridProperties(config["nanofluid"])
    coeffs = props.combined_coefficients(config["physics"])
    coeffs["eta0"] = float(config["domain"].get("eta0", 4.0))
    return coeffs


def to_numpy_dict(coeffs: Dict[str, float]) -> Dict[str, float]:
    """Ensure coefficients are plain Python floats (BVP solver helper)."""
    return {k: float(np.asarray(v).squeeze()) for k, v in coeffs.items()}
