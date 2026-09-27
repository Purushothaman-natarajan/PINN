"""Governing physics: residuals and boundary conditions."""

from src.physics.governing_equations import pde_residuals
from src.physics.boundary_conditions import bc_residual

__all__ = ["pde_residuals", "bc_residual"]
