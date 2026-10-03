"""Shared epsilon schedules used by characterization tests and future runners."""

from __future__ import annotations


def linear_offline_epsilon(time_step: int, time_max: int, eps_decay: float, eps_end: float) -> float:
    """Offline main.py formula: max(((time_max - eps_decay * t) / time_max), eps_end)."""
    return max(((time_max - eps_decay * time_step) / time_max), eps_end)


def exponential_epsilon(time_step: int, eps_start: float, eps_decay: float, eps_end: float) -> float:
    """Matched/web formula: max(eps_start * (eps_decay ** t), eps_end)."""
    return max(eps_start * (eps_decay ** time_step), eps_end)
