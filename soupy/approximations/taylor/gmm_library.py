"""Helpers for loading 1D Gaussian-mixture libraries from the bundled text file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


_LIBRARY_FILENAME = "gmm_library_1D.txt"
_DEFAULT_RULE = 1


def _is_numeric_line(text: str) -> bool:
    if not text:
        return False
    if text[0].isdigit():
        return True
    return len(text) > 1 and text[0] in "+-" and text[1].isdigit()


@lru_cache(maxsize=1)
def _library_blocks():
    path = Path(__file__).with_name(_LIBRARY_FILENAME)
    text = path.read_text(encoding="utf-8")

    blocks = []
    current = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if _is_numeric_line(line):
            current.append(line)
        elif current:
            blocks.append(tuple(current))
            current = []
    if current:
        blocks.append(tuple(current))

    if len(blocks) < 3:
        raise RuntimeError(
            f"Expected at least 3 numeric rule blocks in {path}, found {len(blocks)}."
        )

    return blocks

# The following line is used to avoid repeated parsing of the library file when called many times with the same rule. 
@lru_cache(maxsize=3)
def _parse_rule(rule: int):
    if rule not in (1, 2, 3):
        raise ValueError(f"Unsupported GMM library rule {rule}; expected 1, 2, or 3.")

    block = _library_blocks()[rule - 1]
    table = {}

    for line in block:
        values = [float(token) for token in line.split()]
        n_components = int(values[0])
        sigma = float(values[1])
        expected = 2 + 2 * n_components
        if len(values) < expected:
            raise RuntimeError(
                f"Malformed GMM row for N={n_components}: expected at least {expected} values, got {len(values)}."
            )

        means = np.asarray(values[2 : 2 + n_components], dtype=float)
        weights = np.asarray(values[2 + n_components : 2 + 2 * n_components], dtype=float)
        table[n_components] = {
            "weights": weights,
            "means": means,
            "sigma": sigma,
        }

    return table


def get_1d_gmm_library_mixture(n_components: int, rule: int = _DEFAULT_RULE, warn: bool = True):
    """Load a 1D Gaussian mixture from the bundled library.

    Falls back to ``mixture_data.get_1d_mixture`` when ``n_components`` is not
    available in the selected rule, while printing a warning when requested.
    """

    if n_components == 1:
        return {
            "weights": np.array([1.0], dtype=float),
            "means": np.array([0.0], dtype=float),
            "sigma": 1.0,
        }

    try:
        table = _parse_rule(rule)
    except ValueError:
        if warn:
            print(
                f"[GMM Library] Warning: unsupported rule={rule}. Falling back to rule {_DEFAULT_RULE}."
            )
        rule = _DEFAULT_RULE
        table = _parse_rule(rule)

    if n_components in table:
        entry = table[n_components]
        return {
            "weights": np.array(entry["weights"], copy=True),
            "means": np.array(entry["means"], copy=True),
            "sigma": float(entry["sigma"]),
        }
    print("n components not found in the library for the selected rule.")

    supported = sorted(table.keys())
    if warn:
        print(
            "[GMM Library] Warning: "
            f"rule {rule} supports N_mix values {supported[0]}..{supported[-1]} (odd only). "
            f"Requested N_mix={n_components}; falling back to mixture_data.get_1d_mixture."
        )

    try:
        from .mixture_data import get_1d_mixture
    except ImportError:
        import importlib.util

        mixture_data_path = Path(__file__).with_name("mixture_data.py")
        spec = importlib.util.spec_from_file_location("mixture_data_fallback", mixture_data_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_1d_mixture = module.get_1d_mixture
    return get_1d_mixture(n_components)


__all__ = ["get_1d_gmm_library_mixture"]
