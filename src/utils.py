"""
utils.py — General-purpose helpers for the khipu-ml pipeline.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


def save_figure(fig: plt.Figure, filename: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure to the outputs/ directory."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = OUTPUTS_DIR / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Figure saved: {out}")
    return out


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame as CSV to the outputs/ directory."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = OUTPUTS_DIR / filename
    df.to_csv(out, index=False)
    print(f"CSV saved: {out}")
    return out


def set_plot_style() -> None:
    """Apply a consistent matplotlib style for the project."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "serif",
    })
