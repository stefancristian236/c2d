from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import rv_continuous, norm, gennorm, laplace
from scipy.stats import kstest

#define the output directory
#removed if statements at every fit
#unviersal
OUT_DIR = Path("../../out/plots/distributions")


#veriy path
def _ensure_out_dir(dir_path: Path = OUT_DIR) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def fit_and_plot_distribution(
    arr: np.ndarray,
    dist: rv_continuous,
    dist_label: str,
    filename: str,
    param_names: Sequence[str] | None = None,
    bins: str | int = "auto",
    dpi: int = 300,
    show: bool = True,
    ax: Axes | None = None,
) -> tuple[Figure, Axes, tuple, float, float]:
    """
    fit any scipy.stats continuous distribution to `arr`, plot histogram +
    fitted PDF, save to disk.

    returns (fig, ax, fitted_params, ks_statistic, ks_pvalue).
    """
    out_dir = _ensure_out_dir()

    params = dist.fit(arr)
    frozen = dist(*params)

    x = np.linspace(frozen.ppf(0.001), frozen.ppf(0.999), 1000)
    pdf_vals = frozen.pdf(x)

    mean, median, std = frozen.mean(), frozen.median(), frozen.std()

    ks_stat, ks_pval = kstest(arr, dist.name, args=params)

    if param_names is None:
        param_label = ", ".join(f"{p:.4f}" for p in params)
    else:
        param_label = "\n".join(f"{n}: {p:.4f}" for n, p in zip(param_names, params))

    label = (
        f"Fitted {dist_label}\n"
        f"{param_label}\n"
        f"Mean: {mean:.4f}  Median: {median:.4f}  Std: {std:.4f}\n"
        f"KS stat: {ks_stat:.4f}  (p={ks_pval:.3g})"
    )

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    ax.hist(arr, bins=bins, density=True, alpha=0.6, color="blue", label="Error Bins")
    ax.plot(x, pdf_vals, "r--", lw=1, alpha=0.8, label=label)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, pdf_vals.max() * 1.05)
    ax.set_title(f"{dist_label} Distribution")
    ax.set_xlabel("Error Values")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.5)
    ax.legend(fontsize=8)

    if owns_fig:
        fig.savefig(out_dir / filename, dpi=dpi)
        if show:
            plt.show()

    return fig, ax, params, ks_stat, ks_pval


def laplacian_fitting(arr: np.ndarray, **kwargs) -> tuple:
    return fit_and_plot_distribution(
        arr, laplace, "Laplace", "LaplaceDist.png",
        param_names=["loc", "scale"], **kwargs,
    )


def generalised_gaussian_fitting(arr: np.ndarray, **kwargs) -> tuple:
    return fit_and_plot_distribution(
        arr, gennorm, "Generalised Gaussian", "GenGaussianDist.png",
        param_names=["beta", "loc", "scale"], **kwargs,
    )


def normal_fitting(arr: np.ndarray, **kwargs) -> tuple:
    return fit_and_plot_distribution(
        arr, norm, "Normal", "NormalDist.png",
        param_names=["loc", "scale"], **kwargs,
    )


def compare_fits(arr: np.ndarray, filename: str = "ComparisonDist.png", dpi: int = 300) -> dict:
    """
    fit Normal, Laplace and Generalised Gaussian on the same axes 
    """
    out_dir = _ensure_out_dir()
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.hist(arr, bins="auto", density=True, alpha=0.4, color="gray", label="Error Bins")

    results = {}
    for label, dist, names in [
        ("Normal", norm, ["loc", "scale"]),
        ("Laplace", laplace, ["loc", "scale"]),
        ("Gen. Gaussian", gennorm, ["beta", "loc", "scale"]),
    ]:
        _, _, params, ks_stat, ks_pval = fit_and_plot_distribution(
            arr, dist, label, "", param_names=names, ax=ax, show=False
        )
        results[label] = {"params": params, "ks_stat": ks_stat, "ks_pval": ks_pval}

    ax.set_title("Distribution Comparison")
    ax.set_xlabel("Error Values")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.5)
    ax.legend(fontsize=7)
    fig.savefig(out_dir / filename, dpi=dpi)
    plt.show()

    best = min(results, key=lambda k: results[k]["ks_stat"])
    print(f"Best fit by KS statistic: {best}")

    return results