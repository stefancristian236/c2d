from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .. import cutout as cut

# define output directories
OUT_DIR = Path("../../out/plots/analysis")


# veriy path
def _ensure_out_dir(dir_path: Path = OUT_DIR) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


# ---------------------------------------------------------------------------
# cube reduction
# ---------------------------------------------------------------------------

def reduce_block_median(
    interpolated_cube: np.ndarray,
    non_interpolated_cube: np.ndarray,
    tile_x: int,
    tile_y: int,
) -> np.ndarray:
    
    """
    perform the reduction fo interpolated cube
    from h x w x tile_x * tile_y -> h / tile_x x w / tile_y x tile_x * tile_y
    """
    
    h, w, bands = interpolated_cube.shape
    small_h, small_w = non_interpolated_cube.shape[:2]

    pad_h = (small_h * tile_x) - h
    pad_w = (small_w * tile_y) - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"interpolated_cube ({h}x{w}) is larger than "
            f"non_interpolated_cube * tile ({small_h * tile_x}x{small_w * tile_y})"
        )

    padded = np.pad(
        interpolated_cube,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=np.nan,
    )

    reshaped = padded.reshape(small_h, tile_x, small_w, tile_y, bands)
    return np.nanmedian(reshaped, axis=(1, 3))


# ---------------------------------------------------------------------------
# per-band statistics
# ---------------------------------------------------------------------------

def per_band_median(
    cube: np.ndarray,
    wl: np.ndarray | None = None,
    output_filepath: str | None = None,
) -> np.ndarray:
    """
    iterate trough the cube 
    perform median on every tile_x * tile_y level
    """
    medians = np.median(cube, axis=(0, 1))

    lines = [
        f"band = {b} / {wl[b]} has median = {medians[b]}\n" if wl is not None
        else f"band = {b} has median = {medians[b]}\n"
        for b in range(cube.shape[2])
    ]
    print("".join(lines), end="")

    if output_filepath:
        with open(output_filepath, "w") as f:
            f.writelines(lines)

    return medians


def band_extremes(cube: np.ndarray, wl: np.ndarray) -> tuple:
    """
    extract:
    the coresponding indices of extremes
    return the indices and corespong wavelenghts of:
    maximum and minimum values
    """
    band_maxs = np.nanmax(cube, axis=(0, 1))
    band_mins = np.nanmin(cube, axis=(0, 1))
    max_idx = np.argmax(band_maxs)
    min_idx = np.argmin(band_mins)
    return wl[max_idx], wl[min_idx], max_idx, min_idx


def count_above_mean_per_band(cube: np.ndarray) -> np.ndarray:
    """
    return the number of values > than mean values
    """
    per_layer_mean = np.mean(cube, axis=(0, 1))
    return np.sum(cube > per_layer_mean, axis=(0, 1))


# ---------------------------------------------------------------------------
# 3x3 neighbourhood patterns
# ---------------------------------------------------------------------------

def _validate_3x3(section: np.ndarray) -> None:
    if section.shape != (3, 3):
        raise ValueError("the section needs to be a 3x3 square!")


def star_pattern(section: np.ndarray) -> float:
    """
    median of the 4-connected neighbours + center of a 3x3 section:

        . 1 .         1
        3 4 5   ->  3 4 5   (median of these 5)
        . 7 .         7
    """
    _validate_3x3(section)
    star = [section[0, 1], section[1, 0], section[1, 2], section[2, 1], section[1, 1]]
    return float(np.median(star))


def square_pattern(section: np.ndarray) -> float:
    """
    median of the 8 pixels surrounding the center of a 3x3 section.
    """
    _validate_3x3(section)
    outer_ring = np.delete(section, 4)
    return float(np.median(outer_ring))


def extreme_pixel_indices(section: np.ndarray, mode: str = "max") -> np.ndarray:
    """
    given a section extract mins or max
    """
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")
    val = np.max(section) if mode == "max" else np.min(section)
    
    return np.argwhere(section == val)

# ---------------------------------------------------------------------------
# interpolated vs. non-interpolated comparison
# ---------------------------------------------------------------------------

def plot_band_medians(
    interp: np.ndarray,
    not_interp: np.ndarray,
    wl_arr: np.ndarray,
    save_as: str | None = "band_medians.png",
    show: bool = True,
):
    """
    overlaid interpoalted and not interpolated median values for comparison
    """
    interp_median = np.median(interp, axis=(0, 1))
    not_interp_median = np.median(not_interp, axis=(0, 1))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wl_arr, interp_median, "o--", color="red", label="interpolated")
    ax.plot(wl_arr, not_interp_median, "o--", color="black", label="not interpolated")
    ax.grid(alpha=0.5)
    ax.legend()

    if save_as:
        fig.savefig(_ensure_out_dir() / save_as, dpi=300)
    if show:
        plt.show()

    return fig, ax


def band_correlations(arr1: np.ndarray, arr2: np.ndarray, wl: np.ndarray) -> dict:
    """
    calculate the correlation matrices per band
    """
    results = {}
    for band in range(arr1.shape[2]):
        corr = np.corrcoef(arr1[:, :, band], arr2[:, :, band])[0, 1]
        print(f"{wl[band]} -> {corr}")
        results[wl[band]] = corr
    return results


def compare_zone(non_inter_cube, inter_cube, center, size, wl_arr) -> tuple:
    """
    extract the same ROI from both interpolated and compare them
    returns the zone data + correlation matrices + plots
    """
    zone_interp = cut.cut_cube(inter_cube, center, size)
    zone_ref = cut.cut_cube(non_inter_cube, center, size)
    band_correlations(zone_ref, zone_interp, wl_arr)
    plot_band_medians(zone_interp, zone_ref, wl_arr)
    return zone_interp, zone_ref


def median_difference_histogram(
    interpolated_cube: np.ndarray,
    not_interpolated_cube: np.ndarray,
    save_as: str | None = "median_diff_hist.png",
    show: bool = True,
):
    """
    performs the median of interpolate and not interpolated data
    return the histogram of the abs difference cube
    """
    inter_median = np.median(interpolated_cube, axis=(0, 1))
    ref_median = np.median(not_interpolated_cube, axis=(0, 1))
    diff_median = np.abs(inter_median - ref_median)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(diff_median, bins="auto", color="blue", alpha=0.6)
    ax.set_xlabel("Absolute median difference")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.5)

    if save_as:
        fig.savefig(_ensure_out_dir() / save_as, dpi=300)
    if show:
        plt.show()

    return fig, ax