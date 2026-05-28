from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import re
from typing import cast
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
SOURCE_IMAGE = ROOT / "eb8e8bad-a4b2-41a5-99b0-2dd4a42cbeac.png"
OUTPUT_DIR = ROOT / "outputs"
PALETTE_URL = "https://pd.anqstar.com/colors"
GRID_SIZE = 100
SAMPLE_FACTOR = 8
WHITE_THRESHOLD = 245
EDGE_SHARPEN_STRENGTH = 1.0
DOMINANT_COLOR_SIGMA = 30.0

# Define ROIs for Chtholly
HAT_ROI = {
    "x_min": 0.10,
    "x_max": 0.90,
    "y_min": 0.05,
    "y_max": 0.45,
}
RIBBON_BAND_ROI = {
    "x_min": 0.30,
    "x_max": 0.82,
    "y_min": 0.10,
    "y_max": 0.28,
}
RIBBON_BOW_ROI = {
    "x_min": 0.63,
    "x_max": 0.96,
    "y_min": 0.12,
    "y_max": 0.37,
}
CLOAK_ROI = {
    "x_min": 0.05,
    "x_max": 0.95,
    "y_min": 0.40,
    "y_max": 1.00,
}
CLOAK_HEM_ROI = {
    "x_min": 0.14,
    "x_max": 0.88,
    "y_min": 0.72,
    "y_max": 1.00,
}
CLOAK_TORSO_ROI = {
    "x_min": 0.14,
    "x_max": 0.90,
    "y_min": 0.28,
    "y_max": 0.68,
}
HAT_BAND_ROI = {
    "x_min": 0.54,
    "x_max": 0.78,
    "y_min": 0.08,
    "y_max": 0.22,
}
SKIRT_ROI = {
    "x_min": 0.24,
    "x_max": 0.76,
    "y_min": 0.58,
    "y_max": 0.93,
}
UPPER_CLOAK_ROI = {
    "x_min": 0.16,
    "x_max": 0.86,
    "y_min": 0.34,
    "y_max": 0.63,
}
LOWER_CLOAK_ROI = {
    "x_min": 0.08,
    "x_max": 0.92,
    "y_min": 0.63,
    "y_max": 0.95,
}
HAIR_ROI = {
    "x_min": 0.14,
    "x_max": 0.88,
    "y_min": 0.16,
    "y_max": 0.82,
}
SOCKS_ROI = {
    "x_min": 0.34,
    "x_max": 0.68,
    "y_min": 0.86,
    "y_max": 0.99,
}

def output_stem() -> str:
    return f"chatgpt_mard_{GRID_SIZE}x{GRID_SIZE}"

def fetch_palette() -> list[dict[str, object]]:
    req = Request(PALETTE_URL, headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    pattern = re.compile(
        r'title="([A-HM]\d{1,2}) · (#[0-9A-Fa-f]{6}) · RGB\((\d+),\s*(\d+),\s*(\d+)\)"'
    )
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for code, hex_value, red, green, blue in pattern.findall(html):
        if code in seen:
            continue
        seen.add(code)
        entries.append(
            {
                "code": code,
                "hex": hex_value.upper(),
                "rgb": np.array([int(red), int(green), int(blue)], dtype=np.uint8),
            }
        )
    return sorted(entries, key=lambda i: (str(i["code"])[0], int(str(i["code"])[1:])))

def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64) / 255.0
    mask = rgb <= 0.04045
    rgb_linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb_linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    xyz_scaled = xyz / white
    epsilon = 216 / 24389
    kappa = 24389 / 27
    fx = np.where(xyz_scaled[..., 0] > epsilon, np.cbrt(xyz_scaled[..., 0]), (kappa * xyz_scaled[..., 0] + 16) / 116)
    fy = np.where(xyz_scaled[..., 1] > epsilon, np.cbrt(xyz_scaled[..., 1]), (kappa * xyz_scaled[..., 1] + 16) / 116)
    fz = np.where(xyz_scaled[..., 2] > epsilon, np.cbrt(xyz_scaled[..., 2]), (kappa * xyz_scaled[..., 2] + 16) / 116)
    
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([l, a, b], axis=-1)

def find_subject_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.any(rgb < WHITE_THRESHOLD, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, rgb.shape[1]-1, rgb.shape[0]-1
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    padding = max(8, int(round(max(x1-x0, y1-y0) * 0.02)))
    return max(0, x0 - padding), max(0, y0 - padding), min(rgb.shape[1] - 1, x1 + padding), min(rgb.shape[0] - 1, y1 + padding)

def median_downsample(image: Image.Image, width: int, height: int, sample_factor: int) -> Image.Image:
    hires = image.resize((width * sample_factor, height * sample_factor), Image.Resampling.LANCZOS)
    arr = np.array(hires, dtype=np.uint8).reshape(height, sample_factor, width, sample_factor, 3)
    return Image.fromarray(np.median(arr, axis=(1, 3)).astype(np.uint8), mode="RGB")


def dominant_cluster_downsample(image: Image.Image, width: int, height: int, sample_factor: int) -> Image.Image:
    hires = image.resize((width * sample_factor, height * sample_factor), Image.Resampling.LANCZOS)
    arr = np.array(hires, dtype=np.float32).reshape(height, sample_factor, width, sample_factor, 3)
    arr = np.transpose(arr, (0, 2, 1, 3, 4))

    axis = np.linspace(-(sample_factor - 1) / 2, (sample_factor - 1) / 2, sample_factor, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    spatial_sigma = max(1.0, sample_factor * 0.38)
    spatial_weights = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * spatial_sigma**2)).reshape(-1)
    color_sigma_sq = float(DOMINANT_COLOR_SIGMA**2)

    downsampled = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            block = arr[row, col].reshape(-1, 3)
            diff = block[:, None, :] - block[None, :, :]
            color_dist_sq = np.sum(diff * diff, axis=2)
            support = np.exp(-color_dist_sq / (2.0 * color_sigma_sq))
            scores = support @ spatial_weights
            dominant_idx = int(np.argmax(scores))
            cluster_weights = spatial_weights * support[dominant_idx]
            if float(cluster_weights.sum()) < 1e-6:
                cluster_weights = spatial_weights
            color = np.average(block, axis=0, weights=cluster_weights)
            downsampled[row, col] = np.clip(np.round(color), 0, 255).astype(np.uint8)

    return Image.fromarray(downsampled, mode="RGB")

def edge_aware_sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    arr = np.array(image, dtype=np.float32)
    blur = np.array(image.filter(ImageFilter.GaussianBlur(radius=0.9)), dtype=np.float32)
    high_pass = arr - blur
    edge = np.array(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    weight = (0.45 + 1.2 * np.clip(edge * 1.6, 0.0, 1.0)) * strength
    boosted = np.clip(arr + high_pass * weight[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(boosted, mode="RGB").filter(ImageFilter.UnsharpMask(radius=0.75, percent=int(70 * strength), threshold=1))

def build_grid(source_path: Path) -> Image.Image:
    source = Image.open(source_path).convert("RGB")
    source_arr = np.array(source)
    x0, y0, x1, y1 = find_subject_bbox(source_arr)
    cropped = source.crop((x0, y0, x1 + 1, y1 + 1))

    enhanced = ImageOps.autocontrast(cropped, cutoff=1)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.08)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.12)
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.0, percent=90, threshold=2))
    enhanced = edge_aware_sharpen(enhanced, strength=0.85 * EDGE_SHARPEN_STRENGTH)

    scale = min(GRID_SIZE / enhanced.width, GRID_SIZE / enhanced.height)
    target_width = max(1, int(round(enhanced.width * scale)))
    target_height = max(1, int(round(enhanced.height * scale)))

    sampled = dominant_cluster_downsample(enhanced, target_width, target_height, SAMPLE_FACTOR)
    sampled = edge_aware_sharpen(sampled, strength=0.5 * EDGE_SHARPEN_STRENGTH)
    canvas = Image.new("RGB", (GRID_SIZE, GRID_SIZE), (255, 255, 255))
    canvas.paste(sampled, ((GRID_SIZE - target_width) // 2, (GRID_SIZE - target_height) // 2))
    return canvas

def roi_mask_from_bounds(bounds: dict[str, float]) -> np.ndarray:
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    x0, x1 = int(round(GRID_SIZE * bounds["x_min"])), int(round(GRID_SIZE * bounds["x_max"]))
    y0, y1 = int(round(GRID_SIZE * bounds["y_min"])), int(round(GRID_SIZE * bounds["y_max"]))
    mask[y0:y1, x0:x1] = True
    return mask


def combined_roi_mask(*bounds: dict[str, float]) -> np.ndarray:
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for bound in bounds:
        mask |= roi_mask_from_bounds(bound)
    return mask

def build_local_stats(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    luminance = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    chroma = np.max(pixels, axis=2) - np.min(pixels, axis=2)
    padded = np.pad(luminance, 1, mode="edge")
    neighbor_min = np.full((GRID_SIZE, GRID_SIZE), 255.0)
    neighbor_max = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            if dx == 1 and dy == 1: continue
            window = padded[dy : dy + GRID_SIZE, dx : dx + GRID_SIZE]
            neighbor_min = np.minimum(neighbor_min, window)
            neighbor_max = np.maximum(neighbor_max, window)
    return luminance, chroma, neighbor_min, neighbor_max


def build_ribbon_tint_masks(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    luminance, chroma, neighbor_min, _ = build_local_stats(pixels)
    ribbon_roi = combined_roi_mask(RIBBON_BAND_ROI, RIBBON_BOW_ROI)

    pixels_i16 = pixels.astype(np.int16)
    pink_bias = pixels_i16[..., 0] - pixels_i16[..., 1]
    violet_bias = pixels_i16[..., 2] - pixels_i16[..., 1]
    tinted_bias = pixels_i16[..., 0] + pixels_i16[..., 2] - 2 * pixels_i16[..., 1]

    ribbon_base = (
        ribbon_roi
        & (luminance >= 176)
        & (chroma >= 3)
        & (chroma <= 92)
        & (neighbor_min <= 165)
    )
    blush_mask = ribbon_base & (
        (pink_bias >= 4)
        | ((pink_bias + violet_bias) >= 7)
        | ((luminance >= 205) & (tinted_bias >= 2))
    )
    lavender_mask = ribbon_base & ~blush_mask & ((violet_bias >= 11) | (tinted_bias >= 24))
    soft_mask = ribbon_base & ~blush_mask & ~lavender_mask
    return blush_mask, lavender_mask, soft_mask


def build_ribbon_full_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, _ = build_local_stats(pixels)
    ribbon_roi = combined_roi_mask(RIBBON_BAND_ROI, RIBBON_BOW_ROI)
    pixels_i16 = pixels.astype(np.int16)
    tinted_bias = pixels_i16[..., 0] + pixels_i16[..., 2] - 2 * pixels_i16[..., 1]
    return (
        ribbon_roi
        & (luminance >= 142)
        & (luminance <= 252)
        & (chroma <= 92)
        & (neighbor_min <= 235)
        & (tinted_bias >= -10)
    )


def build_hat_band_pink_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, _ = build_local_stats(pixels)
    roi = roi_mask_from_bounds(HAT_BAND_ROI)
    return (
        roi
        & (luminance >= 174)
        & (luminance <= 252)
        & (chroma <= 64)
        & (neighbor_min <= 244)
    )

def build_hat_black_relief_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(HAT_ROI)
    # Detect black fabric areas that have slight gradients/details
    return (
        roi_mask
        & (luminance >= 10)
        & (luminance <= 65)
        & (chroma <= 30)
        & ((neighbor_max - neighbor_min) >= 12)  # Some local texture
    )


def build_cloak_relief_masks(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    cloak_roi = roi_mask_from_bounds(CLOAK_ROI)
    local_span = neighbor_max - neighbor_min
    base = (
        cloak_roi
        & (luminance >= 8)
        & (luminance <= 82)
        & (chroma <= 34)
        & (neighbor_min <= 90)
        & (local_span >= 10)
    )
    shadow_mask = base & (luminance <= 26)
    midtone_mask = base & (luminance > 26) & (luminance <= 46)
    lift_mask = base & (luminance > 46) & (luminance <= 82) & (local_span >= 14)
    return shadow_mask, midtone_mask, lift_mask


def build_cloak_hem_cool_shadow_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    hem_roi = roi_mask_from_bounds(CLOAK_HEM_ROI)
    local_span = neighbor_max - neighbor_min
    return (
        hem_roi
        & (luminance >= 10)
        & (luminance <= 30)
        & (chroma <= 30)
        & (neighbor_min <= 44)
        & (local_span >= 12)
    )


def build_cloak_torso_fold_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    torso_roi = roi_mask_from_bounds(CLOAK_TORSO_ROI)
    local_span = neighbor_max - neighbor_min
    return (
        torso_roi
        & (luminance >= 24)
        & (luminance <= 60)
        & (chroma <= 34)
        & (neighbor_min <= 86)
        & (local_span >= 14)
    )


def build_cloak_torso_edge_lift_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    torso_roi = roi_mask_from_bounds(CLOAK_TORSO_ROI)
    local_span = neighbor_max - neighbor_min
    return (
        torso_roi
        & (luminance >= 50)
        & (luminance <= 78)
        & (chroma <= 34)
        & (neighbor_min <= 72)
        & (local_span >= 20)
    )


def build_skirt_blue_trim_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, _, neighbor_max = build_local_stats(pixels)
    skirt_roi = roi_mask_from_bounds(SKIRT_ROI)
    pixels_i16 = pixels.astype(np.int16)
    blue_bias = pixels_i16[..., 2] - pixels_i16[..., 0]
    cyan_bias = pixels_i16[..., 2] - pixels_i16[..., 1]
    return (
        skirt_roi
        & (luminance >= 138)
        & (luminance <= 230)
        & (chroma >= 8)
        & (blue_bias >= 8)
        & (cyan_bias >= -6)
        & (neighbor_max >= 165)
    )


def build_skirt_white_fold_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    skirt_roi = roi_mask_from_bounds(SKIRT_ROI)
    local_span = neighbor_max - neighbor_min
    return (
        skirt_roi
        & (luminance >= 178)
        & (luminance <= 246)
        & (chroma <= 42)
        & (neighbor_min <= 224)
        & (local_span >= 8)
    )


def build_sock_gray_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi = roi_mask_from_bounds(SOCKS_ROI)
    local_span = neighbor_max - neighbor_min
    return (
        roi
        & (luminance >= 172)
        & (luminance <= 252)
        & (chroma <= 72)
        & (neighbor_min <= 250)
        & (local_span >= 1)
    )


def build_upper_cloak_balance_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, _, neighbor_max = build_local_stats(pixels)
    roi = roi_mask_from_bounds(UPPER_CLOAK_ROI)
    return roi & (luminance >= 10) & (luminance <= 76) & (chroma <= 34) & (neighbor_max >= 28)


def build_lower_cloak_balance_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi = roi_mask_from_bounds(LOWER_CLOAK_ROI)
    local_span = neighbor_max - neighbor_min
    return roi & (luminance >= 10) & (luminance <= 80) & (chroma <= 36) & (neighbor_min <= 80) & (local_span >= 8)


def build_hair_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, _, _ = build_local_stats(pixels)
    hair_roi = roi_mask_from_bounds(HAIR_ROI)
    pixels_i16 = pixels.astype(np.int16)
    blue_bias = pixels_i16[..., 2] - pixels_i16[..., 0]
    cyan_bias = pixels_i16[..., 2] - pixels_i16[..., 1]
    return (
        hair_roi
        & (luminance >= 40)
        & (luminance <= 232)
        & (chroma >= 18)
        & (blue_bias >= 18)
        & (cyan_bias >= 4)
    )

def nearest_indices_from_subset(distances: np.ndarray, subset_indices: np.ndarray) -> np.ndarray:
    return subset_indices[np.argmin(distances[:, subset_indices], axis=1)]


def cleanup_quantized_noise(
    code_grid: np.ndarray,
    pixels: np.ndarray,
    distances: np.ndarray,
    palette_rgb: np.ndarray,
    code_lookup: list[str],
    protected_mask: np.ndarray,
) -> np.ndarray:
    refined = code_grid.copy()
    hair_mask = build_hair_mask(pixels)
    luminance = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    pink_noise_codes = {"D12", "E2", "E3", "E8", "E17", "E18", "E19", "E24", "F21"}
    blue_family_indices = np.array(
        [
            idx
            for idx, rgb in enumerate(palette_rgb)
            if int(rgb[2]) >= int(rgb[0]) + 24 and int(rgb[2]) >= int(rgb[1]) - 8 and (int(np.max(rgb)) - int(np.min(rgb))) >= 18
        ],
        dtype=np.int64,
    )
    light_blue_family_indices = np.array(
        [
            idx
            for idx, rgb in enumerate(palette_rgb)
            if int(rgb[2]) >= int(rgb[0]) and int(rgb[2]) >= int(rgb[1]) - 16 and (0.2126 * float(rgb[0]) + 0.7152 * float(rgb[1]) + 0.0722 * float(rgb[2])) >= 170.0
        ],
        dtype=np.int64,
    )

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            flat_index = row * GRID_SIZE + col
            current = int(refined[row, col])
            y0 = max(0, row - 1)
            y1 = min(GRID_SIZE, row + 2)
            x0 = max(0, col - 1)
            x1 = min(GRID_SIZE, col + 2)
            neighbors = refined[y0:y1, x0:x1].reshape(-1)
            center_offset = (row - y0) * (x1 - x0) + (col - x0)
            neighbors = np.delete(neighbors, center_offset)
            unique_codes, counts = np.unique(neighbors, return_counts=True)

            current_support = 0
            if np.any(unique_codes == current):
                current_support = int(counts[unique_codes == current][0])

            if hair_mask[row, col] and code_lookup[current] in pink_noise_codes:
                preferred_blue_indices = light_blue_family_indices if luminance[row, col] >= 170.0 else blue_family_indices
                blue_neighbors = unique_codes[np.isin(unique_codes, preferred_blue_indices)]
                candidate_pool = blue_neighbors if blue_neighbors.size else preferred_blue_indices
                if candidate_pool.size == 0:
                    candidate_pool = blue_family_indices
                if candidate_pool.size:
                    candidate = int(candidate_pool[np.argmin(distances[flat_index, candidate_pool])])
                    threshold = 42.0 if luminance[row, col] >= 170.0 else 18.0
                    if distances[flat_index, candidate] <= distances[flat_index, current] + threshold:
                        refined[row, col] = candidate
                        continue

            if protected_mask[row, col] or current_support > 1 or unique_codes.size == 0:
                continue

            dominant_index = int(np.argmax(counts))
            dominant_code = int(unique_codes[dominant_index])
            if dominant_code == current or int(counts[dominant_index]) < 3:
                continue

            if distances[flat_index, dominant_code] <= distances[flat_index, current] + 10.0:
                refined[row, col] = dominant_code

    return refined

def quantize_to_palette(image: Image.Image, palette: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.array(image, dtype=np.uint8)
    pixel_lab = rgb_to_lab(pixels.reshape(-1, 3))
    palette_rgb = np.stack([cast(np.ndarray, entry["rgb"]) for entry in palette], axis=0)
    palette_lab = rgb_to_lab(palette_rgb)

    distances = np.sum((pixel_lab[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)
    indices = np.argmin(distances, axis=1)
    code_lookup = [str(entry["code"]) for entry in palette]

    # Subsets to add detail to black areas (using dark grays, blues, Browns, instead of flat black)
    subset_lookup = {
        "ribbon_blush": {"E17", "E18", "E8", "E2", "E19", "F21"},
        "ribbon_lavender": {"E24", "D26", "D8", "E17"},
        "ribbon_soft": {"E17", "E18", "E8", "E2"},
        "ribbon_no_white": {"E17", "E8", "E2", "D23", "D26", "E24"},
        "ribbon_aggressive_pink": {"E17", "E18", "E8", "E2", "E19", "F21", "E3"},
        "hat_band_pale_pink": {"E17", "E8", "E18", "E2"},
        "hat_black_relief": {"H5", "H6", "H15", "H17", "C12", "C18", "C29", "D10", "M15"},
        "skirt_blue_trim": {"C28", "D11", "C13", "C14", "D17", "C24", "C27"},
        "skirt_white_fold": {"D16", "H22", "H11", "C28", "C13", "C14", "H9"},
        "sock_light_gray": {"D16", "H22", "H11", "H3"},
        "cloak_shadow": {"H16", "H6", "D10", "C18", "C12"},
        "cloak_hem_cool_shadow": {"H6", "C12", "C18", "D10", "C29"},
        "cloak_midtone": {"H6", "H5", "C12", "C18", "C29", "M3"},
        "cloak_torso_fold": {"H5", "M3", "C12", "C18", "H23", "H3"},
        "cloak_torso_edge_lift": {"H3", "H23", "M3", "D11"},
        "cloak_lift": {"H5", "M3", "H23", "H3", "C12"},
        "upper_cloak_balance": {"H6", "H5", "C12", "C18"},
        "lower_cloak_balance": {"H6", "H5", "C12", "C18"},
    }
    subset_indices = {
        key: np.array([idx for idx, code in enumerate(code_lookup) if code in codes], dtype=np.int64)
        for key, codes in subset_lookup.items()
    }

    current_codes = np.array([code_lookup[i] for i in indices])
    neutral_ribbon_codes = np.isin(
        current_codes,
        ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D1", "D8", "D9", "D11", "D12", "D16", "D23", "D26", "E3", "E24"],
    )
    blush_mask, lavender_mask, soft_mask = build_ribbon_tint_masks(pixels)

    if subset_indices["ribbon_blush"].size:
        blush_flat = blush_mask.reshape(-1) & neutral_ribbon_codes
        indices = np.where(blush_flat, nearest_indices_from_subset(distances, subset_indices["ribbon_blush"]), indices)

    if subset_indices["ribbon_lavender"].size:
        lavender_flat = lavender_mask.reshape(-1) & neutral_ribbon_codes
        indices = np.where(lavender_flat, nearest_indices_from_subset(distances, subset_indices["ribbon_lavender"]), indices)

    if subset_indices["ribbon_soft"].size:
        soft_flat = soft_mask.reshape(-1) & neutral_ribbon_codes
        indices = np.where(soft_flat, nearest_indices_from_subset(distances, subset_indices["ribbon_soft"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    ribbon_no_white_codes = np.isin(current_codes, ["H1", "H2", "H9", "H10", "H11", "H17", "H22"])
    if subset_indices["ribbon_no_white"].size:
        no_white_flat = build_ribbon_full_mask(pixels).reshape(-1) & ribbon_no_white_codes
        indices = np.where(no_white_flat, nearest_indices_from_subset(distances, subset_indices["ribbon_no_white"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    skirt_neutral_codes = np.isin(current_codes, ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D16", "C28"])
    if subset_indices["skirt_blue_trim"].size:
        skirt_blue_flat = build_skirt_blue_trim_mask(pixels).reshape(-1) & skirt_neutral_codes
        indices = np.where(skirt_blue_flat, nearest_indices_from_subset(distances, subset_indices["skirt_blue_trim"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    skirt_fold_codes = np.isin(current_codes, ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D16", "C13", "C14", "C28"])
    if subset_indices["skirt_white_fold"].size:
        skirt_fold_flat = build_skirt_white_fold_mask(pixels).reshape(-1) & skirt_fold_codes
        indices = np.where(skirt_fold_flat, nearest_indices_from_subset(distances, subset_indices["skirt_white_fold"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    sock_white_codes = np.isin(current_codes, ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D16"])
    if subset_indices["sock_light_gray"].size:
        sock_gray_flat = build_sock_gray_mask(pixels).reshape(-1) & sock_white_codes
        indices = np.where(sock_gray_flat, nearest_indices_from_subset(distances, subset_indices["sock_light_gray"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    ribbon_soft_push_codes = np.isin(current_codes, ["D1", "D8", "D9", "D11", "D12", "D23", "D26", "E3", "E24"])
    full_ribbon_mask = (blush_mask | lavender_mask | soft_mask).reshape(-1)
    if subset_indices["ribbon_soft"].size:
        soft_push_flat = full_ribbon_mask & ribbon_soft_push_codes
        indices = np.where(soft_push_flat, nearest_indices_from_subset(distances, subset_indices["ribbon_soft"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    ribbon_aggressive_codes = np.isin(current_codes, ["D1", "D2", "D8", "D9", "D11", "D12", "D23", "D26", "E3", "E24", "H1", "H2", "H9", "H10", "H11", "H17", "H22"])
    if subset_indices["ribbon_aggressive_pink"].size:
        ribbon_aggressive_flat = build_ribbon_full_mask(pixels).reshape(-1) & ribbon_aggressive_codes
        indices = np.where(ribbon_aggressive_flat, nearest_indices_from_subset(distances, subset_indices["ribbon_aggressive_pink"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    hat_band_codes = np.isin(current_codes, ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D1", "D8", "D9", "D11", "D23", "D26", "E3", "E24"])
    if subset_indices["hat_band_pale_pink"].size:
        hat_band_flat = build_hat_band_pink_mask(pixels).reshape(-1) & hat_band_codes
        indices = np.where(hat_band_flat, nearest_indices_from_subset(distances, subset_indices["hat_band_pale_pink"]), indices)

    fabric_mask = build_hat_black_relief_mask(pixels).reshape(-1)
    if subset_indices["hat_black_relief"].size:
        current_codes = np.array([code_lookup[i] for i in indices])
        # Only apply relief if it originally quantized to something very dark like H7 (pure black)
        overdark = np.isin(current_codes, ["H7", "H16", "H15"])
        relief_mask = fabric_mask & overdark
        indices = np.where(relief_mask, nearest_indices_from_subset(distances, subset_indices["hat_black_relief"]), indices)

    cloak_shadow_mask, cloak_midtone_mask, cloak_lift_mask = build_cloak_relief_masks(pixels)
    current_codes = np.array([code_lookup[i] for i in indices])
    dark_cloak_codes = np.isin(current_codes, ["H7", "H16", "H15", "H6", "H5", "C12", "C18", "D10"])

    if subset_indices["cloak_shadow"].size:
        shadow_flat = cloak_shadow_mask.reshape(-1) & dark_cloak_codes
        indices = np.where(shadow_flat, nearest_indices_from_subset(distances, subset_indices["cloak_shadow"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    hem_cool_mask = build_cloak_hem_cool_shadow_mask(pixels).reshape(-1)
    cool_shadow_codes = np.isin(current_codes, ["H7", "H16", "H15", "H6", "C12", "C18", "D10", "C29"])
    if subset_indices["cloak_hem_cool_shadow"].size:
        hem_flat = hem_cool_mask & cool_shadow_codes
        indices = np.where(hem_flat, nearest_indices_from_subset(distances, subset_indices["cloak_hem_cool_shadow"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    dark_cloak_codes = np.isin(current_codes, ["H7", "H16", "H15", "H6", "H5", "C12", "C18", "D10", "C29"])
    if subset_indices["cloak_midtone"].size:
        midtone_flat = cloak_midtone_mask.reshape(-1) & dark_cloak_codes
        indices = np.where(midtone_flat, nearest_indices_from_subset(distances, subset_indices["cloak_midtone"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    torso_fold_codes = np.isin(current_codes, ["H7", "H16", "H15", "H6", "H5", "C12", "C18", "D10", "C29", "M3"])
    if subset_indices["cloak_torso_fold"].size:
        torso_flat = build_cloak_torso_fold_mask(pixels).reshape(-1) & torso_fold_codes
        indices = np.where(torso_flat, nearest_indices_from_subset(distances, subset_indices["cloak_torso_fold"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    torso_edge_lift_codes = np.isin(current_codes, ["H7", "H16", "H15", "H6", "H5", "C12", "C18", "D10", "C29", "M3", "H3", "H23"])
    if subset_indices["cloak_torso_edge_lift"].size:
        torso_edge_flat = build_cloak_torso_edge_lift_mask(pixels).reshape(-1) & torso_edge_lift_codes
        indices = np.where(torso_edge_flat, nearest_indices_from_subset(distances, subset_indices["cloak_torso_edge_lift"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    lift_cloak_codes = np.isin(current_codes, ["H7", "H16", "H15", "H6", "H5", "C12", "C18", "D10", "C29", "M3"])
    if subset_indices["cloak_lift"].size:
        lift_flat = cloak_lift_mask.reshape(-1) & lift_cloak_codes
        indices = np.where(lift_flat, nearest_indices_from_subset(distances, subset_indices["cloak_lift"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    upper_balance_codes = np.isin(current_codes, ["H7", "H16", "M3", "D10"])
    if subset_indices["upper_cloak_balance"].size:
        upper_balance_flat = build_upper_cloak_balance_mask(pixels).reshape(-1) & upper_balance_codes
        indices = np.where(upper_balance_flat, nearest_indices_from_subset(distances, subset_indices["upper_cloak_balance"]), indices)

    current_codes = np.array([code_lookup[i] for i in indices])
    lower_balance_codes = np.isin(current_codes, ["H16", "H7", "B23", "M3"])
    if subset_indices["lower_cloak_balance"].size:
        lower_balance_flat = build_lower_cloak_balance_mask(pixels).reshape(-1) & lower_balance_codes
        indices = np.where(lower_balance_flat, nearest_indices_from_subset(distances, subset_indices["lower_cloak_balance"]), indices)

    code_grid = indices.reshape(GRID_SIZE, GRID_SIZE)
    code_lookup_arr = np.array(code_lookup)
    code_grid_flat = code_grid.reshape(-1)

    if subset_indices["ribbon_soft"].size:
        late_ribbon_codes = np.isin(code_lookup_arr[code_grid_flat], ["D1", "D8", "D9", "D11", "D12", "D23", "D26", "E3", "E24"])
        late_ribbon_mask = (blush_mask | lavender_mask | soft_mask).reshape(-1) & late_ribbon_codes
        code_grid_flat = np.where(late_ribbon_mask, nearest_indices_from_subset(distances, subset_indices["ribbon_soft"]), code_grid_flat)

    if subset_indices["ribbon_no_white"].size:
        late_white_codes = np.isin(code_lookup_arr[code_grid_flat], ["H1", "H2", "H9", "H10", "H11", "H17", "H22"])
        late_no_white_mask = build_ribbon_full_mask(pixels).reshape(-1) & late_white_codes
        code_grid_flat = np.where(late_no_white_mask, nearest_indices_from_subset(distances, subset_indices["ribbon_no_white"]), code_grid_flat)

    if subset_indices["ribbon_aggressive_pink"].size:
        late_aggressive_codes = np.isin(code_lookup_arr[code_grid_flat], ["D1", "D2", "D8", "D9", "D11", "D12", "D23", "D26", "E3", "E24", "H1", "H2", "H9", "H10", "H11", "H17", "H22"])
        late_aggressive_mask = build_ribbon_full_mask(pixels).reshape(-1) & late_aggressive_codes
        code_grid_flat = np.where(late_aggressive_mask, nearest_indices_from_subset(distances, subset_indices["ribbon_aggressive_pink"]), code_grid_flat)
        forced_ribbon_mask = build_ribbon_full_mask(pixels).reshape(-1)
        code_grid_flat = np.where(forced_ribbon_mask, nearest_indices_from_subset(distances, subset_indices["ribbon_aggressive_pink"]), code_grid_flat)

    if subset_indices["hat_band_pale_pink"].size:
        late_hat_band_codes = np.isin(code_lookup_arr[code_grid_flat], ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D1", "D8", "D9", "D11", "D23", "D26", "E3", "E24"])
        late_hat_band_mask = build_hat_band_pink_mask(pixels).reshape(-1) & late_hat_band_codes
        code_grid_flat = np.where(late_hat_band_mask, nearest_indices_from_subset(distances, subset_indices["hat_band_pale_pink"]), code_grid_flat)

    if subset_indices["skirt_blue_trim"].size:
        late_skirt_blue_codes = np.isin(code_lookup_arr[code_grid_flat], ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D16", "C28"])
        late_skirt_blue_mask = build_skirt_blue_trim_mask(pixels).reshape(-1) & late_skirt_blue_codes
        code_grid_flat = np.where(late_skirt_blue_mask, nearest_indices_from_subset(distances, subset_indices["skirt_blue_trim"]), code_grid_flat)

    if subset_indices["skirt_white_fold"].size:
        late_skirt_fold_codes = np.isin(code_lookup_arr[code_grid_flat], ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D16", "C13", "C14", "C28"])
        late_skirt_fold_mask = build_skirt_white_fold_mask(pixels).reshape(-1) & late_skirt_fold_codes
        code_grid_flat = np.where(late_skirt_fold_mask, nearest_indices_from_subset(distances, subset_indices["skirt_white_fold"]), code_grid_flat)

    if subset_indices["sock_light_gray"].size:
        late_sock_codes = np.isin(code_lookup_arr[code_grid_flat], ["H1", "H2", "H9", "H10", "H11", "H17", "H22", "D16"])
        late_sock_mask = build_sock_gray_mask(pixels).reshape(-1) & late_sock_codes
        code_grid_flat = np.where(late_sock_mask, nearest_indices_from_subset(distances, subset_indices["sock_light_gray"]), code_grid_flat)

    if subset_indices["cloak_torso_edge_lift"].size:
        late_torso_codes = np.isin(code_lookup_arr[code_grid_flat], ["H7", "H16", "H6", "H5", "C12", "C18", "B23", "M12"])
        late_torso_mask = build_cloak_torso_edge_lift_mask(pixels).reshape(-1) & late_torso_codes
        code_grid_flat = np.where(late_torso_mask, nearest_indices_from_subset(distances, subset_indices["cloak_torso_edge_lift"]), code_grid_flat)

    if subset_indices["upper_cloak_balance"].size:
        late_upper_codes = np.isin(code_lookup_arr[code_grid_flat], ["H7", "H16", "M3", "D10"])
        late_upper_mask = build_upper_cloak_balance_mask(pixels).reshape(-1) & late_upper_codes
        code_grid_flat = np.where(late_upper_mask, nearest_indices_from_subset(distances, subset_indices["upper_cloak_balance"]), code_grid_flat)

    if subset_indices["lower_cloak_balance"].size:
        late_lower_codes = np.isin(code_lookup_arr[code_grid_flat], ["H16", "H7", "B23", "M3"])
        late_lower_mask = build_lower_cloak_balance_mask(pixels).reshape(-1) & late_lower_codes
        code_grid_flat = np.where(late_lower_mask, nearest_indices_from_subset(distances, subset_indices["lower_cloak_balance"]), code_grid_flat)

    light_blue_family_indices = np.array(
        [
            idx
            for idx, rgb in enumerate(palette_rgb)
            if int(rgb[2]) >= int(rgb[0]) and int(rgb[2]) >= int(rgb[1]) - 16 and (0.2126 * float(rgb[0]) + 0.7152 * float(rgb[1]) + 0.0722 * float(rgb[2])) >= 170.0
        ],
        dtype=np.int64,
    )
    if light_blue_family_indices.size:
        late_hair_codes = np.isin(code_lookup_arr[code_grid_flat], ["D12", "E2", "E3", "E8", "E17", "E18", "E19", "E24", "F21"])
        late_hair_mask = build_hair_mask(pixels).reshape(-1) & late_hair_codes
        code_grid_flat = np.where(late_hair_mask, nearest_indices_from_subset(distances, light_blue_family_indices), code_grid_flat)

    code_grid = code_grid_flat.reshape(GRID_SIZE, GRID_SIZE)
    protected_mask = (
        combined_roi_mask(HAT_ROI, RIBBON_BAND_ROI, RIBBON_BOW_ROI)
        | build_ribbon_full_mask(pixels)
        | build_hat_band_pink_mask(pixels)
        | build_skirt_blue_trim_mask(pixels)
        | build_skirt_white_fold_mask(pixels)
        | build_sock_gray_mask(pixels)
        | build_cloak_torso_fold_mask(pixels)
        | build_cloak_torso_edge_lift_mask(pixels)
        | build_cloak_hem_cool_shadow_mask(pixels)
        | build_upper_cloak_balance_mask(pixels)
        | build_lower_cloak_balance_mask(pixels)
    )
    code_grid = cleanup_quantized_noise(code_grid, pixels, distances, palette_rgb, code_lookup, protected_mask)
    if subset_indices["ribbon_aggressive_pink"].size:
        ribbon_force_grid = nearest_indices_from_subset(distances, subset_indices["ribbon_aggressive_pink"]).reshape(GRID_SIZE, GRID_SIZE)
        code_grid = np.where(build_ribbon_full_mask(pixels), ribbon_force_grid, code_grid)
    if subset_indices["hat_band_pale_pink"].size:
        hat_band_force_grid = nearest_indices_from_subset(distances, subset_indices["hat_band_pale_pink"]).reshape(GRID_SIZE, GRID_SIZE)
        code_grid = np.where(build_hat_band_pink_mask(pixels), hat_band_force_grid, code_grid)
    if subset_indices["sock_light_gray"].size:
        sock_force_grid = nearest_indices_from_subset(distances, subset_indices["sock_light_gray"]).reshape(GRID_SIZE, GRID_SIZE)
        code_grid = np.where(build_sock_gray_mask(pixels), sock_force_grid, code_grid)
    quantized = palette_rgb[code_grid.reshape(-1)].reshape(GRID_SIZE, GRID_SIZE, 3).astype(np.uint8)
    return code_grid, quantized

def write_code_matrix(path: Path, code_grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(e["code"]) for e in palette]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("    " + " ".join(f"{col:>4}" for col in range(1, GRID_SIZE + 1)) + "\n")
        for row in range(GRID_SIZE):
            handle.write(f"{row + 1:>3} " + " ".join(f"{code_lookup[idx]:>4}" for idx in code_grid[row]) + "\n")

def write_code_csv(path: Path, code_grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(e["code"]) for e in palette]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + [str(col) for col in range(1, GRID_SIZE + 1)])
        for row in range(GRID_SIZE):
            writer.writerow([str(row + 1)] + [code_lookup[idx] for idx in code_grid[row]])

def write_usage_csv(path: Path, code_grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(e["code"]) for e in palette]
    counts = Counter(code_lookup[idx] for idx in code_grid.reshape(-1))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "count", "hex", "rgb"])
        for e in palette:
            if (cnt := counts.get(str(e["code"]), 0)) > 0:
                rgb = cast(np.ndarray, e["rgb"])
                writer.writerow([str(e["code"]), cnt, str(e["hex"]), f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"])

def draw_chart(path: Path, quantized: np.ndarray, code_grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    cell = 20
    margin_left, margin_top = 42, 32
    img = Image.new("RGB", (margin_left + GRID_SIZE * cell + 1, margin_top + GRID_SIZE * cell + 1), (255, 255, 255))
    draw, font = ImageDraw.Draw(img), ImageFont.load_default()
    for row in range(GRID_SIZE): draw.text((8, margin_top + row * cell + 5), f"{row + 1:02d}", fill=(40,40,40), font=font)
    for col in range(GRID_SIZE): draw.text((margin_left + col * cell + 4, 8), f"{col + 1:02d}", fill=(40,40,40), font=font)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x0, y0 = margin_left + col * cell, margin_top + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=tuple(int(v) for v in quantized[row, col]), outline=(160, 160, 160))
    img.save(path)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()
    global GRID_SIZE
    GRID_SIZE = args.size
    OUTPUT_DIR.mkdir(exist_ok=True)
    palette = fetch_palette()
    sampled = build_grid(SOURCE_IMAGE)
    code_grid, quantized = quantize_to_palette(sampled, palette)
    stem = output_stem()
    Image.fromarray(quantized, mode="RGB").resize((GRID_SIZE * 16, GRID_SIZE * 16), Image.Resampling.NEAREST).save(OUTPUT_DIR / f"{stem}_quantized.png")
    draw_chart(OUTPUT_DIR / f"{stem}_chart.png", quantized, code_grid, palette)
    write_code_matrix(OUTPUT_DIR / f"{stem}_codes.txt", code_grid, palette)
    write_code_csv(OUTPUT_DIR / f"{stem}_codes.csv", code_grid, palette)
    write_usage_csv(OUTPUT_DIR / f"{stem}_usage.csv", code_grid, palette)
    print(f"Generated Chtholly outputs in {OUTPUT_DIR} with size {GRID_SIZE}")

if __name__ == "__main__":
    main()
