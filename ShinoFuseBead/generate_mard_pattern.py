from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
SOURCE_IMAGE = ROOT / "ChatGPT Image 2026年7月11日 13_57_17.png"
OUTPUT_DIR = ROOT / "outputs"
PALETTE_URL = "https://pd.anqstar.com/colors"
GRID_SIZE = 100
SAMPLE_FACTOR = 8
WHITE_THRESHOLD = 245
EDGE_SHARPEN_STRENGTH = 1.0
DOMINANT_COLOR_SIGMA = 30.0

# Region definitions for Shino (normalized 0-1 grid coordinates).
# The sun emblem sits on the character's left shoulder, i.e. the viewer's right.
SUN_EMBLEM_ROI = {
    "x_min": 0.62,
    "x_max": 0.86,
    "y_min": 0.56,
    "y_max": 0.82,
}
HAIR_ROI = {
    "x_min": 0.10,
    "x_max": 0.90,
    "y_min": 0.05,
    "y_max": 0.48,
}
FACE_ROI = {
    "x_min": 0.28,
    "x_max": 0.58,
    "y_min": 0.30,
    "y_max": 0.58,
}
EYES_ROI = {
    "x_min": 0.30,
    "x_max": 0.70,
    "y_min": 0.40,
    "y_max": 0.50,
}
COLLAR_ROI = {
    "x_min": 0.30,
    "x_max": 0.62,
    "y_min": 0.52,
    "y_max": 0.76,
}
TORSO_ROI = {
    "x_min": 0.20,
    "x_max": 0.74,
    "y_min": 0.58,
    "y_max": 0.96,
}
GOLD_TRIM_ROI = {
    "x_min": 0.14,
    "x_max": 0.88,
    "y_min": 0.56,
    "y_max": 0.98,
}
BLUE_GEM_ROI = {
    "x_min": 0.44,
    "x_max": 0.56,
    "y_min": 0.66,
    "y_max": 0.80,
}
SWORDS_ROI = {
    "x_min": 0.70,
    "x_max": 1.00,
    "y_min": 0.05,
    "y_max": 0.80,
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
    if len(entries) != 221:
        raise RuntimeError(f"Expected 221 palette colors, got {len(entries)}")
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
        return 0, 0, rgb.shape[1] - 1, rgb.shape[0] - 1
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    padding = max(8, int(round(max(x1 - x0, y1 - y0) * 0.02)))
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(rgb.shape[1] - 1, x1 + padding),
        min(rgb.shape[0] - 1, y1 + padding),
    )


def median_downsample(image: Image.Image, width: int, height: int, sample_factor: int) -> Image.Image:
    """Median-cut downsampling.

    Upscales with Lanczos, then for each output cell takes the per-channel
    median over the corresponding block.  The median is robust to the
    anti-aliasing halos and stray pixels that plague average-based
    resamplers, so it produces much cleaner fills for large flat regions
    such as skin, hair or armor.
    """
    hires = image.resize((width * sample_factor, height * sample_factor), Image.Resampling.LANCZOS)
    arr = np.array(hires, dtype=np.uint8).reshape(height, sample_factor, width, sample_factor, 3)
    return Image.fromarray(np.median(arr, axis=(1, 3)).astype(np.uint8), mode="RGB")


def edge_aware_sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    arr = np.array(image, dtype=np.float32)
    blur = np.array(image.filter(ImageFilter.GaussianBlur(radius=0.9)), dtype=np.float32)
    high_pass = arr - blur
    edge = np.array(image.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
    weight = (0.45 + 1.2 * np.clip(edge * 1.6, 0.0, 1.0)) * strength
    boosted = np.clip(arr + high_pass * weight[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(boosted, mode="RGB").filter(
        ImageFilter.UnsharpMask(radius=0.75, percent=int(70 * strength), threshold=1)
    )


def boost_sun_emblem(image: Image.Image) -> Image.Image:
    """Pre-boost the golden sun emblem so it survives quantization.

    Increases saturation, contrast and brightness inside the sun emblem ROI
    while leaving the surrounding dark blue fabric untouched.  This makes the
    golden rays and center pop against the navy shoulder.
    """
    w, h = image.size
    x0 = int(round(w * SUN_EMBLEM_ROI["x_min"]))
    x1 = int(round(w * SUN_EMBLEM_ROI["x_max"]))
    y0 = int(round(h * SUN_EMBLEM_ROI["y_min"]))
    y1 = int(round(h * SUN_EMBLEM_ROI["y_max"]))
    region = image.crop((x0, y0, x1, y1))
    region = ImageEnhance.Color(region).enhance(1.60)
    region = ImageEnhance.Contrast(region).enhance(1.35)
    region = ImageEnhance.Brightness(region).enhance(1.12)
    result = image.copy()
    result.paste(region, (x0, y0))
    return result


def detect_black_outline_mask(image: Image.Image) -> np.ndarray:
    """Find the black ink-line pixels in the source.

    Detects very dark, low-chroma pixels and then refines the mask with a
    morphological close so that the line stays connected after downsampling.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    luminance = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    chroma = np.max(arr, axis=2) - np.min(arr, axis=2)
    dark = (luminance <= 55) & (chroma <= 25)

    mask = Image.fromarray((dark.astype(np.uint8) * 255), mode="L")
    # Close small gaps, then erode to keep only the line core.
    mask = mask.filter(ImageFilter.MinFilter(3))
    mask = mask.filter(ImageFilter.MaxFilter(3))
    mask = mask.filter(ImageFilter.MaxFilter(3))
    mask = mask.filter(ImageFilter.MinFilter(3))
    return np.array(mask, dtype=np.uint8) > 127


def downsample_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Aggregate a high-resolution boolean mask to the output grid.

    A cell becomes True if at least ~25% of its source block is covered.
    Pairs well with the median downsampling so outline lines survive the
    resolution drop without bleeding into the surrounding fills.
    """
    img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr >= 0.25


def majority_smooth(code_grid: np.ndarray, pixels: np.ndarray, distances: np.ndarray, protected_mask: np.ndarray, palette_size: int) -> np.ndarray:
    """Iteratively remove isolated noise inside flat regions.

    A cell is rewritten only when it is unprotected, is currently different
    from the dominant color in its 8-neighborhood, and that dominant color
    already covers at least 6 of the 8 cells.  A tiny distance budget keeps
    the pass from forcing jumps across palette families.
    """
    refined = code_grid.copy()
    for _ in range(2):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if protected_mask[row, col]:
                    continue
                current = int(refined[row, col])
                y0 = max(0, row - 1)
                y1 = min(GRID_SIZE, row + 2)
                x0 = max(0, col - 1)
                x1 = min(GRID_SIZE, col + 2)
                patch = refined[y0:y1, x0:x1].reshape(-1)
                unique, counts = np.unique(patch, return_counts=True)
                dominant_idx = int(np.argmax(counts))
                if int(counts[dominant_idx]) < 6:
                    continue
                if unique[dominant_idx] == current:
                    continue
                if distances[row * GRID_SIZE + col, unique[dominant_idx]] > distances[row * GRID_SIZE + col, current] + 4.0:
                    continue
                refined[row, col] = unique[dominant_idx]
    return refined


def build_grid(source_path: Path) -> Image.Image:
    source = Image.open(source_path).convert("RGB")
    source_arr = np.array(source)
    x0, y0, x1, y1 = find_subject_bbox(source_arr)
    cropped = source.crop((x0, y0, x1 + 1, y1 + 1))

    # Capture the source's black outline before any contrast lift, so we
    # can later carve those pixels back into the quantized grid.
    outline_mask = detect_black_outline_mask(cropped)

    # Global color/contrast enhancement.
    enhanced = ImageOps.autocontrast(cropped, cutoff=1)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.10)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.15)
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.0, percent=95, threshold=2))
    enhanced = edge_aware_sharpen(enhanced, strength=0.90 * EDGE_SHARPEN_STRENGTH)

    # Region-specific boost for the iconic sun emblem.
    enhanced = boost_sun_emblem(enhanced)

    scale = min(GRID_SIZE / enhanced.width, GRID_SIZE / enhanced.height)
    target_width = max(1, int(round(enhanced.width * scale)))
    target_height = max(1, int(round(enhanced.height * scale)))

    # Median downsampling eliminates the pepper noise that the dominant-
    # cluster resampler introduced inside large flat regions.
    sampled = median_downsample(enhanced, target_width, target_height, SAMPLE_FACTOR)
    # Rescale the outline mask to the sampled grid (it lives in the same
    # normalized 0-1 coordinate space as the crop).
    sampled_outline = downsample_mask(outline_mask, target_width, target_height)

    # Store the outline mask on the canvas for the quantizer to pick up.
    canvas = Image.new("RGB", (GRID_SIZE, GRID_SIZE), (255, 255, 255))
    offset_x = (GRID_SIZE - target_width) // 2
    offset_y = (GRID_SIZE - target_height) // 2
    canvas.paste(sampled, (offset_x, offset_y))

    full_outline = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    full_outline[offset_y:offset_y + target_height, offset_x:offset_x + target_width] = sampled_outline
    canvas.info["outline_mask"] = full_outline
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
            if dx == 1 and dy == 1:
                continue
            window = padded[dy : dy + GRID_SIZE, dx : dx + GRID_SIZE]
            neighbor_min = np.minimum(neighbor_min, window)
            neighbor_max = np.maximum(neighbor_max, window)
    return luminance, chroma, neighbor_min, neighbor_max


def build_neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    count = np.zeros(mask.shape, dtype=np.uint8)
    for dy, dx in ((0, 1), (1, 0), (1, 2), (2, 1)):
        count += padded[dy : dy + GRID_SIZE, dx : dx + GRID_SIZE]
    return count


def fill_linear_gaps(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant")
    left = padded[1 : 1 + GRID_SIZE, 0:GRID_SIZE]
    right = padded[1 : 1 + GRID_SIZE, 2 : 2 + GRID_SIZE]
    up = padded[0:GRID_SIZE, 1 : 1 + GRID_SIZE]
    down = padded[2 : 2 + GRID_SIZE, 1 : 1 + GRID_SIZE]
    return mask | (left & right) | (up & down)


# ---------------------------------------------------------------------------
# Feature masks
# ---------------------------------------------------------------------------

def build_sun_emblem_mask(pixels: np.ndarray) -> np.ndarray:
    """Detect the golden sun emblem on the left shoulder.

    Uses a permissive gold detector and then dilates/bridges the result so
    that thin rays remain visible after quantization.
    """
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(SUN_EMBLEM_ROI)
    pixels_i16 = pixels.astype(np.int16)
    # Golden: R and G high, B low.  Keep thresholds generous so that even
    # slightly desaturated gold pixels from the downsampled image are caught.
    gold = (
        (pixels_i16[..., 0] >= 100)
        & (pixels_i16[..., 1] >= 75)
        & (pixels_i16[..., 2] <= pixels_i16[..., 1] + 5)
        & (pixels_i16[..., 0] + pixels_i16[..., 1] - 2 * pixels_i16[..., 2] >= 40)
    )
    local_span = neighbor_max - neighbor_min
    base = (
        roi_mask
        & gold
        & (luminance >= 70)
        & (luminance <= 245)
        & (chroma >= 12)
        & (local_span >= 8)
    )
    # Bridge small gaps inside the emblem so the rays stay connected.
    neighbor_count = build_neighbor_count(base)
    bridge = (
        roi_mask
        & (luminance >= 55)
        & (chroma >= 10)
        & gold
        & (neighbor_count >= 1)
    )
    emblem = base | bridge
    emblem = fill_linear_gaps(emblem)
    emblem = fill_linear_gaps(emblem)
    emblem = fill_linear_gaps(emblem)
    # Small dilation: if a gold pixel is adjacent to the emblem, include it.
    emblem = fill_linear_gaps(emblem)
    return emblem & roi_mask


def build_sun_center_mask(pixels: np.ndarray) -> np.ndarray:
    """Brightest circular center of the sun emblem."""
    luminance, chroma, _, _ = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(SUN_EMBLEM_ROI)
    pixels_i16 = pixels.astype(np.int16)
    bright_gold = (
        (pixels_i16[..., 0] >= 170)
        & (pixels_i16[..., 1] >= 140)
        & (pixels_i16[..., 2] <= 110)
        & (pixels_i16[..., 0] + pixels_i16[..., 1] - 2 * pixels_i16[..., 2] >= 90)
    )
    return roi_mask & bright_gold & (luminance >= 160) & (chroma >= 35)


def build_hair_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, _, _ = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(HAIR_ROI)
    pixels_i16 = pixels.astype(np.int16)
    # Light blue/cyan: B dominant, G close to B.
    blue_bias = pixels_i16[..., 2] - pixels_i16[..., 0]
    cyan_bias = pixels_i16[..., 2] - pixels_i16[..., 1]
    return (
        roi_mask
        & (luminance >= 60)
        & (luminance <= 245)
        & (chroma >= 10)
        & (blue_bias >= 18)
        & (cyan_bias >= -8)
    )


def build_eyes_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(EYES_ROI)
    pixels_i16 = pixels.astype(np.int16)
    blue_bias = pixels_i16[..., 2] - pixels_i16[..., 0]
    cyan_bias = pixels_i16[..., 2] - pixels_i16[..., 1]
    local_span = neighbor_max - neighbor_min
    return (
        roi_mask
        & (luminance >= 30)
        & (luminance <= 170)
        & (chroma >= 20)
        & (blue_bias >= 25)
        & (cyan_bias >= -5)
        & (neighbor_min <= 120)
        & (local_span >= 20)
    )


def build_skin_mask(pixels: np.ndarray) -> np.ndarray:
    """Detect face/skin cells.

    The face spans from very light cream highlights (low chroma, very high
    luminance) to saturated peach (high chroma).  All of these are warm
    (R >= G > B) and live inside the face ROI, so we accept a wide range
    and rely on the ROI itself to keep the mask from leaking.
    """
    luminance, chroma, _, _ = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(FACE_ROI)
    pixels_i16 = pixels.astype(np.int16)
    # Anything peach/warm/cream inside the face ROI counts as skin.  Allow
    # very bright cream highlights and reject only deep reds or grays.
    warm = (
        (pixels_i16[..., 0] >= pixels_i16[..., 1] - 5)
        & (pixels_i16[..., 1] > pixels_i16[..., 2] - 5)
        & (pixels_i16[..., 0] >= 70)
        & (pixels_i16[..., 1] >= 50)
        & (pixels_i16[..., 2] <= 150)
    )
    return roi_mask & warm & (luminance >= 80) & (luminance <= 248)


def build_collar_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(COLLAR_ROI)
    local_span = neighbor_max - neighbor_min
    return (
        roi_mask
        & (luminance >= 165)
        & (luminance <= 252)
        & (chroma <= 45)
        & (neighbor_min >= 120)
        & (local_span >= 5)
    )


def build_dark_outfit_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = combined_roi_mask(TORSO_ROI, SUN_EMBLEM_ROI)
    pixels_i16 = pixels.astype(np.int16)
    blue_bias = pixels_i16[..., 2] - pixels_i16[..., 0]
    # Dark blue fabric: low luminance, low chroma, slight blue bias.
    dark_blue = (
        (luminance >= 8)
        & (luminance <= 75)
        & (chroma <= 50)
        & (blue_bias >= -10)
        & (neighbor_min <= 90)
    )
    return roi_mask & dark_blue


def build_gold_trim_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(GOLD_TRIM_ROI)
    pixels_i16 = pixels.astype(np.int16)
    # Stricter gold: skip pixels that are too desaturated (skin-like) and
    # require a real yellow chroma so the face is never matched.
    gold = (
        (pixels_i16[..., 0] >= 150)
        & (pixels_i16[..., 1] >= 120)
        & (pixels_i16[..., 2] <= pixels_i16[..., 1] - 25)
        & (pixels_i16[..., 0] + pixels_i16[..., 1] - 2 * pixels_i16[..., 2] >= 90)
    )
    local_span = neighbor_max - neighbor_min
    return (
        roi_mask
        & gold
        & (luminance >= 110)
        & (luminance <= 250)
        & (chroma >= 45)
        & (local_span >= 20)
    )


def build_blue_gem_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(BLUE_GEM_ROI)
    pixels_i16 = pixels.astype(np.int16)
    blue_bias = pixels_i16[..., 2] - pixels_i16[..., 0]
    cyan_bias = pixels_i16[..., 2] - pixels_i16[..., 1]
    local_span = neighbor_max - neighbor_min
    return (
        roi_mask
        & (luminance >= 60)
        & (luminance <= 210)
        & (chroma >= 20)
        & (blue_bias >= 20)
        & (cyan_bias >= -5)
        & (neighbor_min <= 160)
        & (local_span >= 25)
    )


def build_sword_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, _, _ = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(SWORDS_ROI)
    # Swords are silvery/light gray with low chroma.
    return (
        roi_mask
        & (luminance >= 100)
        & (luminance <= 235)
        & (chroma <= 30)
    )


def build_face_base_mask(pixels: np.ndarray) -> np.ndarray:
    """Select the plain facial skin that should become the single A1 cream.

    This mask covers the whole face ROI except for obvious non-skin
    features (eyes, hair, collar, outlines, sun emblem).  It is used
    together with a post-quantization skin-family check so that every
    flesh-tone cell is recolored to A1 while eyes, blush and outlines
    survive.
    """
    roi_mask = roi_mask_from_bounds(FACE_ROI)
    protected = (
        build_eyes_mask(pixels)
        | build_hair_mask(pixels)
        | build_sun_emblem_mask(pixels)
        | build_gold_trim_mask(pixels)
        | build_blue_gem_mask(pixels)
        | build_dark_outfit_mask(pixels)
    )
    return roi_mask & ~protected


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def nearest_indices_from_subset(distances: np.ndarray, subset_indices: np.ndarray) -> np.ndarray:
    return subset_indices[np.argmin(distances[:, subset_indices], axis=1)]


def quantize_to_palette(image: Image.Image, palette: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.array(image, dtype=np.uint8)
    pixel_lab = rgb_to_lab(pixels.reshape(-1, 3))
    palette_rgb = np.stack([cast(np.ndarray, entry["rgb"]) for entry in palette], axis=0)
    palette_lab = rgb_to_lab(palette_rgb)

    # Pull the outline mask off the canvas (filled in build_grid).
    outline_mask = image.info.get("outline_mask")
    if outline_mask is None:
        outline_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

    distances = np.sum((pixel_lab[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)
    indices = np.argmin(distances, axis=1)
    code_lookup = [str(entry["code"]) for entry in palette]

    subset_lookup = {
        # Bright yellows/golds for the sun emblem and trim.
        "sun_emblem": {"A4", "A5", "A8", "A13", "A15", "A17", "A20", "A26", "G5"},
        "sun_center": {"A4", "A8", "A15", "A17", "A20"},
        "gold_trim": {"A4", "A5", "A8", "A26", "A6", "G5", "G6", "A13"},
        # Light blues/cyans for hair.
        "hair": {"C2", "C3", "C4", "C13", "C14", "C21", "C23", "C24", "C27", "C28", "D17"},
        # Rich blues for eyes.
        "eyes": {"C5", "C6", "C7", "C8", "C9", "C10", "C11", "C16", "C17", "C20", "C26"},
        # Peach/pink/cream skin tones.
        "skin": {"A1", "A2", "A18", "A25", "F17", "G13", "G21", "M9", "M13"},
        # Whites and near-whites for the collar.
        "collar": {"H1", "H2", "H9", "H11", "H14", "H22", "H3"},
        # Dark blues for the main outfit.
        "dark_outfit": {"C12", "C18", "C29", "D3", "D4", "D10", "D15", "D22", "B22"},
        # Bright cyans/blues for the chest gem.
        "blue_gem": {"C4", "C5", "C10", "C11", "C17", "C24", "C26"},
        # Silvery grays for swords in the background.
        "sword": {"H1", "H2", "H3", "H9", "H10", "H11", "H14", "H20", "H22"},
    }
    subset_indices = {
        key: np.array([idx for idx, code in enumerate(code_lookup) if code in codes], dtype=np.int64)
        for key, codes in subset_lookup.items()
    }

    current_codes = np.array([code_lookup[i] for i in indices])

    # Sun emblem: force golden subset.  Any pixel inside the detected emblem
    # that is not already a chosen gold is pushed toward the gold family so
    # the rays stay clearly visible.  Skin/eyes/hair cells are protected
    # so the emblem mask never overwrites them.
    if subset_indices["sun_emblem"].size:
        sun_flat = build_sun_emblem_mask(pixels).reshape(-1)
        sun_emblem_codes = {"A4", "A5", "A8", "A13", "A15", "A17", "A20", "A26", "G5"}
        protected_codes = (
            {"A1", "A2", "A18", "A25", "F17", "G13", "G21", "M9", "M13"}  # skin
            | {"H1", "H2", "H9", "H11", "H14", "H22"}  # collar whites
            | {"C5", "C6", "C7", "C8", "C9", "C10", "C11", "C16", "C17", "C20", "C26"}  # eyes
            | {"C2", "C3", "C4", "C13", "C14", "C21", "C23", "C24", "C27", "C28", "D17"}  # hair
        )
        already_sun = np.isin(current_codes, list(sun_emblem_codes))
        already_protected = np.isin(current_codes, list(protected_codes))
        indices = np.where(
            sun_flat & ~already_sun & ~already_protected,
            nearest_indices_from_subset(distances, subset_indices["sun_emblem"]),
            indices,
        )

    # Sun center: brightest golds.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["sun_center"].size:
        center_flat = build_sun_center_mask(pixels).reshape(-1)
        bright_center_codes = np.isin(
            current_codes,
            ["A4", "A5", "A6", "A8", "A13", "A26", "G5", "G6", "H1", "H2", "H9", "H11"],
        )
        indices = np.where(
            center_flat & bright_center_codes,
            nearest_indices_from_subset(distances, subset_indices["sun_center"]),
            indices,
        )

    # Hair: force light blue family.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["hair"].size:
        hair_flat = build_hair_mask(pixels).reshape(-1)
        hair_replace_codes = np.isin(
            current_codes,
            ["H1", "H2", "H3", "H9", "H10", "H11", "H14", "H17", "H20", "H22",
             "D1", "D8", "D9", "D11", "D16", "D23", "D26", "E3", "E24"],
        )
        indices = np.where(
            hair_flat & hair_replace_codes,
            nearest_indices_from_subset(distances, subset_indices["hair"]),
            indices,
        )

    # Eyes: force rich blues.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["eyes"].size:
        eyes_flat = build_eyes_mask(pixels).reshape(-1)
        eye_replace_codes = np.isin(
            current_codes,
            ["H1", "H2", "H3", "H9", "H10", "H11", "H14", "H17", "H20", "H22",
             "C12", "C18", "C29", "D4", "D10", "D15", "D22"],
        )
        indices = np.where(
            eyes_flat & eye_replace_codes,
            nearest_indices_from_subset(distances, subset_indices["eyes"]),
            indices,
        )

    # Skin: force flesh tones.  This runs early (before gold_trim) and
    # rewrites any cell inside the face ROI that is not already a skin
    # palette code, so gold/orange contamination from later passes is
    # pulled back into the face palette.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["skin"].size:
        skin_flat = build_skin_mask(pixels).reshape(-1)
        skin_palette = {"A1", "A2", "A18", "A25", "F17", "G13", "G21", "M9", "M13"}
        already_skin = np.isin(current_codes, list(skin_palette))
        indices = np.where(
            skin_flat & ~already_skin,
            nearest_indices_from_subset(distances, subset_indices["skin"]),
            indices,
        )

    # Collar: force whites.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["collar"].size:
        collar_flat = build_collar_mask(pixels).reshape(-1)
        collar_replace_codes = np.isin(
            current_codes,
            ["H1", "H2", "H3", "H9", "H10", "H11", "H14", "H17", "H20", "H22",
             "D1", "D8", "D9", "D11", "D16", "D23", "D26", "E3", "E24"],
        )
        indices = np.where(
            collar_flat & collar_replace_codes,
            nearest_indices_from_subset(distances, subset_indices["collar"]),
            indices,
        )

    # Dark outfit: keep navy/dark blues and avoid grays.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["dark_outfit"].size:
        outfit_flat = build_dark_outfit_mask(pixels).reshape(-1)
        outfit_replace_codes = np.isin(
            current_codes,
            ["H5", "H6", "H7", "H15", "H16", "H17", "H23", "M3", "M12", "M15"],
        )
        indices = np.where(
            outfit_flat & outfit_replace_codes,
            nearest_indices_from_subset(distances, subset_indices["dark_outfit"]),
            indices,
        )

    # Gold trim: make sure yellow accents stay yellow.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["gold_trim"].size:
        gold_flat = build_gold_trim_mask(pixels).reshape(-1)
        gold_replace_codes = np.isin(
            current_codes,
            ["H1", "H2", "H3", "H9", "H10", "H11", "H14", "H17", "H20", "H22",
             "C12", "C18", "C29", "D4", "D10", "D15", "D22", "B22"],
        )
        indices = np.where(
            gold_flat & gold_replace_codes,
            nearest_indices_from_subset(distances, subset_indices["gold_trim"]),
            indices,
        )

    # Blue gem: force bright cyans/blues.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["blue_gem"].size:
        gem_flat = build_blue_gem_mask(pixels).reshape(-1)
        gem_replace_codes = np.isin(
            current_codes,
            ["H1", "H2", "H3", "H9", "H10", "H11", "H14", "H17", "H20", "H22",
             "C12", "C18", "C29", "D4", "D10", "D15", "D22"],
        )
        indices = np.where(
            gem_flat & gem_replace_codes,
            nearest_indices_from_subset(distances, subset_indices["blue_gem"]),
            indices,
        )

    # Swords: silvery grays.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["sword"].size:
        sword_flat = build_sword_mask(pixels).reshape(-1)
        sword_replace_codes = np.isin(
            current_codes,
            ["H1", "H2", "H3", "H9", "H10", "H11", "H14", "H17", "H20", "H22",
             "C28", "D11", "D17"],
        )
        indices = np.where(
            sword_flat & sword_replace_codes,
            nearest_indices_from_subset(distances, subset_indices["sword"]),
            indices,
        )

    # Final pass: the sun emblem must stay golden even if other region masks
    # overlapped it (the shoulder is also part of the torso/dark-outfit region).
    # Skip cells that have already been locked to a non-gold region (skin,
    # eyes, hair, etc.) so this pass can only add gold, never steal other
    # colors.
    current_codes = np.array([code_lookup[i] for i in indices])
    if subset_indices["sun_emblem"].size:
        sun_flat = build_sun_emblem_mask(pixels).reshape(-1)
        sun_emblem_codes = {"A4", "A5", "A8", "A13", "A15", "A17", "A20", "A26", "G5"}
        protected_codes = (
            {"A1", "A2", "A18", "A25", "F17", "G13", "G21", "M9", "M13"}  # skin
            | {"H1", "H2", "H9", "H11", "H14", "H22"}  # collar whites
            | {"C5", "C6", "C7", "C8", "C9", "C10", "C11", "C16", "C17", "C20", "C26"}  # eyes
            | {"C2", "C3", "C4", "C13", "C14", "C21", "C23", "C24", "C27", "C28", "D17"}  # hair
        )
        already_sun = np.isin(current_codes, list(sun_emblem_codes))
        already_protected = np.isin(current_codes, list(protected_codes))
        indices = np.where(
            sun_flat & ~already_sun & ~already_protected,
            nearest_indices_from_subset(distances, subset_indices["sun_emblem"]),
            indices,
        )

    # Black outline pass: any cell covered by the detected ink-line is
    # forced to H7 (true black).  This re-establishes crisp contours
    # around the silhouette, hair, eyes, and sun emblem.
    black_code = code_lookup.index("H7") if "H7" in code_lookup else None
    if black_code is not None:
        indices = np.where(outline_mask.reshape(-1), black_code, indices)

    code_grid = indices.reshape(GRID_SIZE, GRID_SIZE)

    # Build the protected mask: anything that is a deliberate boundary
    # (sun emblem, gold trim, eyes, outline, color-region borders) must not
    # be wiped out by the smoothing pass.  The complement is treated as
    # fillable terrain where pepper noise can be safely replaced by the
    # majority neighborhood color.
    protected_mask = (
        outline_mask
        | build_sun_emblem_mask(pixels)
        | build_gold_trim_mask(pixels)
        | build_eyes_mask(pixels)
        | build_blue_gem_mask(pixels)
        | build_hair_mask(pixels)
        | build_skin_mask(pixels)
        | build_dark_outfit_mask(pixels)
        | build_sword_mask(pixels)
    )
    code_grid = majority_smooth(code_grid, pixels, distances, protected_mask, len(palette))

    # Force the plain facial skin to a single cream color (A1 #FAF4C8).
    # Any cell inside the face ROI that has already been classified as a
    # flesh tone is recolored to A1.  Eyes, hair, collar, blush and outlines
    # are not in the skin family, so they keep their original colors.
    a1_code = code_lookup.index("A1") if "A1" in code_lookup else None
    if a1_code is not None:
        face_base = build_face_base_mask(pixels)
        skin_family = {
            "A1", "A2", "A18", "A25",
            "F17", "G13", "G16", "G21",
            "M9", "M13", "A21", "A23", "A24",
        }
        skin_indices = np.array([code_lookup.index(c) for c in skin_family if c in code_lookup], dtype=np.int64)
        is_skin_family = np.isin(code_grid, skin_indices)
        code_grid = np.where(face_base & is_skin_family, a1_code, code_grid)

    quantized = palette_rgb[code_grid.reshape(-1)].reshape(GRID_SIZE, GRID_SIZE, 3).astype(np.uint8)
    return code_grid, quantized


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

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
    for row in range(GRID_SIZE):
        draw.text((8, margin_top + row * cell + 5), f"{row + 1:02d}", fill=(40, 40, 40), font=font)
    for col in range(GRID_SIZE):
        draw.text((margin_left + col * cell + 4, 8), f"{col + 1:02d}", fill=(40, 40, 40), font=font)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x0, y0 = margin_left + col * cell, margin_top + row * cell
            draw.rectangle(
                (x0, y0, x0 + cell, y0 + cell),
                fill=tuple(int(v) for v in quantized[row, col]),
                outline=(160, 160, 160),
            )
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
    Image.fromarray(quantized, mode="RGB").resize((GRID_SIZE * 16, GRID_SIZE * 16), Image.Resampling.NEAREST).save(
        OUTPUT_DIR / f"{stem}_quantized.png"
    )
    draw_chart(OUTPUT_DIR / f"{stem}_chart.png", quantized, code_grid, palette)
    write_code_matrix(OUTPUT_DIR / f"{stem}_codes.txt", code_grid, palette)
    write_code_csv(OUTPUT_DIR / f"{stem}_codes.csv", code_grid, palette)
    write_usage_csv(OUTPUT_DIR / f"{stem}_usage.csv", code_grid, palette)
    print(f"Generated Shino outputs in {OUTPUT_DIR} with size {GRID_SIZE}")


if __name__ == "__main__":
    main()
