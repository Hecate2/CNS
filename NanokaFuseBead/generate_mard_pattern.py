from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
SOURCE_IMAGE = ROOT / "ChatGPT Image 2026年5月22日 17_50_51.png"
OUTPUT_DIR = ROOT / "outputs"
PALETTE_URL = "https://pd.anqstar.com/colors"
GRID_SIZE = 52
SAMPLE_FACTOR = 8
WHITE_THRESHOLD = 245
EDGE_SHARPEN_STRENGTH = 1.0
WAIST_WHITE_ROI = {
    "x_min": 0.05,
    "x_max": 0.33,
    "y_min": 0.40,
    "y_max": 0.70,
}
SKIRT_ROI = {
    "x_min": 0.29,
    "x_max": 0.61,
    "y_min": 0.58,
    "y_max": 0.86,
}
BOOTS_ROI = {
    "x_min": 0.29,
    "x_max": 0.58,
    "y_min": 0.83,
    "y_max": 1.00,
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

    return sorted(entries, key=palette_sort_key)


def palette_sort_key(item: dict[str, object]) -> tuple[str, int]:
    code = str(item["code"])
    return code[0], int(code[1:])


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64) / 255.0
    mask = rgb <= 0.04045
    rgb_linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    return rgb_linear @ matrix.T


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    white = np.array([0.95047, 1.0, 1.08883])
    xyz_scaled = xyz / white
    epsilon = 216 / 24389
    kappa = 24389 / 27

    def f(values: np.ndarray) -> np.ndarray:
        return np.where(values > epsilon, np.cbrt(values), (kappa * values + 16) / 116)

    fx, fy, fz = np.moveaxis(f(xyz_scaled), -1, 0)
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([l, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    return xyz_to_lab(rgb_to_xyz(rgb))


def find_subject_bbox(rgb: np.ndarray) -> tuple[int, int, int, int]:
    mask = np.any(rgb < WHITE_THRESHOLD, axis=2)
    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    padding = max(8, int(round(max(width, height) * 0.02)))
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(rgb.shape[1] - 1, x1 + padding)
    y1 = min(rgb.shape[0] - 1, y1 + padding)
    return x0, y0, x1, y1


def median_downsample(image: Image.Image, width: int, height: int, sample_factor: int) -> Image.Image:
    hires = image.resize((width * sample_factor, height * sample_factor), Image.Resampling.LANCZOS)
    arr = np.array(hires, dtype=np.uint8)
    arr = arr.reshape(height, sample_factor, width, sample_factor, 3)
    median = np.median(arr, axis=(1, 3)).astype(np.uint8)
    return Image.fromarray(median, mode="RGB")


def edge_aware_sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    arr = np.array(image, dtype=np.float32)

    # High-pass term keeps boundaries between major color regions crisp.
    blur = np.array(image.filter(ImageFilter.GaussianBlur(radius=0.9)), dtype=np.float32)
    high_pass = arr - blur

    # Edge mask prevents boosting flat regions too aggressively.
    edge_map = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edge = np.array(edge_map, dtype=np.float32) / 255.0
    edge = np.clip(edge * 1.6, 0.0, 1.0)
    weight = (0.45 + 1.2 * edge) * strength

    boosted = arr + high_pass * weight[..., None]
    boosted = np.clip(boosted, 0, 255).astype(np.uint8)
    result = Image.fromarray(boosted, mode="RGB")
    return result.filter(ImageFilter.UnsharpMask(radius=0.75, percent=int(70 * strength), threshold=1))


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

    crop_width, crop_height = enhanced.size
    scale = min(GRID_SIZE / crop_width, GRID_SIZE / crop_height)
    target_width = max(1, int(round(crop_width * scale)))
    target_height = max(1, int(round(crop_height * scale)))

    sampled = median_downsample(enhanced, target_width, target_height, SAMPLE_FACTOR)
    sampled = edge_aware_sharpen(sampled, strength=0.65 * EDGE_SHARPEN_STRENGTH)
    canvas = Image.new("RGB", (GRID_SIZE, GRID_SIZE), (255, 255, 255))
    offset_x = (GRID_SIZE - target_width) // 2
    offset_y = (GRID_SIZE - target_height) // 2
    canvas.paste(sampled, (offset_x, offset_y))
    return canvas


def roi_mask_from_bounds(bounds: dict[str, float]) -> np.ndarray:
    x0 = int(round(GRID_SIZE * bounds["x_min"]))
    x1 = int(round(GRID_SIZE * bounds["x_max"]))
    y0 = int(round(GRID_SIZE * bounds["y_min"]))
    y1 = int(round(GRID_SIZE * bounds["y_max"]))
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    mask[y0:y1, x0:x1] = True
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


def build_waist_white_priority_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(WAIST_WHITE_ROI)

    return (
        roi_mask
        & (luminance >= 175)
        & (chroma <= 40)
        & (neighbor_min <= 95)
        & ((neighbor_max - neighbor_min) >= 110)
    )


def build_skirt_grid_priority_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(SKIRT_ROI)
    base = (
        roi_mask
        & (neighbor_min <= 40)
        & ((neighbor_max - neighbor_min) >= 70)
        & (((luminance >= 85) & (chroma <= 28)) | (luminance >= 150))
    )
    neighbor_count = build_neighbor_count(base)
    bridge = (
        roi_mask
        & (luminance >= 55)
        & (luminance <= 170)
        & (chroma <= 24)
        & ((neighbor_max - neighbor_min) >= 55)
        & (neighbor_count >= 2)
    )
    grid = base | bridge
    grid = fill_linear_gaps(grid)
    grid = fill_linear_gaps(grid)
    return grid & roi_mask & (luminance >= 45) & (chroma <= 32)


def build_boot_metal_priority_mask(pixels: np.ndarray) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(BOOTS_ROI)
    base = (
        roi_mask
        & (luminance >= 150)
        & (chroma <= 22)
        & (neighbor_min <= 60)
        & ((neighbor_max - neighbor_min) >= 100)
    )
    neighbor_count = build_neighbor_count(base)
    bridge = (
        roi_mask
        & (luminance >= 105)
        & (chroma <= 18)
        & ((neighbor_max - neighbor_min) >= 70)
        & (neighbor_count >= 1)
    )
    metal = base | bridge
    metal = fill_linear_gaps(metal)
    return metal & roi_mask & (luminance >= 95) & (chroma <= 24)


def build_dark_relief_mask(
    pixels: np.ndarray, skirt_grid_mask: np.ndarray, boot_metal_mask: np.ndarray
) -> np.ndarray:
    luminance, chroma, neighbor_min, neighbor_max = build_local_stats(pixels)
    roi_mask = roi_mask_from_bounds(SKIRT_ROI) | roi_mask_from_bounds(BOOTS_ROI)
    return (
        roi_mask
        & ~skirt_grid_mask
        & ~boot_metal_mask
        & (luminance >= 18)
        & (luminance <= 72)
        & (chroma >= 4)
        & ((neighbor_max - neighbor_min) >= 20)
        & (neighbor_max >= 35)
    )


def nearest_indices_from_subset(distances: np.ndarray, subset_indices: np.ndarray) -> np.ndarray:
    subset_distances = distances[:, subset_indices]
    return subset_indices[np.argmin(subset_distances, axis=1)]


def quantize_to_palette(image: Image.Image, palette: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.array(image, dtype=np.uint8)
    flat_pixels = pixels.reshape(-1, 3)
    pixel_lab = rgb_to_lab(flat_pixels)
    palette_rgb = np.stack([entry["rgb"] for entry in palette], axis=0)
    palette_lab = rgb_to_lab(palette_rgb)

    distances = np.sum((pixel_lab[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)
    indices = np.argmin(distances, axis=1)

    code_lookup = [str(entry["code"]) for entry in palette]

    subset_lookup = {
        "white": {"H1", "H2"},
        "skirt_grid": {"H9", "H10", "H11", "H22", "H3", "H4", "H20"},
        "boot_metal": {"H1", "H9", "H10", "H11", "H14", "H20", "H22"},
        "dark_relief": {"H5", "H6", "C12", "C18", "C29", "D10", "M15"},
    }
    subset_indices = {
        key: np.array([index for index, code in enumerate(code_lookup) if code in codes], dtype=np.int64)
        for key, codes in subset_lookup.items()
    }

    white_mask = build_waist_white_priority_mask(pixels).reshape(-1)
    skirt_grid_mask = build_skirt_grid_priority_mask(pixels)
    boot_metal_mask = build_boot_metal_priority_mask(pixels)
    dark_relief_mask = build_dark_relief_mask(pixels, skirt_grid_mask, boot_metal_mask)

    if subset_indices["white"].size:
        indices = np.where(white_mask, nearest_indices_from_subset(distances, subset_indices["white"]), indices)

    if subset_indices["skirt_grid"].size:
        grid_flat = skirt_grid_mask.reshape(-1)
        indices = np.where(grid_flat, nearest_indices_from_subset(distances, subset_indices["skirt_grid"]), indices)

    if subset_indices["boot_metal"].size:
        metal_flat = boot_metal_mask.reshape(-1)
        indices = np.where(metal_flat, nearest_indices_from_subset(distances, subset_indices["boot_metal"]), indices)

    if subset_indices["dark_relief"].size:
        dark_flat = dark_relief_mask.reshape(-1)
        current_codes = np.array([code_lookup[index] for index in indices])
        overdark = np.isin(current_codes, ["H7", "H16"])
        relief_mask = dark_flat & overdark
        indices = np.where(relief_mask, nearest_indices_from_subset(distances, subset_indices["dark_relief"]), indices)

    quantized = palette_rgb[indices].reshape(GRID_SIZE, GRID_SIZE, 3).astype(np.uint8)
    return indices.reshape(GRID_SIZE, GRID_SIZE), quantized


def write_code_matrix(path: Path, code_grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(entry["code"]) for entry in palette]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("    " + " ".join(f"{col:>4}" for col in range(1, GRID_SIZE + 1)) + "\n")
        for row in range(GRID_SIZE):
            codes = [f"{code_lookup[index]:>4}" for index in code_grid[row]]
            handle.write(f"{row + 1:>3} " + " ".join(codes) + "\n")


def write_code_csv(path: Path, code_grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(entry["code"]) for entry in palette]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + [str(col) for col in range(1, GRID_SIZE + 1)])
        for row in range(GRID_SIZE):
            writer.writerow([str(row + 1)] + [code_lookup[index] for index in code_grid[row]])


def write_usage_csv(path: Path, code_grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(entry["code"]) for entry in palette]
    hex_lookup = [str(entry["hex"]) for entry in palette]
    counts = Counter(code_lookup[index] for index in code_grid.reshape(-1))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "count", "hex", "rgb"])
        for entry in palette:
            code = str(entry["code"])
            count = counts.get(code, 0)
            if count == 0:
                continue
            rgb = entry["rgb"]
            rgb_text = f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"
            writer.writerow([code, count, str(entry["hex"]), rgb_text])


def draw_chart(path: Path, quantized: np.ndarray, code_grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    cell = 20
    margin_left = 42
    margin_top = 32
    width = margin_left + GRID_SIZE * cell + 1
    height = margin_top + GRID_SIZE * cell + 1
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for row in range(GRID_SIZE):
        y = margin_top + row * cell
        draw.text((8, y + 5), f"{row + 1:02d}", fill=(40, 40, 40), font=font)
    for col in range(GRID_SIZE):
        x = margin_left + col * cell
        draw.text((x + 4, 8), f"{col + 1:02d}", fill=(40, 40, 40), font=font)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x0 = margin_left + col * cell
            y0 = margin_top + row * cell
            x1 = x0 + cell
            y1 = y0 + cell
            fill = tuple(int(value) for value in quantized[row, col])
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(160, 160, 160))

    image.save(path)


def save_quantized_image(path: Path, quantized: np.ndarray) -> None:
    image = Image.fromarray(quantized, mode="RGB")
    image = image.resize((GRID_SIZE * 16, GRID_SIZE * 16), Image.Resampling.NEAREST)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MARD bead pattern from source image")
    parser.add_argument("--size", type=int, default=52, help="Output grid side length, e.g. 52 or 64")
    args = parser.parse_args()

    if args.size < 16 or args.size > 256:
        raise ValueError("--size must be between 16 and 256")

    global GRID_SIZE
    GRID_SIZE = int(args.size)

    OUTPUT_DIR.mkdir(exist_ok=True)

    palette = fetch_palette()
    sampled = build_grid(SOURCE_IMAGE)
    code_grid, quantized = quantize_to_palette(sampled, palette)

    stem = output_stem()
    save_quantized_image(OUTPUT_DIR / f"{stem}_quantized.png", quantized)
    draw_chart(OUTPUT_DIR / f"{stem}_chart.png", quantized, code_grid, palette)
    write_code_matrix(OUTPUT_DIR / f"{stem}_codes.txt", code_grid, palette)
    write_code_csv(OUTPUT_DIR / f"{stem}_codes.csv", code_grid, palette)
    write_usage_csv(OUTPUT_DIR / f"{stem}_usage.csv", code_grid, palette)

    print("Generated outputs in", OUTPUT_DIR, "with size", GRID_SIZE)


if __name__ == "__main__":
    main()
