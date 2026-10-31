"""Fuse-bead (MARD) pattern generator for the rose-witch-hat illustration.

Design notes
------------
This is a deliberately "no-enhancement" variant of the pipeline used by
ChthollyFuseBead / ShinoFuseBead / NanokaFuseBead:

* No bounding-box crop.
* No autocontrast / saturation / contrast / unsharp / edge sharpening.
* No global noise removal.

What is kept is only the part that decides how the beads look:

* kernel-weighted dominant-colour cell sampling
* CIELAB nearest-neighbour palette quantization + a global colour budget
* four hand-directed touches, each confined to its own region:
  - the white collar / shoulder mantle gets de-speckled,
  - the chest brooch is lifted onto the bright blue family with a specular,
  - each iris gets a white gleam,
  - a faint nose dash and mouth dash are drawn on the face.

Everything outside those regions keeps its raw quantized beads.

Known limit of the no-enhancement choice: the artwork's palest tones (the rose
petals, the flat parts of the face) sit within dE ~5 of pure white, and the
MARD card has almost no near-white pink steps.  Those areas therefore collapse
to H2 unless the grid is large enough for each petal to own several cells, so
``--size 140`` (or a mild saturation lift, which this script deliberately does
not do) is the lever if the rose needs to read more strongly.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SOURCE_IMAGE = ROOT / "蓝发少女与玫瑰魔女帽.png"
OUTPUT_DIR = ROOT / "outputs"
PALETTE_URL = "https://pd.anqstar.com/colors"
PALETTE_CACHE = ROOT / "palette_cache.json"

GRID_SIZE = 100
SAMPLE_FACTOR = 9

# --- dominant-colour cell sampling -----------------------------------------
DOMINANT_COLOR_SIGMA = 26.0
SPATIAL_SIGMA_RATIO = 0.42

# --- colour budget (agglomerative palette merging) --------------------------
MAX_COLORS = 40
PALETTE_MERGE_DELTA_E = 9.0

# --- targeted clean-up: white collar / shoulder mantle ONLY -----------------
GARMENT_ROI = {"x_min": 0.10, "x_max": 0.94, "y_min": 0.60, "y_max": 1.00}
FABRIC_RAMP = ("H2", "D16", "D11", "D1")
FABRIC_LIGHTNESS_MIN = 70.0
FABRIC_CHROMA_MAX = 26.0
FABRIC_NEUTRAL_A_MIN = -9.0
FABRIC_PASSES = 3
SPECK_MAX_CELLS = 4

# --- brooch (chest gem) highlight ------------------------------------------
GEM_ROI = {"x_min": 0.455, "x_max": 0.600, "y_min": 0.685, "y_max": 0.795}
GEM_CHROMA_MIN = 32.0
GEM_LIGHTNESS_MAX = 82.0
GEM_TONES = ("C6", "C24", "C13")

# --- eye highlights ---------------------------------------------------------
EYES_ROI = {"x_min": 0.330, "x_max": 0.680, "y_min": 0.420, "y_max": 0.530}
IRIS_CHROMA_MIN = 38.0
IRIS_LIGHTNESS_MAX = 66.0
IRIS_WARM_A_MIN = 5.0

# --- nose / mouth (normalised anchors, scaled to any grid size) -------------
NOSE_X = 0.510
NOSE_TOP_Y = 0.509
NOSE_CELLS = 2
MOUTH_CENTER_X = 0.520
MOUTH_Y = 0.5595
MOUTH_WIDTH = 4


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
def fetch_palette() -> list[dict[str, object]]:
    """Fetch the MARD colour card, with a local JSON cache as fallback."""
    if PALETTE_CACHE.exists():
        try:
            raw = json.loads(PALETTE_CACHE.read_text(encoding="utf-8"))
            if len(raw) >= 200:
                return _palette_from_raw(raw)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    req = Request(PALETTE_URL, headers={"User-Agent": "Mozilla/5.0"})
    html = urlopen(req, timeout=30).read().decode("utf-8", "ignore")
    pattern = re.compile(
        r'title="([A-HM]\d{1,2}) · (#[0-9A-Fa-f]{6}) · RGB\((\d+),\s*(\d+),\s*(\d+)\)"'
    )
    raw: list[dict[str, object]] = []
    seen: set[str] = set()
    for code, hex_value, red, green, blue in pattern.findall(html):
        if code in seen:
            continue
        seen.add(code)
        raw.append(
            {
                "code": code,
                "hex": hex_value.upper(),
                "rgb": [int(red), int(green), int(blue)],
            }
        )
    if len(raw) < 200:
        raise RuntimeError(f"Palette fetch returned only {len(raw)} colours")
    PALETTE_CACHE.write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    return _palette_from_raw(raw)


def _palette_from_raw(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for item in raw:
        rgb = np.array(item["rgb"], dtype=np.uint8)
        entries.append({"code": str(item["code"]), "hex": str(item["hex"]), "rgb": rgb})
    return sorted(entries, key=lambda i: (str(i["code"])[0], int(str(i["code"])[1:])))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float64) / 255.0
    mask = rgb <= 0.04045
    linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    scaled = xyz / white
    epsilon = 216 / 24389
    kappa = 24389 / 27
    fx = np.where(scaled[..., 0] > epsilon, np.cbrt(scaled[..., 0]), (kappa * scaled[..., 0] + 16) / 116)
    fy = np.where(scaled[..., 1] > epsilon, np.cbrt(scaled[..., 1]), (kappa * scaled[..., 1] + 16) / 116)
    fz = np.where(scaled[..., 2] > epsilon, np.cbrt(scaled[..., 2]), (kappa * scaled[..., 2] + 16) / 116)
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


# ---------------------------------------------------------------------------
# Sampling: kernel-weighted dominant colour per cell (old algorithm #4)
# ---------------------------------------------------------------------------
def sample_cells(source: Image.Image, width: int, height: int, factor: int) -> Image.Image:
    """Downsample by seeking the dominant colour cluster inside every cell.

    For each cell an 8-ish pixel block is taken at a resolution close to the
    source, so no artificial detail is invented.  Inside the block every pixel
    votes for itself with a Gaussian colour-similarity kernel; the winner is
    the local colour mode.  The cell value is the similarity-weighted average
    of that mode's cluster, which keeps anti-aliased edges from bleeding in.
    """
    hires = source.resize((width * factor, height * factor), Image.Resampling.LANCZOS)
    blocks = np.asarray(hires, dtype=np.float32).reshape(height, factor, width, factor, 3)
    blocks = np.transpose(blocks, (0, 2, 1, 3, 4))

    axis = np.linspace(-(factor - 1) / 2, (factor - 1) / 2, factor, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    sigma_s = max(1.0, factor * SPATIAL_SIGMA_RATIO)
    spatial = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * sigma_s**2)).reshape(-1)
    colour_denom = 2.0 * float(DOMINANT_COLOR_SIGMA) ** 2

    out = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            block = blocks[row, col].reshape(-1, 3)
            delta = block[:, None, :] - block[None, :, :]
            dist_sq = np.einsum("ijk,ijk->ij", delta, delta)
            support = np.exp(-dist_sq / colour_denom)
            scores = support @ spatial
            mode = int(np.argmax(scores))

            weights = spatial * support[mode]
            total = float(weights.sum())
            if total < 1e-6:
                weights, total = spatial, float(spatial.sum())
            out[row, col] = np.clip(np.rint((block * weights[:, None]).sum(axis=0) / total), 0, 255)

    return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# Colour budget: agglomerative palette merging (keeps bead count sane)
# ---------------------------------------------------------------------------
def reduce_palette(
    dist_sq: np.ndarray,
    max_colors: int,
    merge_delta_e: float,
    code_lookup: list[str] | None = None,
    required_codes: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Merge rarely used palette entries into their nearest neighbour.

    A colour may only disappear when doing so displaces 90% of its cells by
    less than ``merge_delta_e`` in CIELAB.  This is perceptually safe: it
    removes palette noise the eye would never resolve, and it is the single
    strongest "turn similar colours into one patch" lever available.
    """
    keep = np.ones(dist_sq.shape[1], dtype=bool)
    blocked = np.zeros(dist_sq.shape[1], dtype=bool)
    if code_lookup is not None:
        for code in required_codes:
            if code in code_lookup:
                blocked[code_lookup.index(code)] = True
    idx = np.argmin(dist_sq, axis=1)
    huge = np.iinfo(np.int64).max

    while int(keep.sum()) > max_colors:
        counts = np.bincount(idx, minlength=dist_sq.shape[1])
        priority = np.where(keep & ~blocked & (counts > 0), counts, huge)
        merged = False
        for victim in np.argsort(priority):
            if priority[victim] == huge:
                break
            cells = np.where(idx == victim)[0]
            alternatives = np.where(keep)[0]
            alternatives = alternatives[alternatives != victim]
            if alternatives.size == 0:
                blocked[victim] = True
                continue
            sampled = np.sqrt(np.min(dist_sq[np.ix_(cells, alternatives)], axis=1))
            if np.percentile(sampled, 90) > merge_delta_e:
                blocked[victim] = True
                continue
            keep[victim] = False
            best = np.argmin(dist_sq[np.ix_(cells, alternatives)], axis=1)
            idx[cells] = alternatives[best]
            merged = True
            break
        if not merged:
            break

    kept = np.where(keep)[0]
    idx = kept[np.argmin(dist_sq[:, kept], axis=1)]
    compact = -np.ones(keep.size, dtype=np.int64)
    compact[kept] = np.arange(kept.size)
    return compact[idx], kept


# ---------------------------------------------------------------------------
# Region consolidation: absorb tiny connected patches (strengthened #7)
# ---------------------------------------------------------------------------
def label_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of equal labels."""
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for start_y in range(height):
        for start_x in range(width):
            if visited[start_y, start_x]:
                continue
            code = grid[start_y, start_x]
            visited[start_y, start_x] = True
            stack = [(start_y, start_x)]
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                if y > 0 and not visited[y - 1, x] and grid[y - 1, x] == code:
                    visited[y - 1, x] = True
                    stack.append((y - 1, x))
                if y + 1 < height and not visited[y + 1, x] and grid[y + 1, x] == code:
                    visited[y + 1, x] = True
                    stack.append((y + 1, x))
                if x > 0 and not visited[y, x - 1] and grid[y, x - 1] == code:
                    visited[y, x - 1] = True
                    stack.append((y, x - 1))
                if x + 1 < width and not visited[y, x + 1] and grid[y, x + 1] == code:
                    visited[y, x + 1] = True
                    stack.append((y, x + 1))
            components.append(cells)
    return components


def roi_mask(shape: tuple[int, int], bounds: dict[str, float]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    x0, x1 = int(round(width * bounds["x_min"])), int(round(width * bounds["x_max"]))
    y0, y1 = int(round(height * bounds["y_min"])), int(round(height * bounds["y_max"]))
    mask[y0:y1, x0:x1] = True
    return mask


def lab_channels(pixels_lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return pixels_lab[..., 0], np.hypot(pixels_lab[..., 1], pixels_lab[..., 2])


def compact_lookup(kept: np.ndarray, code_lookup: list[str], codes: tuple[str, ...]) -> np.ndarray:
    """Palette positions (in the reduced palette) of the requested codes."""
    position = {int(full): slot for slot, full in enumerate(kept)}
    found: list[int] = []
    for code in codes:
        if code not in code_lookup:
            continue
        slot = position.get(code_lookup.index(code))
        if slot is not None:
            found.append(slot)
    return np.array(found, dtype=np.int64)


def flatten_region(grid: np.ndarray, mask: np.ndarray, passes: int) -> np.ndarray:
    """Majority filter applied ONLY inside ``mask``.

    Everything outside the mask is frozen, so this cannot touch the rest of the
    pattern.  Used to weld the speckled shading of the white collar/mantle into
    a few flat tones.
    """
    height, width = grid.shape
    for _ in range(passes):
        updated = grid.copy()
        changed = 0
        for y in range(height):
            for x in range(width):
                if not mask[y, x]:
                    continue
                current = int(grid[y, x])
                patch = grid[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2].reshape(-1)
                codes, counts = np.unique(patch, return_counts=True)
                winner = int(np.argmax(counts))
                if int(codes[winner]) != current and int(counts[winner]) >= 4:
                    updated[y, x] = int(codes[winner])
                    changed += 1
        grid = updated
        if changed == 0:
            break
    return grid


def clean_white_fabric(
    grid: np.ndarray,
    kept: np.ndarray,
    code_lookup: list[str],
    pixels_lab: np.ndarray,
    palette_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Removes the speckle on the white collar and shoulder mantle.

    Only cells that are (a) inside the garment ROI, (b) light and (c) not cyan
    (the cyan hair strands that cross the mantle have a strongly negative a*)
    are candidates.  Each candidate is snapped onto the neutral ramp *by
    lightness*, so the blue cast of the shading disappears but the value
    structure of the folds survives; then the region is majority-filtered.
    Nothing outside the mask moves.
    """
    height, width = grid.shape
    lightness, chroma = lab_channels(pixels_lab)
    region = roi_mask((height, width), GARMENT_ROI)
    gate = (
        region
        & (lightness >= FABRIC_LIGHTNESS_MIN)
        & (chroma <= FABRIC_CHROMA_MAX)
        & (pixels_lab[..., 1] >= FABRIC_NEUTRAL_A_MIN)
    )

    ramp = compact_lookup(kept, code_lookup, FABRIC_RAMP)
    if not gate.any() or ramp.size == 0:
        return grid, gate

    ramp_lightness = palette_lab[kept[ramp], 0]
    rows, cols = np.where(gate)
    nearest = np.argmin(
        np.abs(lightness[rows, cols][:, None] - ramp_lightness[None, :]), axis=1
    )
    grid[rows, cols] = ramp[nearest]
    grid = flatten_region(grid, gate, FABRIC_PASSES)
    grid = absorb_specks(grid, region & (lightness >= FABRIC_LIGHTNESS_MIN), ramp)
    return grid, gate


def absorb_specks(
    grid: np.ndarray,
    region: np.ndarray,
    ramp: np.ndarray,
    max_cells: int = SPECK_MAX_CELLS,
) -> np.ndarray:
    """Eats tiny off-palette specks sitting on the white fabric.

    A hair strand crossing the mantle stays: it is a long connected run of its
    own colour.  A two-bead blob of the same colour is just residue from the
    downsampling and gets replaced by the surrounding fabric tone.
    """
    ramp_codes = {int(i) for i in ramp}
    height, width = grid.shape
    for cells in label_components(grid):
        if len(cells) >= max_cells:
            continue
        if not all(region[y, x] for y, x in cells):
            continue
        own = int(grid[cells[0][0], cells[0][1]])
        if own in ramp_codes:
            continue
        votes: Counter[int] = Counter()
        for y, x in cells:
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    neighbour = int(grid[ny, nx])
                    if neighbour in ramp_codes:
                        votes[neighbour] += 1
        if not votes:
            continue
        replacement = int(votes.most_common(1)[0][0])
        for y, x in cells:
            grid[y, x] = replacement
    return grid


def brighten_gem(
    grid: np.ndarray,
    dist_sq: np.ndarray,
    kept: np.ndarray,
    code_lookup: list[str],
    pixels_lab: np.ndarray,
) -> np.ndarray:
    """Lifts the chest brooch onto the brightest blue family plus one specular."""
    height, width = grid.shape
    lightness, chroma = lab_channels(pixels_lab)
    region = roi_mask((height, width), GEM_ROI)
    gem = region & (chroma >= GEM_CHROMA_MIN) & (lightness <= GEM_LIGHTNESS_MAX)

    components = [c for c in label_components(gem.astype(np.int64)) if gem[c[0][0], c[0][1]]]
    if not components:
        return grid
    stone = max(components, key=len)

    tones = compact_lookup(kept, code_lookup, GEM_TONES)
    if tones.size:
        rows = np.array([c[0] for c in stone])
        cols = np.array([c[1] for c in stone])
        nearest = np.argmin(dist_sq[np.ix_(rows * width + cols, kept[tones])], axis=1)
        grid[rows, cols] = tones[nearest]

    white = compact_lookup(kept, code_lookup, ("H2", "H1"))
    rows = np.array([c[0] for c in stone])
    cols = np.array([c[1] for c in stone])
    specular = int(np.argmax(lightness[rows, cols]))
    if white.size:
        grid[rows[specular], cols[specular]] = int(white[0])
    return grid


def add_eye_highlights(
    grid: np.ndarray,
    kept: np.ndarray,
    code_lookup: list[str],
    pixels_lab: np.ndarray,
) -> np.ndarray:
    """Adds a white specular bead to the upper-left quadrant of every iris."""
    height, width = grid.shape
    lightness, chroma = lab_channels(pixels_lab)
    region = roi_mask((height, width), EYES_ROI)
    iris = (
        region
        & (chroma >= IRIS_CHROMA_MIN)
        & (lightness <= IRIS_LIGHTNESS_MAX)
        & (pixels_lab[..., 1] >= IRIS_WARM_A_MIN)
    )

    components = [c for c in label_components(iris.astype(np.int64)) if iris[c[0][0], c[0][1]]]
    components = [c for c in components if len(c) >= 4]
    white = compact_lookup(kept, code_lookup, ("H2", "H1"))
    gleam = compact_lookup(kept, code_lookup, ("C13", "C24"))
    if not white.size or not gleam.size:
        return grid

    iris_codes = {int(i) for i in compact_lookup(kept, code_lookup, ("C16", "C7", "C12", "C24", "C13"))}
    for cells in components:
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        top, bottom = min(ys), max(ys)
        left, right = min(xs), max(xs)
        target = (top + max(1, (bottom - top) // 4), left + max(1, (right - left) // 4))
        if int(grid[target]) not in iris_codes:
            nearby = [
                (y, x)
                for y in range(top, bottom + 1)
                for x in range(left, right + 1)
                if int(grid[y, x]) in iris_codes
            ]
            if not nearby:
                continue
            nearby.sort(key=lambda p: (p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2)
            target = nearby[0]
        grid[target] = int(white[0])
        neighbour = (target[0], min(width - 1, target[1] + 1))
        if int(grid[neighbour]) in iris_codes and int(grid[neighbour]) != int(white[0]):
            grid[neighbour] = int(gleam[0])
    return grid


def draw_face_marks(
    grid: np.ndarray,
    kept: np.ndarray,
    code_lookup: list[str],
) -> np.ndarray:
    """A two-bead nose dash and a four-bead mouth line, one tone below the skin.

    Same recipe as the hand-drawn Shino pattern: a single short vertical dash
    for the nose and one short horizontal dash for the mouth, both in a tone
    that is barely darker than the face, so they read as a suggestion instead
    of line art.
    """
    height, width = grid.shape
    scale = height / 100.0
    nose_tone = compact_lookup(kept, code_lookup, ("E15", "D23", "D16"))
    mouth_tone = compact_lookup(kept, code_lookup, ("E15", "D16"))
    if not nose_tone.size or not mouth_tone.size:
        return grid

    nose_col = int(round(NOSE_X * width))
    nose_top = int(round(NOSE_TOP_Y * height))
    nose_height = max(2, int(round(NOSE_CELLS * scale)))
    for y in range(nose_top, min(height, nose_top + nose_height)):
        grid[y, nose_col] = int(nose_tone[0])

    mouth_row = int(round(MOUTH_Y * height))
    mouth_width = max(2, int(round(MOUTH_WIDTH * scale)))
    mouth_col = int(round(MOUTH_CENTER_X * width - mouth_width / 2))
    tone = int(mouth_tone[0])
    for dx in range(mouth_width):
        col = mouth_col + dx
        if 0 <= col < width:
            grid[mouth_row, col] = tone
    return grid


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_pattern(
    source_path: Path,
    grid_size: int,
    max_colors: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], Image.Image]:
    source = Image.open(source_path).convert("RGB")

    scale = min(grid_size / source.width, grid_size / source.height)
    target_width = max(1, int(round(source.width * scale)))
    target_height = max(1, int(round(source.height * scale)))

    sampled = sample_cells(source, target_width, target_height, SAMPLE_FACTOR)

    canvas = Image.new("RGB", (grid_size, grid_size), (255, 255, 255))
    offset_x = (grid_size - target_width) // 2
    offset_y = (grid_size - target_height) // 2
    canvas.paste(sampled, (offset_x, offset_y))

    palette = fetch_palette()
    code_lookup = [str(entry["code"]) for entry in palette]
    palette_rgb = np.stack([cast(np.ndarray, entry["rgb"]) for entry in palette], axis=0)
    palette_lab = rgb_to_lab(palette_rgb)

    pixels = np.asarray(canvas, dtype=np.uint8)
    pixels_lab = rgb_to_lab(pixels.reshape(-1, 3))
    dist_sq = np.sum((pixels_lab[:, None, :] - palette_lab[None, :, :]) ** 2, axis=2)

    required = FABRIC_RAMP + GEM_TONES + ("E15", "C16", "C7", "C12")
    grid, kept = reduce_palette(
        dist_sq, max_colors, PALETTE_MERGE_DELTA_E, code_lookup, required
    )
    grid = grid.reshape(grid_size, grid_size)
    sampled_lab = pixels_lab.reshape(grid_size, grid_size, 3)

    # The only denoising pass in the whole script, and it is confined to the
    # white collar / shoulder mantle (everything else keeps its raw beads).
    grid, _ = clean_white_fabric(grid, kept, code_lookup, sampled_lab, palette_lab)
    grid = brighten_gem(grid, dist_sq, kept, code_lookup, sampled_lab)
    grid = add_eye_highlights(grid, kept, code_lookup, sampled_lab)
    grid = draw_face_marks(grid, kept, code_lookup)

    reduced_palette = [palette[i] for i in kept]
    used_codes = sorted({code_lookup[int(kept[i])] for i in grid.reshape(-1)})
    print(f"  palette entries: {len(palette)} -> {len(kept)} (budget {max_colors})")
    print(f"  beads actually used: {len(used_codes)} colours")
    return grid, kept, reduced_palette, canvas


def render_swatches(grid: np.ndarray, palette: list[dict[str, object]]) -> np.ndarray:
    palette_rgb = np.stack([cast(np.ndarray, e["rgb"]) for e in palette], axis=0)
    return palette_rgb[grid].astype(np.uint8)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_code_matrix(path: Path, grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(e["code"]) for e in palette]
    size = grid.shape[0]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("    " + " ".join(f"{col:>4}" for col in range(1, size + 1)) + "\n")
        for row in range(size):
            handle.write(f"{row + 1:>3} " + " ".join(f"{code_lookup[int(i)]:>4}" for i in grid[row]) + "\n")


def write_code_csv(path: Path, grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(e["code"]) for e in palette]
    size = grid.shape[0]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + [str(col) for col in range(1, size + 1)])
        for row in range(size):
            writer.writerow([str(row + 1)] + [code_lookup[int(i)] for i in grid[row]])


def write_usage_csv(path: Path, grid: np.ndarray, palette: list[dict[str, object]]) -> None:
    code_lookup = [str(e["code"]) for e in palette]
    counts = Counter(code_lookup[int(i)] for i in grid.reshape(-1))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["code", "count", "hex", "rgb"])
        for entry in sorted(palette, key=lambda e: -counts.get(str(e["code"]), 0)):
            count = counts.get(str(entry["code"]), 0)
            if count == 0:
                continue
            rgb = cast(np.ndarray, entry["rgb"])
            writer.writerow(
                [str(entry["code"]), count, str(entry["hex"]), f"{int(rgb[0])},{int(rgb[1])},{int(rgb[2])}"]
            )


def draw_chart(path: Path, swatches: np.ndarray, palette: list[dict[str, object]]) -> None:
    size = swatches.shape[0]
    cell = 20
    margin_left, margin_top = 42, 32
    image = Image.new("RGB", (margin_left + size * cell + 1, margin_top + size * cell + 1), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for row in range(size):
        draw.text((8, margin_top + row * cell + 5), f"{row + 1:02d}", fill=(40, 40, 40), font=font)
    for col in range(size):
        draw.text((margin_left + col * cell + 4, 8), f"{col + 1:02d}", fill=(40, 40, 40), font=font)
    for row in range(size):
        for col in range(size):
            x0, y0 = margin_left + col * cell, margin_top + row * cell
            draw.rectangle(
                (x0, y0, x0 + cell, y0 + cell),
                fill=tuple(int(v) for v in swatches[row, col]),
                outline=(170, 170, 170),
            )
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a MARD fuse-bead pattern")
    parser.add_argument("--size", type=int, default=GRID_SIZE)
    parser.add_argument("--colors", type=int, default=MAX_COLORS, help="global colour budget")
    parser.add_argument("--source", type=Path, default=SOURCE_IMAGE)
    args = parser.parse_args()

    if not 16 <= args.size <= 256:
        raise ValueError("--size must be between 16 and 256")

    started = time.perf_counter()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Sampling {args.source.name} at {args.size}x{args.size} ...")
    grid, kept, palette, canvas = build_pattern(args.source, args.size, args.colors)
    swatches = render_swatches(grid, palette)

    stem = f"witchhat_mard_{args.size}x{args.size}"
    Image.fromarray(np.asarray(canvas, dtype=np.uint8), mode="RGB").resize(
        (args.size * 16, args.size * 16), Image.Resampling.NEAREST
    ).save(OUTPUT_DIR / f"{stem}_sampled.png")
    Image.fromarray(swatches, mode="RGB").resize(
        (args.size * 16, args.size * 16), Image.Resampling.NEAREST
    ).save(OUTPUT_DIR / f"{stem}_quantized.png")
    draw_chart(OUTPUT_DIR / f"{stem}_chart.png", swatches, palette)
    write_code_matrix(OUTPUT_DIR / f"{stem}_codes.txt", grid, palette)
    write_code_csv(OUTPUT_DIR / f"{stem}_codes.csv", grid, palette)
    write_usage_csv(OUTPUT_DIR / f"{stem}_usage.csv", grid, palette)

    total_colors = len({int(i) for i in grid.reshape(-1)})
    print(f"Done in {time.perf_counter() - started:.1f}s -> {OUTPUT_DIR / stem}.*")
    print(f"Colours used: {total_colors}, beads: {grid.size}")


if __name__ == "__main__":
    main()
