from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

CACHE_FILE = "tiles_cache.npz"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")
GRID = 3  # под-сетка отпечатка: GRID x GRID под-ячеек


# ---------- цвет ----------

def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB [0..255] -> CIE Lab. rgb формы (..., 3). Возвращает (..., 3)."""
    rgb = rgb.astype(np.float64) / 255.0
    # sRGB -> линейный
    mask = rgb > 0.04045
    rgb = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    # линейный RGB -> XYZ (D65)
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb @ m.T
    # нормировка по белой точке D65
    white = np.array([0.95047, 1.0, 1.08883])
    xyz = xyz / white
    eps = 0.008856
    kappa = 903.3
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _fingerprint(img: Image.Image) -> np.ndarray:
    """3x3 Lab-отпечаток квадратного изображения -> вектор длины 27."""
    img = img.convert("RGB").resize((GRID * 8, GRID * 8))
    arr = np.asarray(img).reshape(GRID, 8, GRID, 8, 3)
    # средний цвет каждой из GRID*GRID под-ячеек
    cells = arr.mean(axis=(1, 3))            # (GRID, GRID, 3) в RGB
    lab = _srgb_to_lab(cells.reshape(-1, 3))  # (GRID*GRID, 3)
    return lab.reshape(-1)                     # 27


def _center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s))


# ---------- плитки ----------

def _list_tiles(tiles_dir: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(tiles_dir):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
                paths.append(os.path.join(root, f))
    return paths


def load_tile_fingerprints(tiles_dir: str, cache: str = CACHE_FILE):
    """Возвращает (paths, fingerprints[N,27]). Кэширует в .npz."""
    paths = _list_tiles(tiles_dir)
    if not paths:
        raise SystemExit(f"В '{tiles_dir}/' нет картинок ({', '.join(IMG_EXT)}).")

    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        if list(data["paths"]) == paths:
            print(f"Отпечатки плиток из кэша: {len(paths)} шт.")
            return paths, data["fp"]

    print(f"Считаю отпечатки {len(paths)} плиток...")
    fps = np.empty((len(paths), GRID * GRID * 3), dtype=np.float32)
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                fps[i] = _fingerprint(_center_square(im))
        except Exception as e:  # битый файл — заполняем «недостижимым» отпечатком
            print(f"  пропуск {p}: {e}", file=sys.stderr)
            fps[i] = 1e6
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)}")
    np.savez(cache, paths=np.array(paths), fp=fps)
    return paths, fps


# ---------- сборка ----------

def build(target_path: str, tiles_dir: str, out_path: str,
          cols: int = 80, tile_px: int = 50, overlay: float = 0.15,
          no_repeat: int = 3) -> int:
    paths, tile_fps = load_tile_fingerprints(tiles_dir)

    target = Image.open(target_path).convert("RGB")
    tw, th = target.size
    rows = max(1, round(cols * th / tw))
    print(f"Сетка: {cols} x {rows} = {cols * rows} ячеек; плитка {tile_px}px")

    # Отпечаток каждой ячейки: ресайз до cols*GRID x rows*GRID, тогда блок
    # GRID x GRID пикселей = ровно 3x3 под-цвета ячейки.
    fp_small = np.asarray(target.resize((cols * GRID, rows * GRID)))  # (r*3,c*3,3)
    cell_lab = _srgb_to_lab(fp_small.reshape(-1, 3)).reshape(
        rows, GRID, cols, GRID, 3)
    cell_lab = cell_lab.transpose(0, 2, 1, 3, 4).reshape(
        rows, cols, GRID * GRID * 3).astype(np.float32)  # (rows, cols, 27)
    # Средний цвет ячейки для overlay.
    avg_rgb = np.asarray(target.resize((cols, rows)))  # (rows, cols, 3)

    canvas = Image.new("RGB", (cols * tile_px, rows * tile_px))
    used: dict[int, int] = {}  # tile index -> номер последней ячейки
    usage: Counter[int] = Counter()  # tile index -> сколько раз вставлена

    tile_cache: dict[int, Image.Image] = {}

    def get_tile(idx: int) -> Image.Image:
        if idx not in tile_cache:
            with Image.open(paths[idx]) as im:
                tile_cache[idx] = _center_square(im).convert("RGB").resize(
                    (tile_px, tile_px))
        return tile_cache[idx]

    for r in range(rows):
        for c in range(cols):
            cell_fp = cell_lab[r, c]                       # (27,)
            d = np.sum((tile_fps - cell_fp) ** 2, axis=1)  # до всех плиток
            order = np.argsort(d)
            cur = r * cols + c
            idx = int(order[0])
            # антиповтор: не брать плитку, использованную в радиусе no_repeat рядов
            for cand in order:
                cand = int(cand)
                last = used.get(cand)
                if last is None or (cur - last) > no_repeat * cols:
                    idx = cand
                    break
            used[idx] = cur
            usage[idx] += 1

            tile = get_tile(idx)
            if overlay > 0:
                tint = Image.new("RGB", tile.size,
                                 tuple(int(x) for x in avg_rgb[r, c]))
                tile = Image.blend(tile, tint, overlay)
            canvas.paste(tile, (c * tile_px, r * tile_px))
        print(f"  ряд {r + 1}/{rows}")

    canvas.save(out_path)
    print(f"\nГотово: {out_path}  ({canvas.size[0]}x{canvas.size[1]}px)")
    _print_stats(cols, rows, len(paths), usage)
    return 0


def _print_stats(cols: int, rows: int, n_tiles: int, usage: Counter[int]) -> None:
    cells = cols * rows
    distinct = len(usage)
    avg = cells / distinct if distinct else 0
    mx = max(usage.values()) if usage else 0
    headers = ["cols", "сетка", "ячеек", "использовано фото",
               "ср. повторов", "макс. повтор 1 фото"]
    values = [str(cols), f"{cols}×{rows}", str(cells),
              f"{distinct} / {n_tiles}", f"{avg:.1f}×", str(mx)]
    widths = [max(len(h), len(v)) for h, v in zip(headers, values)]
    line = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
    row = "  ".join(v.rjust(w) for v, w in zip(values, widths))
    print()
    print(line)
    print(row)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, help="фото лица")
    p.add_argument("--tiles", default="tiles", help="папка с плитками")
    p.add_argument("--out", default="mosaic.png", help="выходной файл")
    p.add_argument("--cols", type=int, default=80, help="ячеек по ширине")
    p.add_argument("--tile-px", type=int, default=50, help="размер плитки, px")
    p.add_argument("--overlay", type=float, default=0.15,
                   help="подмес цвета 0..1 (меньше = сильнее видно фотки)")
    p.add_argument("--no-repeat", type=int, default=3,
                   help="радиус (в рядах) запрета повтора плитки")
    a = p.parse_args()
    sys.exit(build(a.target, a.tiles, a.out, a.cols, a.tile_px,
                   a.overlay, a.no_repeat))
