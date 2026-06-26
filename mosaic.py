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


def _fingerprint(img: Image.Image, grid: int = GRID) -> np.ndarray:
    """grid×grid Lab-отпечаток квадратного изображения -> вектор длины grid²·3."""
    img = img.convert("RGB").resize((grid * 8, grid * 8))
    arr = np.asarray(img).reshape(grid, 8, grid, 8, 3)
    # средний цвет каждой из grid*grid под-ячеек
    cells = arr.mean(axis=(1, 3))            # (grid, grid, 3) в RGB
    lab = _srgb_to_lab(cells.reshape(-1, 3))  # (grid*grid, 3)
    return lab.reshape(-1)                     # grid²·3


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


def load_tile_fingerprints(tiles_dir: str, grid: int = GRID, cache: str = CACHE_FILE):
    """Возвращает (paths, fingerprints[N, grid²·3]). Кэширует в .npz.

    Кэш привязан к grid: при другом grid отпечатки пересчитываются.
    """
    paths = _list_tiles(tiles_dir)
    if not paths:
        raise SystemExit(f"В '{tiles_dir}/' нет картинок ({', '.join(IMG_EXT)}).")

    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        if int(data.get("grid", -1)) == grid and list(data["paths"]) == paths:
            print(f"Отпечатки плиток из кэша: {len(paths)} шт. (grid={grid})")
            return paths, data["fp"]

    print(f"Считаю отпечатки {len(paths)} плиток (grid={grid})...")
    fps = np.empty((len(paths), grid * grid * 3), dtype=np.float32)
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                fps[i] = _fingerprint(_center_square(im), grid)
        except Exception as e:  # битый файл — заполняем «недостижимым» отпечатком
            print(f"  пропуск {p}: {e}", file=sys.stderr)
            fps[i] = 1e6
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)}")
    np.savez(cache, paths=np.array(paths), fp=fps, grid=grid)
    return paths, fps


# ---------- сборка ----------

REPEAT_MODES = ("none", "linear", "square", "circle", "cap")
BLEND_MODES = ("color", "original")


def build(target_path: str, tiles_dir: str, out_path: str,
          cols: int = 80, tile_px: int = 50, overlay: float = 0.15,
          no_repeat: int = 3, repeat_mode: str = "square",
          blend: str = "color", best_k: int = 1, grid: int = GRID) -> int:
    if repeat_mode not in REPEAT_MODES:
        raise SystemExit(f"repeat_mode должен быть одним из {REPEAT_MODES}")
    if blend not in BLEND_MODES:
        raise SystemExit(f"blend должен быть одним из {BLEND_MODES}")
    if grid < 1:
        raise SystemExit("grid должен быть >= 1")
    paths, tile_fps = load_tile_fingerprints(tiles_dir, grid)

    target = Image.open(target_path).convert("RGB")
    tw, th = target.size
    rows = max(1, round(cols * th / tw))
    print(f"Сетка: {cols} x {rows} = {cols * rows} ячеек; плитка {tile_px}px")

    # Отпечаток каждой ячейки: ресайз до cols*grid x rows*grid, тогда блок
    # grid x grid пикселей = ровно grid×grid под-цветов ячейки.
    fp_small = np.asarray(target.resize((cols * grid, rows * grid)))
    cell_lab = _srgb_to_lab(fp_small.reshape(-1, 3)).reshape(
        rows, grid, cols, grid, 3)
    cell_lab = cell_lab.transpose(0, 2, 1, 3, 4).reshape(
        rows, cols, grid * grid * 3).astype(np.float32)  # (rows, cols, grid²·3)
    # Средний цвет ячейки для overlay.
    avg_rgb = np.asarray(target.resize((cols, rows)))  # (rows, cols, 3)

    print(f"Отпечаток: {grid}×{grid} ({grid * grid * 3} чисел). "
          f"Анти-повтор: '{repeat_mode}', n={no_repeat}, best_k={best_k}; "
          f"блендинг '{blend}'")

    canvas = Image.new("RGB", (cols * tile_px, rows * tile_px))
    grid = np.full((rows, cols), -1, dtype=np.int32)  # какая плитка в каждой ячейке
    usage: Counter[int] = Counter()  # tile index -> сколько раз вставлена
    last_used: dict[int, int] = {}  # tile index -> линейный номер последней ячейки

    def conflict(cand: int, r: int, c: int) -> bool:
        """True, если плитку cand нельзя ставить в (r, c) по текущему режиму."""
        if no_repeat <= 1 or repeat_mode == "none":
            return False
        if repeat_mode == "cap":
            # глобальный лимит: фотка использована не больше no_repeat раз
            return usage[cand] >= no_repeat
        if repeat_mode == "linear":
            # линейное окно: не повторять в пределах no_repeat рядов по ходу обхода
            last = last_used.get(cand)
            return last is not None and (r * cols + c - last) <= no_repeat * cols
        # геометрические режимы: ищем такую же плитку рядом среди заполненных
        rad = no_repeat - 1
        r0 = max(0, r - rad)
        c0, c1 = max(0, c - rad), min(cols - 1, c + rad)
        for rr in range(r0, r + 1):
            for cc in range(c0, c1 + 1):
                if rr == r and cc >= c:
                    break
                if grid[rr, cc] != cand:
                    continue
                if repeat_mode == "square":  # квадрат n x n (Чебышёв)
                    return True
                # circle: запрет только внутри круга радиуса no_repeat (евклид)
                if (rr - r) ** 2 + (cc - c) ** 2 < no_repeat ** 2:
                    return True
        return False

    tile_cache: dict[int, Image.Image] = {}

    def get_tile(idx: int) -> Image.Image:
        if idx not in tile_cache:
            with Image.open(paths[idx]) as im:
                tile_cache[idx] = _center_square(im).convert("RGB").resize(
                    (tile_px, tile_px))
        return tile_cache[idx]

    k = max(1, best_k)
    tint_per_cell = overlay > 0 and blend == "color"
    for r in range(rows):
        for c in range(cols):
            cell_fp = cell_lab[r, c]                       # (27,)
            d = np.sum((tile_fps - cell_fp) ** 2, axis=1)  # до всех плиток
            order = np.argsort(d)
            # собираем до k ближайших допустимых (прошедших ограничение)
            # кандидатов; если все конфликтуют — фолбэк на наименее used фотку
            cands: list[int] = []
            fb = int(order[0])
            for cand in order:
                cand = int(cand)
                if usage[cand] < usage[fb]:
                    fb = cand
                if not conflict(cand, r, c):
                    cands.append(cand)
                    if len(cands) >= k:
                        break
            if not cands:
                idx = fb
            elif len(cands) == 1:
                idx = cands[0]
            else:
                # best-k: случайная из допустимых, вес ~ 1/расстояние
                w = 1.0 / (d[cands] + 1e-6)
                idx = int(np.random.choice(cands, p=w / w.sum()))
            grid[r, c] = idx
            usage[idx] += 1
            last_used[idx] = r * cols + c

            tile = get_tile(idx)
            if tint_per_cell:
                tint = Image.new("RGB", tile.size,
                                 tuple(int(x) for x in avg_rgb[r, c]))
                tile = Image.blend(tile, tint, overlay)
            canvas.paste(tile, (c * tile_px, r * tile_px))
        print(f"  ряд {r + 1}/{rows}")

    if blend == "original" and overlay > 0:
        # блендим реальный оригинал поверх всей мозаики (как worldveil)
        orig = target.resize(canvas.size)
        canvas = Image.blend(canvas, orig, overlay)

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
                   help="параметр n анти-повтора (linear/square/circle — радиус, cap — макс. повторов)")
    p.add_argument("--repeat-mode", choices=REPEAT_MODES, default="square",
                   help="алгоритм анти-повтора: none | linear | square | circle | cap")
    p.add_argument("--best-k", type=int, default=1,
                   help="стохастика поверх: случайная из k ближайших допустимых (1 = выкл)")
    p.add_argument("--blend", choices=BLEND_MODES, default="color",
                   help="overlay: color (цвет ячейки) | original (реальный оригинал)")
    p.add_argument("--grid", type=int, default=GRID,
                   help="под-сетка отпечатка grid×grid (больше = точнее структура/края)")
    a = p.parse_args()
    sys.exit(build(a.target, a.tiles, a.out, a.cols, a.tile_px,
                   a.overlay, a.no_repeat, a.repeat_mode, a.blend, a.best_k, a.grid))
