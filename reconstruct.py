import argparse
import os
import sys

import numpy as np
from PIL import Image

from mosaic import _center_square, _list_tiles

D = 16  # разрешение дескриптора для сопоставления ячейка↔плитка


def _descriptor(img: Image.Image) -> np.ndarray:
    """Центр-квадрат → D×D RGB → вектор. Сравниваем ячейки с плитками этим."""
    sq = _center_square(img).convert("RGB").resize((D, D))
    return np.asarray(sq, dtype=np.float32).reshape(-1)


def _load_tiles(tiles_dirs: list[str]):
    paths = []
    for d in tiles_dirs:
        paths += _list_tiles(d)
    if not paths:
        raise SystemExit("Не найдено плиток в указанных папках.")
    desc = np.empty((len(paths), D * D * 3), dtype=np.float32)
    tiles = []
    for i, p in enumerate(paths):
        im = Image.open(p)
        desc[i] = _descriptor(im)
        tiles.append(_center_square(im).convert("RGB"))
    return paths, desc, tiles


def _assign(mosaic_arr: np.ndarray, cols: int, rows: int, tile_desc: np.ndarray):
    """Для сетки cols×rows вернуть (индексы плиток [rows,cols], средн. расстояние)."""
    H, W = mosaic_arr.shape[:2]
    cell_desc = np.empty((rows * cols, D * D * 3), dtype=np.float32)
    for r in range(rows):
        y0, y1 = r * H // rows, (r + 1) * H // rows
        for c in range(cols):
            x0, x1 = c * W // cols, (c + 1) * W // cols
            cell = Image.fromarray(mosaic_arr[y0:y1, x0:x1]).resize((D, D))
            cell_desc[r * cols + c] = np.asarray(cell, np.float32).reshape(-1)
    # матрица расстояний (Ncells × Ntiles) через (a-b)² = a² + b² - 2ab
    a2 = (cell_desc ** 2).sum(1)[:, None]
    b2 = (tile_desc ** 2).sum(1)[None, :]
    d2 = a2 + b2 - 2 * cell_desc @ tile_desc.T
    idx = d2.argmin(1)
    score = float(np.sqrt(np.maximum(d2[np.arange(len(idx)), idx], 0)).mean())
    return idx.reshape(rows, cols), score


def reconstruct(mosaic_path: str, tiles_dirs: list[str], out_path: str,
                cols: int = 0, tile_px: int = 60, overlay: float = 0.0,
                color_src: str = "", cols_min: int = 30, cols_max: int = 100) -> int:
    paths, tile_desc, tiles = _load_tiles(tiles_dirs)
    print(f"Плиток в библиотеке: {len(paths)}")

    mosaic_img = Image.open(mosaic_path).convert("RGB")
    arr = np.asarray(mosaic_img)
    H, W = arr.shape[:2]
    print(f"Мозаика: {W}×{H}px")

    if cols > 0:
        candidates = [cols]
    else:
        candidates = list(range(cols_min, cols_max + 1))
        print(f"Определяю cols перебором {cols_min}..{cols_max}...")

    best = None
    for cc in candidates:
        rr = max(1, round(cc * H / W))  # ячейки квадратные → rows из аспекта
        grid, score = _assign(arr, cc, rr, tile_desc)
        if best is None or score < best[2]:
            best = (cc, rr, score, grid)
        if cols == 0 and cc % 10 == 0:
            print(f"  cols={cc}: score={score:.1f}")
    cc, rr, score, grid = best
    print(f"Выбрано: cols={cc}, rows={rr} (score={score:.1f})")

    # тон ячеек для overlay: из оригинального лица (если задано), иначе из мозаики
    if overlay > 0:
        src = Image.open(color_src).convert("RGB") if color_src else mosaic_img
        cell_rgb = np.asarray(src.resize((cc, rr)))  # (rr, cc, 3)
        print(f"Overlay {overlay}: тон из "
              f"{'лица ' + color_src if color_src else 'самой мозаики'}")

    canvas = Image.new("RGB", (cc * tile_px, rr * tile_px))
    for r in range(rr):
        for c in range(cc):
            t = tiles[grid[r, c]].resize((tile_px, tile_px))
            if overlay > 0:
                tint = Image.new("RGB", t.size,
                                 tuple(int(x) for x in cell_rgb[r, c]))
                t = Image.blend(t, tint, overlay)
            canvas.paste(t, (c * tile_px, r * tile_px))
    canvas.save(out_path)
    print(f"Готово: {out_path} ({canvas.size[0]}×{canvas.size[1]}px)")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Восстановить мозаику в HQ по готовой (сжатой) картинке")
    p.add_argument("--mosaic", required=True, help="потерянная (сжатая) мозаика")
    p.add_argument("--tiles", required=True,
                   help="папки с плитками через запятую, напр. data/stickers,data/photo")
    p.add_argument("--out", default="reconstructed.png", help="выходной файл")
    p.add_argument("--cols", type=int, default=0, help="число столбцов (0 = авто)")
    p.add_argument("--tile-px", type=int, default=60, help="размер плитки, px")
    p.add_argument("--overlay", type=float, default=0.0,
                   help="подмес тона ячейки 0..1 (0 = чистые плитки)")
    p.add_argument("--color-src", default="",
                   help="откуда брать тон для overlay (оригинал лица); иначе из мозаики")
    a = p.parse_args()
    dirs = [d.strip() for d in a.tiles.split(",") if d.strip()]
    sys.exit(reconstruct(a.mosaic, dirs, a.out, a.cols, a.tile_px,
                         a.overlay, a.color_src))
