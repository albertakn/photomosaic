import argparse
import sys

import mosaic
import scrape

_HELP = (
    "Фотомозаика из фоток Instagram.\n\n"
    "  uv run main.py scrape --user <ссылка-или-username>\n"
    "  uv run main.py build  --target face.jpg --tiles tiles --out mosaic.png"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="photomosaic", description=_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scrape", help="скачать фотки из публичного Instagram")
    s.add_argument("--user", required=True, help="ссылка на профиль или username")
    s.add_argument("--out", default=scrape.TILES_DIR, help="папка для фоток")
    s.add_argument("--delay", type=float, default=2.0, help="пауза между постами, сек")

    b = sub.add_parser("build", help="собрать мозаику из плиток")
    b.add_argument("--target", required=True, help="фото лица")
    b.add_argument("--tiles", default="tiles", help="папка с плитками")
    b.add_argument("--out", default="mosaic.png", help="выходной файл")
    b.add_argument("--cols", type=int, default=80, help="ячеек по ширине")
    b.add_argument("--tile-px", type=int, default=50, help="размер плитки, px")
    b.add_argument("--overlay", type=float, default=0.15,
                   help="подмес цвета 0..1 (меньше = сильнее видно фотки)")
    b.add_argument("--no-repeat", type=int, default=3,
                   help="радиус (в рядах) запрета повтора плитки")

    a = parser.parse_args()
    if a.cmd == "scrape":
        return scrape.scrape(a.user, a.out, a.delay)
    return mosaic.build(a.target, a.tiles, a.out, a.cols, a.tile_px,
                        a.overlay, a.no_repeat)


if __name__ == "__main__":
    sys.exit(main())
