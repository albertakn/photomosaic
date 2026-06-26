import argparse
import io
import os
import re
import sys

import requests
from PIL import Image

API = "https://api.telegram.org"
DEFAULT_PACK = "ranawayfromtheconvent"


def parse_name(name: str) -> str:
    """Достаёт имя пака из ссылки t.me/addstickers/<name> или из голого имени."""
    m = re.search(r"addstickers/([^/?#]+)", name)
    return m.group(1) if m else name.strip().lstrip("@")


def download(name: str, token: str, out_dir: str, bg=(255, 255, 255)) -> int:
    """Качает статичные стикеры пака и кладёт их .png (на фоне bg) в out_dir."""
    name = parse_name(name)
    os.makedirs(out_dir, exist_ok=True)

    r = requests.get(f"{API}/bot{token}/getStickerSet", params={"name": name})
    data = r.json()
    if not data.get("ok"):
        print(f"Ошибка getStickerSet: {data.get('description', data)}", file=sys.stderr)
        return 1

    stickers = data["result"]["stickers"]
    print(f"Пак '{name}': стикеров всего {len(stickers)}")

    saved = skipped = 0
    for i, st in enumerate(stickers):
        if st.get("is_animated") or st.get("is_video"):
            skipped += 1  # .tgs / .webm — не картинка, пропускаем
            continue
        try:
            fp = _get_file_path(st["file_id"], token)
            raw = requests.get(f"{API}/file/bot{token}/{fp}").content
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
            canvas = Image.new("RGB", img.size, bg)
            canvas.paste(img, mask=img.split()[3])  # альфа как маска
            canvas.save(os.path.join(out_dir, f"{name}_{i:03}.png"))
            saved += 1
        except Exception as e:
            print(f"  пропуск {i}: {e}", file=sys.stderr)
            skipped += 1

    print(f"Готово: сохранено {saved}, пропущено {skipped} (анимированные/ошибки).")
    print(f"Папка: {out_dir}/")
    return 0


def _get_file_path(file_id: str, token: str) -> str:
    r = requests.get(f"{API}/bot{token}/getFile", params={"file_id": file_id})
    return r.json()["result"]["file_path"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Скачать статичные Telegram-стикеры как png")
    p.add_argument("--name", default=DEFAULT_PACK,
                   help="ссылка t.me/addstickers/<name> или имя пака")
    p.add_argument("--token", default=os.environ.get("TG_BOT_TOKEN"),
                   help="токен бота от @BotFather (или env TG_BOT_TOKEN)")
    p.add_argument("--out", default="data/stickers", help="папка для png")
    a = p.parse_args()
    if not a.token:
        sys.exit("Нужен токен бота: --token <...> или переменная TG_BOT_TOKEN")
    sys.exit(download(a.name, a.token, a.out))
