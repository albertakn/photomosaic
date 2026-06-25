from __future__ import annotations

import re
import sys
import time

import instaloader
from instaloader.exceptions import (
    ConnectionException,
    InstaloaderException,
    LoginRequiredException,
    ProfileNotExistsException,
    TooManyRequestsException,
)

TILES_DIR = "tiles"


def parse_username(user: str) -> str:
    """Достаёт username из ссылки вида instagram.com/<name>/ или из @name."""
    user = user.strip()
    m = re.search(r"instagram\.com/([^/?#]+)", user)
    if m:
        return m.group(1)
    return user.lstrip("@")


def scrape(user: str, target_dir: str = TILES_DIR, delay: float = 2.0) -> int:
    """Качает все фото-посты аккаунта в target_dir. Возвращает код выхода."""
    username = parse_username(user)
    print(f"Аккаунт: {username}  →  папка: {target_dir}/")

    loader = instaloader.Instaloader(
        dirname_pattern=target_dir,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        post_metadata_txt_pattern="",
        # сохраняем картинки сразу в target_dir, без подпапки на каждый профиль
        filename_pattern="{profile}_{date_utc}_{shortcode}",
    )

    try:
        profile = instaloader.Profile.from_username(loader.context, username)
    except ProfileNotExistsException:
        print(f"Аккаунт '{username}' не найден.", file=sys.stderr)
        return 1
    except (ConnectionException, TooManyRequestsException, LoginRequiredException) as e:
        print(_limit_message(e), file=sys.stderr)
        return 2

    if profile.is_private:
        print(
            f"Аккаунт '{username}' приватный — без логина скачать нельзя.",
            file=sys.stderr,
        )
        return 1

    total = profile.mediacount
    print(f"Найдено постов: {total}. Качаю только изображения...\n")

    done = 0
    try:
        for post in profile.get_posts():
            # download_videos=False: видео не качаются, у альбомов берутся
            # только картинки. download_post сам пропускает уже скачанное.
            loader.download_post(post, target=target_dir)
            done += 1
            print(f"  [{done}/{total}] {post.shortcode}")
            time.sleep(delay)
    except (ConnectionException, TooManyRequestsException, LoginRequiredException) as e:
        print("\n" + _limit_message(e), file=sys.stderr)
        print(f"Скачано до бана: {done}. Запусти снова — докачает остальное.",
              file=sys.stderr)
        return 2
    except InstaloaderException as e:
        print(f"\nОшибка instaloader: {e}", file=sys.stderr)
        return 2

    print(f"\nГотово. Скачано постов: {done}. Картинки в '{target_dir}/'.")
    return 0


def _limit_message(err: Exception) -> str:
    return (
        "Instagram ограничил анонимный доступ "
        f"({type(err).__name__}: {err}).\n"
        "Подожди несколько минут и запусти снова — уже скачанное сохранится "
        "(fast_update)."
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--user", required=True, help="ссылка на профиль или username")
    p.add_argument("--out", default=TILES_DIR, help="папка для фоток")
    p.add_argument("--delay", type=float, default=2.0,
                   help="пауза между постами, сек")
    args = p.parse_args()
    sys.exit(scrape(args.user, args.out, args.delay))
