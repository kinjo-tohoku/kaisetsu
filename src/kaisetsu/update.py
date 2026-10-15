"""新しい版の確認(GitHub Releases)。GUI から使うが Qt には依存しない。

- `check_for_update()` が GitHub の Releases API で最新版を調べ、今の版より新しければ
  `ReleaseInfo` を返す。API のレート制限(認証なしは 1 時間 60 回 / IP)に引っかかったときは、
  `releases/latest` のリダイレクト先からタグだけを読む(変更点の本文は空になる)。
- 送る情報は無い(GitHub にアクセスするだけ)。プロキシは環境変数 HTTPS_PROXY を urllib が読む。
- `download_file()` はインストーラをファイルに落とす(進捗と中止のコールバック付き)。
- `install_kind()` で「インストーラで入れた」「zip 版」「Python 環境」を見分け、GUI が案内を変える。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path
from typing import Literal, cast

from kaisetsu import __version__

REPO = "kinjo-tohoku/kaisetsu"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"
INSTALLER_NAME = "kaisetsu-windows-setup.exe"
ZIP_NAME = "kaisetsu-windows.zip"
INSTALLER_URL = f"{RELEASES_PAGE}/download/{INSTALLER_NAME}"
ZIP_URL = f"{RELEASES_PAGE}/download/{ZIP_NAME}"
USER_AGENT = f"kaisetsu/{__version__}"

InstallKind = Literal["installer", "portable", "python"]


@dataclass(frozen=True)
class ReleaseInfo:
    """GitHub の最新 Release の要約。"""

    version: str
    """"0.1.3" のような版(先頭の v を除いたもの)"""
    tag: str
    """タグ("v0.1.3")"""
    url: str
    """Release ページの URL"""
    notes: str = ""
    """変更点(Markdown)。リダイレクトから版だけ分かったときは空"""
    published: str = ""
    """公開日時(ISO 8601)"""
    installer_url: str = INSTALLER_URL
    installer_size: int = 0
    zip_url: str = ZIP_URL


_VERSION_RE = re.compile(r"^\s*v?(\d+(?:\.\d+)*)(.*?)\s*$")


def parse_version(text: str) -> tuple[tuple[int, ...], int, str]:
    """版の文字列を比較できる組にする。

    "v0.1.3" → ((0, 1, 3), 1, "")。"0.2.0rc1" のような接尾辞付きは同じ数字の正式版より小さい
    (2 番目の要素が 0)。読めない文字列は ((0,), 1, text)。
    """
    m = _VERSION_RE.match(text)
    if not m:
        return ((0,), 1, text)
    nums = tuple(int(x) for x in m.group(1).split("."))
    suffix = m.group(2).strip(" .-_")
    return (nums, 0 if suffix else 1, suffix)


def _padded(nums: tuple[int, ...], n: int) -> tuple[int, ...]:
    return nums + (0,) * (n - len(nums))


def is_newer(candidate: str, current: str) -> bool:
    """candidate が current より新しい版か("0.1.10" > "0.1.9"、"0.2.0rc1" < "0.2.0")。"""
    a_nums, a_final, a_suffix = parse_version(candidate)
    b_nums, b_final, b_suffix = parse_version(current)
    n = max(len(a_nums), len(b_nums))
    return (_padded(a_nums, n), a_final, a_suffix) > (_padded(b_nums, n), b_final, b_suffix)


def parse_release(data: dict[str, object]) -> ReleaseInfo:
    """Releases API の JSON(1 件分)を `ReleaseInfo` にする。"""
    tag = str(data.get("tag_name") or "")
    version = tag[1:] if tag.startswith("v") else tag
    installer_url = INSTALLER_URL
    installer_size = 0
    zip_url = ZIP_URL
    assets = data.get("assets")
    if isinstance(assets, list):
        for a in assets:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or "")
            if name == INSTALLER_NAME:
                installer_url = str(a.get("browser_download_url") or INSTALLER_URL)
                installer_size = int(a.get("size") or 0)
            elif name == ZIP_NAME:
                zip_url = str(a.get("browser_download_url") or ZIP_URL)
    return ReleaseInfo(
        version=version,
        tag=tag,
        url=str(data.get("html_url") or RELEASES_PAGE),
        notes=str(data.get("body") or ""),
        published=str(data.get("published_at") or ""),
        installer_url=installer_url,
        installer_size=installer_size,
        zip_url=zip_url,
    )


def _open(url: str, timeout: float, *, accept: str | None = None) -> HTTPResponse:
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    return cast(HTTPResponse, urllib.request.urlopen(req, timeout=timeout))


def fetch_latest(timeout: float = 10.0) -> ReleaseInfo:
    """GitHub の最新 Release を取る。API が使えなければリダイレクト先のタグだけを読む。"""
    try:
        with _open(RELEASES_API, timeout, accept="application/vnd.github+json") as r:
            data = json.load(r)
        if isinstance(data, dict) and data.get("tag_name"):
            return parse_release(data)
    except (urllib.error.HTTPError, json.JSONDecodeError, ValueError):
        pass  # レート制限(403)など。下でタグだけ読む
    with _open(RELEASES_PAGE, timeout) as r:
        final = str(r.geturl())
    m = re.search(r"/releases/tag/([^/?#]+)", final)
    if not m:
        raise RuntimeError(f"最新版のタグが分かりませんでした: {final}")
    tag = m.group(1)
    return ReleaseInfo(
        version=tag[1:] if tag.startswith("v") else tag,
        tag=tag,
        url=final,
    )


def check_for_update(
    current: str = __version__,
    timeout: float = 10.0,
    fetch: Callable[[], ReleaseInfo] | None = None,
) -> ReleaseInfo | None:
    """今の版より新しい Release があればその情報、無ければ None。

    ネットワークの失敗は例外のまま返す(呼び出し側が「確認できませんでした」と出す)。
    """
    info = fetch() if fetch is not None else fetch_latest(timeout)
    return info if is_newer(info.version, current) else None


class DownloadCancelled(Exception):
    """利用者がダウンロードを中止した。"""


def download_file(
    url: str,
    dest: Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
    timeout: float = 30.0,
    chunk: int = 1 << 16,
) -> Path:
    """URL をファイルに落とす。progress(受信バイト, 総バイト(不明なら 0))、cancel() が真なら中止。

    途中のファイルは `dest` に ".part" を付けた名前に書き、終わってから改名する。
    """
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _open(url, timeout) as r, part.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                if cancel is not None and cancel():
                    raise DownloadCancelled()
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress is not None:
                    progress(done, total)
        if total and done != total:
            raise OSError(f"ダウンロードが途中で切れました({done} / {total} バイト)")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    if dest.exists():
        dest.unlink()
    part.rename(dest)
    return dest


def install_kind() -> InstallKind:
    """どう入っているか: "installer"(Inno Setup)、"portable"(zip 版)、"python"(uv / pip)。"""
    if not getattr(sys, "frozen", False):
        return "python"
    exe_dir = Path(sys.executable).resolve().parent
    if (exe_dir / "unins000.exe").exists():
        return "installer"
    return "portable"


def launch_installer(path: Path) -> None:
    """ダウンロードしたインストーラを別プロセスで起動する(呼び出し側がアプリを終了する)。"""
    if os.name != "nt":
        raise OSError("インストーラは Windows 用です")
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    subprocess.Popen([str(path)], close_fds=True, creationflags=flags)


__all__ = [
    "INSTALLER_URL",
    "RELEASES_PAGE",
    "ZIP_URL",
    "DownloadCancelled",
    "InstallKind",
    "ReleaseInfo",
    "check_for_update",
    "download_file",
    "fetch_latest",
    "install_kind",
    "is_newer",
    "launch_installer",
    "parse_release",
    "parse_version",
]
