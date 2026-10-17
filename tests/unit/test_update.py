"""update: 版の比較、Release JSON の読み取り、ダウンロード、入れ方の判定(ネットワーク無し)。"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import urllib.error
from pathlib import Path

import pytest

from kaisetsu import update
from kaisetsu.update import (
    INSTALLER_URL,
    RELEASES_API,
    RELEASES_PAGE,
    ZIP_URL,
    DownloadCancelled,
    DownloadIntegrityError,
    ReleaseInfo,
    check_for_update,
    download_file,
    download_installer,
    fetch_latest,
    install_kind,
    is_newer,
    parse_digest,
    parse_release,
    parse_version,
)


@pytest.mark.parametrize(
    ("candidate", "current", "newer"),
    [
        ("v0.1.3", "0.1.2", True),
        ("0.1.3", "0.1.3", False),
        ("0.1.2", "0.1.3", False),
        ("0.1.10", "0.1.9", True),
        ("0.2.0", "0.1.10", True),
        ("0.2", "0.1.9", True),
        ("1.0", "1.0.0", False),
        ("1.0.0", "1.0", False),
        ("0.2.0rc1", "0.1.3", True),
        ("0.2.0rc1", "0.2.0", False),
        ("0.2.0", "0.2.0rc1", True),
        ("0.2.0-beta", "0.2.0", False),
        ("abc", "0.1.0", False),
    ],
)
def test_is_newer(candidate: str, current: str, newer: bool) -> None:
    assert is_newer(candidate, current) is newer


def test_parse_version() -> None:
    assert parse_version("v0.1.3") == ((0, 1, 3), 1, "")
    assert parse_version("0.2.0rc1") == ((0, 2, 0), 0, "rc1")
    assert parse_version("???") == ((0,), 1, "???")


def test_parse_release_reads_assets() -> None:
    data = {
        "tag_name": "v0.1.3",
        "html_url": "https://github.com/kinjo-tohoku/kaisetsu/releases/tag/v0.1.3",
        "body": "## kaisetsu 0.1.3\n\n- test",
        "published_at": "2026-10-15T08:00:00Z",
        "assets": [
            {
                "name": "kaisetsu-0.1.3-windows-setup.exe",
                "browser_download_url": "https://x/a",
                "size": 1,
            },
            {
                "name": "kaisetsu-windows-setup.exe",
                "browser_download_url": "https://x/kaisetsu-windows-setup.exe",
                "size": 120_000_000,
            },
            {
                "name": "kaisetsu-windows.zip",
                "browser_download_url": "https://x/kaisetsu-windows.zip",
                "size": 5,
            },
        ],
    }
    info = parse_release(data)
    assert info.version == "0.1.3" and info.tag == "v0.1.3"
    assert info.url.endswith("/tag/v0.1.3")
    assert info.installer_url == "https://x/kaisetsu-windows-setup.exe"
    assert info.installer_size == 120_000_000
    assert info.zip_url == "https://x/kaisetsu-windows.zip"
    assert "0.1.3" in info.notes and info.published.startswith("2026-10-15")


def test_parse_release_without_assets_uses_fixed_urls() -> None:
    info = parse_release({"tag_name": "v9.9.9"})
    assert info.version == "9.9.9"
    assert info.installer_url == INSTALLER_URL and info.zip_url == ZIP_URL
    assert info.url == RELEASES_PAGE and info.notes == "" and info.installer_size == 0


def test_check_for_update_with_injected_fetch() -> None:
    info = ReleaseInfo(version="0.1.4", tag="v0.1.4", url="u")
    assert check_for_update("0.1.3", fetch=lambda: info) is info
    assert check_for_update("0.1.4", fetch=lambda: info) is None
    assert check_for_update("0.2.0", fetch=lambda: info) is None


class _FakeResponse:
    """urlopen の戻り値のまね(with 文、read(n)、headers、geturl)。"""

    def __init__(self, payload: bytes, url: str, headers: dict[str, str] | None = None) -> None:
        self._buf = payload
        self._pos = 0
        self._url = url
        self.headers = headers or {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = len(self._buf) - self._pos
        out = self._buf[self._pos : self._pos + n]
        self._pos += len(out)
        return out

    def geturl(self) -> str:
        return self._url


def test_fetch_latest_uses_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_open(url: str, timeout: float, *, accept: str | None = None) -> _FakeResponse:
        assert url == RELEASES_API and accept
        return _FakeResponse(json.dumps({"tag_name": "v0.1.5", "body": "notes"}).encode(), url)

    monkeypatch.setattr(update, "_open", fake_open)
    info = fetch_latest()
    assert info.version == "0.1.5" and info.notes == "notes"


def test_fetch_latest_falls_back_to_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_open(url: str, timeout: float, *, accept: str | None = None) -> _FakeResponse:
        calls.append(url)
        if url == RELEASES_API:
            raise urllib.error.HTTPError(url, 403, "rate limited", {}, None)  # type: ignore[arg-type]
        return _FakeResponse(b"", "https://github.com/kinjo-tohoku/kaisetsu/releases/tag/v0.1.9")

    monkeypatch.setattr(update, "_open", fake_open)
    info = fetch_latest()
    assert info.version == "0.1.9" and info.tag == "v0.1.9" and info.notes == ""
    assert info.installer_url == INSTALLER_URL
    assert calls == [RELEASES_API, RELEASES_PAGE]


def test_download_file_streams_with_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = bytes(range(256)) * 1000
    monkeypatch.setattr(
        update,
        "_open",
        lambda url, timeout, accept=None: _FakeResponse(
            payload, url, {"Content-Length": str(len(payload))}
        ),
    )
    seen: list[tuple[int, int]] = []
    dest = tmp_path / "setup.exe"
    out = download_file("https://x", dest, progress=lambda d, t: seen.append((d, t)), chunk=100_000)
    assert out == dest and dest.read_bytes() == payload
    assert len(seen) == 3 and seen[-1] == (len(payload), len(payload))
    assert not dest.with_name("setup.exe.part").exists()
    # 上書き
    download_file("https://x", dest, chunk=100_000)
    assert dest.read_bytes() == payload


def test_download_file_cancel_and_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = bytes(range(256)) * 100
    monkeypatch.setattr(
        update,
        "_open",
        lambda url, timeout, accept=None: _FakeResponse(
            payload, url, {"Content-Length": str(len(payload))}
        ),
    )
    answers = iter([False, True])
    with pytest.raises(DownloadCancelled):
        download_file("https://x", tmp_path / "b.exe", cancel=lambda: next(answers), chunk=1000)
    assert not (tmp_path / "b.exe").exists() and not (tmp_path / "b.exe.part").exists()

    monkeypatch.setattr(
        update,
        "_open",
        lambda url, timeout, accept=None: _FakeResponse(
            payload[:1000], url, {"Content-Length": str(len(payload))}
        ),
    )
    with pytest.raises(OSError, match="途中"):
        download_file("https://x", tmp_path / "c.exe")
    assert not (tmp_path / "c.exe").exists() and not (tmp_path / "c.exe.part").exists()


def test_parse_release_reads_asset_digest() -> None:
    good = "sha256:" + "ab" * 32
    data = {
        "tag_name": "v0.1.14",
        "assets": [
            {"name": "kaisetsu-windows-setup.exe", "size": 10, "digest": good},
            {"name": "kaisetsu-windows.zip", "size": 5, "digest": "sha256:" + "cd" * 32},
        ],
    }
    assert parse_release(data).installer_sha256 == "ab" * 32
    data["assets"][0]["digest"] = "md5:" + "ab" * 16  # sha256 以外は使わない
    assert parse_release(data).installer_sha256 == ""
    data["assets"][0]["digest"] = None  # 古い API の応答(digest 無し)
    assert parse_release(data).installer_sha256 == ""
    assert parse_digest("SHA256:" + "AB" * 32) == "ab" * 32
    assert parse_digest("sha256:short") == "" and parse_digest(None) == ""


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(
        update,
        "_open",
        lambda url, timeout, accept=None: _FakeResponse(
            payload, url, {"Content-Length": str(len(payload))}
        ),
    )


def test_download_file_verifies_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = bytes(range(256)) * 300
    _serve(monkeypatch, payload)
    good = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "setup.exe"
    assert download_file("https://x", dest, sha256=good.upper(), chunk=7000) == dest
    assert dest.read_bytes() == payload

    bad = "0" * 64
    with pytest.raises(DownloadIntegrityError, match="SHA-256"):
        download_file("https://x", tmp_path / "bad.exe", sha256=bad, expected_size=len(payload))
    assert not (tmp_path / "bad.exe").exists() and not (tmp_path / "bad.exe.part").exists()
    # digest が合っても大きさの記載と違えば失敗(API の応答が壊れている)
    with pytest.raises(DownloadIntegrityError, match="大きさ"):
        download_file("https://x", tmp_path / "s.exe", sha256=good, expected_size=len(payload) + 1)
    assert not (tmp_path / "s.exe").exists()
    assert issubclass(DownloadIntegrityError, OSError)  # 既存のダイアログは Exception を表示する


def test_download_file_without_digest_checks_size_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    payload = b"MZ" + bytes(4000)
    _serve(monkeypatch, payload)
    with caplog.at_level(logging.INFO, logger="kaisetsu.update"):
        download_file("https://x", tmp_path / "a.exe", expected_size=len(payload))
    assert "ハッシュ未確認" in caplog.text and "大きさ" in caplog.text
    with pytest.raises(DownloadIntegrityError, match="大きさ"):
        download_file("https://x", tmp_path / "b.exe", expected_size=len(payload) - 1)
    assert not (tmp_path / "b.exe").exists()
    # レート制限の経路(大きさ 0、digest 無し)でも「確認済み」にはしない
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="kaisetsu.update"):
        download_file("https://x", tmp_path / "c.exe")
    assert "ハッシュ未確認" in caplog.text and "確認できません" in caplog.text
    assert (tmp_path / "c.exe").read_bytes() == payload


def test_download_installer_passes_release_info(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"MZ" + bytes(1000)
    _serve(monkeypatch, payload)
    info = ReleaseInfo(
        version="0.1.14",
        tag="v0.1.14",
        url="u",
        installer_url="https://x/setup.exe",
        installer_size=len(payload),
        installer_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert download_installer(info, tmp_path / "ok.exe").read_bytes() == payload
    wrong = ReleaseInfo(version="0.1.14", tag="v0.1.14", url="u", installer_sha256="f" * 64)
    with pytest.raises(DownloadIntegrityError):
        download_installer(wrong, tmp_path / "ng.exe")
    assert not (tmp_path / "ng.exe").exists()


def test_install_kind(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert install_kind() == "python"
    exe = tmp_path / "kaisetsu.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    assert install_kind() == "portable"
    (tmp_path / "unins000.exe").write_bytes(b"")
    assert install_kind() == "installer"
