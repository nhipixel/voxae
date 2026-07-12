"""Fetch openly-licensed aerial demo images from Wikimedia Commons.

Searches Commons for aerial/drone photographs, keeps only CC0 / CC BY /
CC BY-SA licensed files (license metadata read programmatically), downloads
up to N images into the demo gallery, and writes ATTRIBUTION.md.

NC research-dataset images must NEVER enter the public gallery — this script
is the only sanctioned gallery source.

Usage:
    uv run python scripts/fetch_demo_images.py --count 16
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import httpx
import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

API = "https://commons.wikimedia.org/w/api.php"
GALLERY = Path(__file__).parent.parent / "voxae" / "demo" / "assets" / "gallery"
ALLOWED_LICENSES = re.compile(r"^(cc0|cc[ -]by(?:[ -]sa)?)[ -]?\d?", re.IGNORECASE)
SEARCH_TERMS = [
    "aerial view neighborhood drone",
    "drone photograph farmland",
    "aerial photograph parking lot",
    "drone aerial construction site",
    "aerial view park trees paths",
]
HEADERS = {"User-Agent": "Voxae-demo-fetcher/0.1 (github.com/nhipixel/voxae)"}


def _search_files(client: httpx.Client, term: str, limit: int) -> list[dict]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {term}",
        "gsrnamespace": 6,  # File:
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size",
        "iiurlwidth": 1600,
    }
    r = client.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    return list(pages.values())


def _license_ok(meta: dict) -> tuple[bool, str, str]:
    ext = meta.get("extmetadata", {})
    short = ext.get("LicenseShortName", {}).get("value", "")
    artist = re.sub(r"<[^>]+>", "", ext.get("Artist", {}).get("value", "unknown")).strip()
    return bool(ALLOWED_LICENSES.match(short.strip())), short, artist


def _ascii(text: str) -> str:
    """Console/filename-safe ASCII fold (legacy Windows consoles choke on accents)."""
    return text.encode("ascii", "replace").decode("ascii")


def _download(client: httpx.Client, url: str) -> bytes:
    """Polite download with one retry on 429 (Wikimedia thumbnail rate limit)."""
    for attempt in (1, 2):
        resp = client.get(url, headers=HEADERS, timeout=60)
        if resp.status_code == 429 and attempt == 1:
            time.sleep(20)
            continue
        resp.raise_for_status()
        return resp.content
    raise httpx.HTTPError("unreachable")


@app.command()
def fetch(count: int = typer.Option(16, min=4, max=30)) -> None:
    GALLERY.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    seen: set[str] = set()
    downloaded = 0

    with httpx.Client(follow_redirects=True) as client:
        for term in SEARCH_TERMS:
            if downloaded >= count:
                break
            for page in _search_files(client, term, limit=12):
                if downloaded >= count:
                    break
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                ok, license_name, artist = _license_ok(info)
                if not ok:
                    continue
                if info.get("width", 0) < 800:
                    continue
                title = page["title"].removeprefix("File:")
                if title in seen:
                    continue
                seen.add(title)
                url = info.get("thumburl") or info["url"]
                dest = GALLERY / re.sub(r"[^\w.\-]+", "_", _ascii(title), flags=re.ASCII)
                if dest.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                try:
                    dest.write_bytes(_download(client, url))
                except httpx.HTTPError as e:
                    console.print(f"[yellow]skip {_ascii(title)}: {e}[/yellow]")
                    continue
                downloaded += 1
                page_url = f"https://commons.wikimedia.org/wiki/File:{title.replace(' ', '_')}"
                rows.append(f"| [{title}]({page_url}) | {artist} | {license_name} |")
                console.print(f"[green]{downloaded:2d}[/green] {_ascii(title)}  ({license_name})")
                time.sleep(2.5)  # stay under Wikimedia's thumbnail rate limit

    attribution = GALLERY.parent / "ATTRIBUTION.md"
    attribution.write_text(
        "# Demo gallery attribution\n\n"
        "All images sourced from Wikimedia Commons under open licenses "
        "(CC0 / CC BY / CC BY-SA). Retrieved by `scripts/fetch_demo_images.py`.\n\n"
        "| File | Author | License |\n|---|---|---|\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    console.print(f"\n[bold green]{downloaded} images -> {GALLERY}[/bold green]")
    console.print(f"attribution -> {attribution}")
    if downloaded < count:
        console.print(
            "[yellow]fewer than requested passed the license filter — rerun or add terms[/yellow]"
        )


if __name__ == "__main__":
    app()
