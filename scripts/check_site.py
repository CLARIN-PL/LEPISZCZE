#!/usr/bin/env python3
"""Validate local links, assets, fragments, and duplicate IDs in the static site."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_PREFIX = "/LEPISZCZE/"


class PageParser(HTMLParser):
    """Collect navigable references and element IDs from one HTML page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)
            elif name == "id" and value:
                self.ids.append(value)


def parse_page(path: Path) -> PageParser:
    parser = PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def resolve_local_target(root: Path, page: Path, link_path: str) -> Path:
    decoded = unquote(link_path)
    if decoded.startswith(PROJECT_PREFIX):
        target = root / decoded.removeprefix(PROJECT_PREFIX)
    elif decoded == PROJECT_PREFIX.removesuffix("/"):
        target = root
    elif decoded.startswith("/"):
        raise ValueError(f"absolute path must start with {PROJECT_PREFIX}")
    else:
        target = page.parent / decoded

    if not decoded or decoded.endswith("/") or target.is_dir():
        target /= "index.html"
    return target.resolve()


def check_site(root: Path) -> tuple[int, int]:
    root = root.resolve()
    pages = sorted(root.rglob("*.html"))
    if not pages:
        raise ValueError(f"No HTML pages found in {root}")

    parsed_pages = {page.resolve(): parse_page(page) for page in pages}
    errors: list[str] = []
    checked_links = 0

    for page, parsed in parsed_pages.items():
        duplicate_ids = sorted({item for item in parsed.ids if parsed.ids.count(item) > 1})
        if duplicate_ids:
            errors.append(f"{page.relative_to(root)}: duplicate IDs: {', '.join(duplicate_ids)}")

        for link in parsed.links:
            url = urlparse(link)
            if url.scheme or url.netloc or link.startswith(("mailto:", "tel:")):
                continue
            checked_links += 1
            try:
                target = resolve_local_target(root, page, url.path)
            except ValueError as error:
                errors.append(f"{page.relative_to(root)}: {link}: {error}")
                continue
            if root not in target.parents and target != root:
                errors.append(f"{page.relative_to(root)}: {link}: resolves outside site root")
                continue
            if not target.exists():
                errors.append(f"{page.relative_to(root)}: {link}: target does not exist")
                continue
            if url.fragment and target.suffix == ".html":
                target_page = parsed_pages.get(target)
                if target_page is None:
                    target_page = parse_page(target)
                    parsed_pages[target] = target_page
                if unquote(url.fragment) not in target_page.ids:
                    errors.append(
                        f"{page.relative_to(root)}: {link}: fragment is missing in "
                        f"{target.relative_to(root)}"
                    )

    if errors:
        raise ValueError("Static site validation failed:\n- " + "\n- ".join(errors))
    return len(pages), checked_links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("docs"))
    args = parser.parse_args()
    pages, links = check_site(args.root)
    print(f"Validated {pages} HTML pages and {links} local links/assets")


if __name__ == "__main__":
    main()
