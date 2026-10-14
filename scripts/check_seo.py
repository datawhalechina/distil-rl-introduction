#!/usr/bin/env python3
"""Validate the rendered DistilRL site against its SEO contract.

This script intentionally checks generated HTML rather than only checking the
Sphinx configuration.  A passing result therefore proves that the browser and
search crawler output contains the expected metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any, Iterable
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET


SITE_URL = "https://datawhalechina.github.io/distil-rl-introduction/"

PAGE_PAIRS: tuple[tuple[str, str], ...] = (
    ("index", "index_zh"),
    ("Contents/0_prelude", "Contents_zh/0_前言"),
    ("Contents/1_intro", "Contents_zh/1_强化学习简介"),
    ("Contents/2_multi_armed_bandits", "Contents_zh/2_多臂老虎机问题"),
    ("Contents/3_markov_decision_process", "Contents_zh/3_马尔可夫决策过程"),
    ("Contents/4_dynamic_programming", "Contents_zh/4_动态规划"),
    ("Contents/5_monte_carlo_methods", "Contents_zh/5_蒙特卡洛方法"),
    ("Contents/6_temporal_difference_learning", "Contents_zh/6_时序差分学习"),
    ("Contents/7_planning_learning_acting", "Contents_zh/7_规划_学习_执行"),
    (
        "Contents/8_on_policy_prediction_with_approximation",
        "Contents_zh/8_使用近似方法进行同策略预测",
    ),
    (
        "Contents/9_on_policy_control_with_approximation",
        "Contents_zh/9_基于函数近似的同策略控制",
    ),
    ("Contents/10_policy_gradient_methods", "Contents_zh/10_策略梯度方法"),
    ("Contents/11_modern_policy_gradient_methods", "Contents_zh/11_现代策略梯度方法"),
)

EXPECTED_PAGENAMES = frozenset(page for pair in PAGE_PAIRS for page in pair)
EXPECTED_LANGUAGE = {
    page: language
    for english, chinese in PAGE_PAIRS
    for page, language in ((english, "en"), (chinese, "zh-CN"))
}
EXPECTED_ALTERNATE = {
    page: alternate
    for english, chinese in PAGE_PAIRS
    for page, alternate in ((english, chinese), (chinese, english))
}
REQUIRED_UTILITY_PAGES = ("search.html", "genindex.html")
OPTIONAL_UTILITY_PAGES = ("blog.html", "blog/drafts.html")


def compact(value: str) -> str:
    return " ".join(value.split())


def normalized_language(value: str) -> str:
    lowered = value.strip().replace("_", "-").lower()
    if lowered == "en" or lowered.startswith("en-"):
        return "en"
    if lowered in {"zh", "zh-cn", "zh-hans"}:
        return "zh-CN"
    return value.strip()


def normalized_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    path = unquote(parsed.path) or "/"
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def expected_url(site_url: str, pagename: str) -> str:
    if pagename == "index":
        return site_url
    path = quote(f"{pagename}.html", safe="/")
    return urljoin(site_url, path)


def html_path(build_dir: Path, pagename: str) -> Path:
    return build_dir / ("index.html" if pagename == "index" else f"{pagename}.html")


class RenderedHTML(HTMLParser):
    """Collect only the rendered signals needed by the SEO checks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_langs: list[str] = []
        self.titles: list[str] = []
        self.meta_names: dict[str, list[str]] = {}
        self.meta_properties: dict[str, list[str]] = {}
        self.canonicals: list[str] = []
        self.alternates: list[tuple[str, str]] = []
        self.relation_links: dict[str, list[str]] = {}
        self.anchors: list[str] = []
        self.ids: set[str] = set()
        self.json_ld_text: list[str] = []
        self.breadcrumb_navs = 0
        self.breadcrumb_items: list[tuple[str | None, str]] = []
        self.visible_relations: dict[str, list[str]] = {}
        self.seo_events: list[str] = []
        self._title_parts: list[str] | None = None
        self._json_parts: list[str] | None = None
        self._in_breadcrumb = False
        self._breadcrumb_item_parts: list[str] | None = None
        self._breadcrumb_item_href: str | None = None

    @staticmethod
    def _attrs(attributes: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.lower(): (value or "") for key, value in attributes}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = self._attrs(attrs)
        tag = tag.lower()
        element_id = attributes.get("id")
        if element_id:
            self.ids.add(unquote(element_id))
        if tag == "a" and attributes.get("name"):
            self.ids.add(unquote(attributes["name"]))

        if tag == "html" and "lang" in attributes:
            self.html_langs.append(attributes["lang"])
        classes = set(attributes.get("class", "").split())
        if tag == "nav" and "distilrl-breadcrumbs" in classes:
            self.breadcrumb_navs += 1
            self._in_breadcrumb = True
        if self._in_breadcrumb and tag == "li":
            self._breadcrumb_item_parts = []
            self._breadcrumb_item_href = None
        if self._in_breadcrumb and tag == "a" and self._breadcrumb_item_parts is not None:
            self._breadcrumb_item_href = attributes.get("href")
        if tag == "a" and attributes.get("href") is not None:
            if "left-prev" in classes:
                self.visible_relations.setdefault("prev", []).append(attributes["href"])
            if "right-next" in classes:
                self.visible_relations.setdefault("next", []).append(attributes["href"])
        if attributes.get("data-seo-event"):
            self.seo_events.append(attributes["data-seo-event"])
        if tag == "title":
            self._title_parts = []
        elif tag == "meta":
            content = attributes.get("content", "")
            if attributes.get("name"):
                key = attributes["name"].strip().lower()
                self.meta_names.setdefault(key, []).append(content)
            if attributes.get("property"):
                key = attributes["property"].strip().lower()
                self.meta_properties.setdefault(key, []).append(content)
        elif tag == "link":
            rels = {item.lower() for item in attributes.get("rel", "").split()}
            href = attributes.get("href", "")
            if "canonical" in rels:
                self.canonicals.append(href)
            if "alternate" in rels and attributes.get("hreflang"):
                self.alternates.append((attributes["hreflang"], href))
            for relation in ("prev", "next"):
                if relation in rels:
                    self.relation_links.setdefault(relation, []).append(href)
        elif tag == "a" and attributes.get("href") is not None:
            self.anchors.append(attributes["href"])
        elif tag == "script" and attributes.get("type", "").lower() == "application/ld+json":
            self._json_parts = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if self._title_parts is not None:
            self._title_parts.append(data)
        if self._json_parts is not None:
            self._json_parts.append(data)
        if self._breadcrumb_item_parts is not None:
            self._breadcrumb_item_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self._title_parts is not None:
            self.titles.append(compact("".join(self._title_parts)))
            self._title_parts = None
        elif tag == "script" and self._json_parts is not None:
            self.json_ld_text.append("".join(self._json_parts).strip())
            self._json_parts = None
        elif tag == "li" and self._in_breadcrumb and self._breadcrumb_item_parts is not None:
            self.breadcrumb_items.append(
                (self._breadcrumb_item_href, compact("".join(self._breadcrumb_item_parts)))
            )
            self._breadcrumb_item_parts = None
            self._breadcrumb_item_href = None
        elif tag == "nav" and self._in_breadcrumb:
            self._in_breadcrumb = False


def parse_html(path: Path) -> RenderedHTML:
    parser = RenderedHTML()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


@dataclass
class Checks:
    errors: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    def one(self, values: list[str], label: str, page: str) -> str | None:
        if len(values) != 1:
            self.fail(f"{page}: expected exactly one {label}, found {len(values)}")
            return values[0] if values else None
        if not compact(values[0]):
            self.fail(f"{page}: {label} is empty")
        return values[0]


def read_registry(path: Path, checks: Checks) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        checks.fail(f"metadata registry does not exist: {path}")
        return {}
    except json.JSONDecodeError as exc:
        checks.fail(f"metadata registry is invalid JSON: {exc}")
        return {}

    if isinstance(raw, dict) and isinstance(raw.get("pages"), list):
        records = raw["pages"]
    elif isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and all(isinstance(value, dict) for value in raw.values()):
        records = []
        for key, value in raw.items():
            record = dict(value)
            record.setdefault("pagename", key)
            records.append(record)
    else:
        checks.fail("metadata registry must be a page list, {pages: [...]}, or pagename mapping")
        return {}

    registry: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            checks.fail(f"metadata record {index} is not an object")
            continue
        pagename = record.get("pagename")
        if not isinstance(pagename, str) or not pagename:
            checks.fail(f"metadata record {index} has no pagename")
            continue
        if pagename in registry:
            checks.fail(f"duplicate metadata pagename: {pagename}")
            continue
        registry[pagename] = record

    actual = set(registry)
    if actual != EXPECTED_PAGENAMES:
        missing = sorted(EXPECTED_PAGENAMES - actual)
        extra = sorted(actual - EXPECTED_PAGENAMES)
        checks.fail(f"metadata whitelist must contain exactly 26 pages; missing={missing}, extra={extra}")

    for page in sorted(EXPECTED_PAGENAMES & actual):
        record = registry[page]
        expected_lang = EXPECTED_LANGUAGE[page]
        if normalized_language(str(record.get("language", ""))) != expected_lang:
            checks.fail(f"{page}: metadata language must be {expected_lang}")
        if record.get("alternate_pagename") != EXPECTED_ALTERNATE[page]:
            checks.fail(
                f"{page}: alternate_pagename must be {EXPECTED_ALTERNATE[page]!r}, "
                f"found {record.get('alternate_pagename')!r}"
            )
        for key in ("seo_title", "description", "og_type", "og_image"):
            if key not in record or not isinstance(record[key], str) or not record[key].strip():
                checks.fail(f"{page}: metadata field {key!r} must be a non-empty string")
        og_image = str(record.get("og_image", ""))
        if urlsplit(og_image).scheme or not og_image.startswith("_static/seo/"):
            checks.fail(f"{page}: og_image must be a site-local asset under _static/seo/")
        canonical_path = record.get("canonical_path")
        if not isinstance(canonical_path, str) or (not canonical_path and page != "index"):
            checks.fail(
                f"{page}: metadata field 'canonical_path' must be a string and may be empty only for index"
            )

    for english, chinese in PAGE_PAIRS:
        if english in registry and chinese in registry:
            if registry[english].get("alternate_pagename") != chinese:
                checks.fail(f"{english}: broken English-to-Chinese language pair")
            if registry[chinese].get("alternate_pagename") != english:
                checks.fail(f"{chinese}: broken Chinese-to-English language pair")

    ids: dict[str, str] = {}
    for pagename, record in registry.items():
        item_id = record.get("id")
        if not isinstance(item_id, str) or not item_id:
            checks.fail(f"{pagename}: metadata field 'id' must be a non-empty string")
        elif item_id in ids:
            checks.fail(f"{pagename}: metadata id duplicates {ids[item_id]}: {item_id}")
        else:
            ids[item_id] = pagename

    language_chains = (
        ["index", *(english for english, _ in PAGE_PAIRS[1:])],
        ["index_zh", *(chinese for _, chinese in PAGE_PAIRS[1:])],
    )
    for chain in language_chains:
        for index, pagename in enumerate(chain):
            record = registry.get(pagename)
            if not record:
                continue
            expected_previous = registry[chain[index - 1]]["id"] if index else None
            expected_next = registry[chain[index + 1]]["id"] if index + 1 < len(chain) else None
            if record.get("previous") != expected_previous:
                checks.fail(
                    f"{pagename}: previous must be {expected_previous!r}, "
                    f"found {record.get('previous')!r}"
                )
            if record.get("next") != expected_next:
                checks.fail(
                    f"{pagename}: next must be {expected_next!r}, found {record.get('next')!r}"
                )

    return registry


def registry_canonical(site_url: str, value: str) -> str:
    value = value.strip()
    if value == "/":
        return site_url
    if urlsplit(value).scheme:
        return value
    if value.startswith("/distil-rl-introduction/"):
        origin = f"{urlsplit(site_url).scheme}://{urlsplit(site_url).netloc}/"
        return urljoin(origin, quote(unquote(value), safe="/"))
    return urljoin(site_url, quote(unquote(value), safe="/"))


def json_types(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        item_type = value.get("@type")
        if isinstance(item_type, str):
            yield item_type
        elif isinstance(item_type, list):
            yield from (item for item in item_type if isinstance(item, str))
        for child in value.values():
            yield from json_types(child)
    elif isinstance(value, list):
        for child in value:
            yield from json_types(child)


def json_objects_of_type(value: Any, item_type: str) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        types = value.get("@type")
        if types == item_type or (isinstance(types, list) and item_type in types):
            yield value
        for child in value.values():
            yield from json_objects_of_type(child, item_type)
    elif isinstance(value, list):
        for child in value:
            yield from json_objects_of_type(child, item_type)


def png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", header[16:24])


def parse_json_ld(parser: RenderedHTML, page: str, checks: Checks) -> list[Any]:
    payloads: list[Any] = []
    if not parser.json_ld_text:
        checks.fail(f"{page}: missing application/ld+json structured data")
        return payloads
    for index, text in enumerate(parser.json_ld_text, start=1):
        try:
            payloads.append(json.loads(text))
        except json.JSONDecodeError as exc:
            checks.fail(f"{page}: JSON-LD block {index} is invalid: {exc}")
    return payloads


def check_content_page(
    page: str,
    parser: RenderedHTML,
    record: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    build_dir: Path,
    site_url: str,
    checks: Checks,
) -> tuple[str | None, str | None, str | None]:
    expected_lang = EXPECTED_LANGUAGE[page]
    expected_canonical = expected_url(site_url, page)

    title = checks.one(parser.titles, "<title>", page)
    description = checks.one(parser.meta_names.get("description", []), "meta description", page)
    canonical = checks.one(parser.canonicals, "canonical link", page)
    viewport = checks.one(parser.meta_names.get("viewport", []), "viewport meta tag", page)
    if viewport and "width=device-width" not in viewport.replace(" ", "").lower():
        checks.fail(f"{page}: viewport must include width=device-width")

    if title:
        if "0.1 documentation" in title.lower():
            checks.fail(f"{page}: title still contains Sphinx version boilerplate")
        if compact(title) != compact(str(record.get("seo_title", ""))):
            checks.fail(f"{page}: rendered title does not match the metadata registry")
        title_limit = 70 if expected_lang == "en" else 55
        if len(compact(title)) > title_limit:
            checks.fail(
                f"{page}: title exceeds the project snippet guardrail "
                f"({len(compact(title))}>{title_limit} characters)"
            )
    if description and compact(description) != compact(str(record.get("description", ""))):
        checks.fail(f"{page}: rendered description does not match the metadata registry")
    if description:
        description_limit = 165 if expected_lang == "en" else 100
        if len(compact(description)) > description_limit:
            checks.fail(
                f"{page}: description exceeds the project snippet guardrail "
                f"({len(compact(description))}>{description_limit} characters)"
            )

    if len(parser.html_langs) != 1:
        checks.fail(f"{page}: expected one <html lang>, found {len(parser.html_langs)}")
    elif normalized_language(parser.html_langs[0]) != expected_lang:
        checks.fail(f"{page}: html lang is {parser.html_langs[0]!r}, expected {expected_lang!r}")

    if canonical:
        parts = urlsplit(canonical)
        if parts.scheme.lower() != "https" or not parts.netloc or parts.query or parts.fragment:
            checks.fail(f"{page}: canonical must be an absolute HTTPS URL without query or fragment")
        if normalized_url(canonical) != normalized_url(expected_canonical):
            checks.fail(f"{page}: canonical is {canonical!r}, expected {expected_canonical!r}")
    canonical_path = record.get("canonical_path")
    if isinstance(canonical_path, str) and canonical_path:
        metadata_url = registry_canonical(site_url, canonical_path)
        if normalized_url(metadata_url) != normalized_url(expected_canonical):
            checks.fail(f"{page}: metadata canonical_path resolves to the wrong preserved URL")

    records_by_id = {item["id"]: item for item in registry.values() if item.get("id")}
    for relation, field in (("prev", "previous"), ("next", "next")):
        values = parser.relation_links.get(relation, [])
        visible_values = parser.visible_relations.get(relation, [])
        target_id = record.get(field)
        if target_id is None:
            if values:
                checks.fail(f"{page}: expected no rel={relation}, found {len(values)}")
            if visible_values:
                checks.fail(f"{page}: expected no visible {relation} link, found {len(visible_values)}")
            continue
        target = records_by_id.get(target_id)
        if len(values) != 1:
            checks.fail(f"{page}: expected exactly one rel={relation}, found {len(values)}")
        if len(visible_values) != 1:
            checks.fail(
                f"{page}: expected exactly one visible {relation} link, found {len(visible_values)}"
            )
        if target:
            expected_relation_url = urljoin(
                site_url, quote(f"{target['pagename']}.html", safe="/")
            )
            for label, candidates in ((f"rel={relation}", values), (f"visible {relation}", visible_values)):
                if len(candidates) != 1:
                    continue
                actual_relation_url = urljoin(expected_canonical, candidates[0])
                if normalized_url(actual_relation_url) != normalized_url(expected_relation_url):
                    checks.fail(
                        f"{page}: {label} points to {actual_relation_url!r}, "
                        f"expected {expected_relation_url!r}"
                    )

    hreflang: dict[str, list[str]] = {}
    for language, href in parser.alternates:
        language = normalized_language(language) if language.lower() != "x-default" else "x-default"
        hreflang.setdefault(language, []).append(href)
    alternate_page = EXPECTED_ALTERNATE[page]
    pair = {
        expected_lang: expected_canonical,
        EXPECTED_LANGUAGE[alternate_page]: expected_url(site_url, alternate_page),
    }
    for language, url in pair.items():
        values = hreflang.get(language, [])
        if len(values) != 1:
            checks.fail(f"{page}: expected one hreflang={language}, found {len(values)}")
        elif normalized_url(values[0]) != normalized_url(url):
            checks.fail(f"{page}: hreflang={language} points to {values[0]!r}, expected {url!r}")
    if page in {"index", "index_zh"}:
        values = hreflang.get("x-default", [])
        if len(values) != 1 or normalized_url(values[0]) != normalized_url(site_url):
            checks.fail(f"{page}: homepage must have one x-default alternate pointing to {site_url}")
    expected_hreflang_keys = set(pair)
    if page in {"index", "index_zh"}:
        expected_hreflang_keys.add("x-default")
    if set(hreflang) != expected_hreflang_keys:
        checks.fail(
            f"{page}: hreflang set must be exactly {sorted(expected_hreflang_keys)}, "
            f"found {sorted(hreflang)}"
        )

    robots = parser.meta_names.get("robots", [])
    if len(robots) > 1:
        checks.fail(f"{page}: expected at most one robots meta tag")
    if robots:
        directives = {part.strip().lower() for part in robots[0].split(",")}
        if "noindex" in directives or "nofollow" in directives:
            checks.fail(f"{page}: canonical content page must remain indexable and followable")

    expected_breadcrumbs = 0 if page in {"index", "index_zh"} else 1
    if parser.breadcrumb_navs != expected_breadcrumbs:
        checks.fail(
            f"{page}: expected {expected_breadcrumbs} visible breadcrumb navigation, "
            f"found {parser.breadcrumb_navs}"
        )
    expected_visible_crumbs = [] if page in {"index", "index_zh"} else record.get("breadcrumb", [])
    if len(parser.breadcrumb_items) != len(expected_visible_crumbs):
        checks.fail(
            f"{page}: visible breadcrumb must contain {len(expected_visible_crumbs)} items, "
            f"found {len(parser.breadcrumb_items)}"
        )
    else:
        for position, ((href, text), crumb) in enumerate(
            zip(parser.breadcrumb_items, expected_visible_crumbs, strict=True), start=1
        ):
            if compact(text) != compact(str(crumb.get("name", ""))):
                checks.fail(f"{page}: visible breadcrumb item {position} has the wrong label")
            if position == len(expected_visible_crumbs):
                if href:
                    checks.fail(f"{page}: current breadcrumb item must not be a link")
            else:
                crumb_page = str(crumb.get("pagename") or page)
                expected_href = expected_url(site_url, crumb_page)
                if not href or normalized_url(urljoin(expected_canonical, href)) != normalized_url(
                    expected_href
                ):
                    checks.fail(f"{page}: visible breadcrumb item {position} has the wrong URL")
    expected_language_switches = 1 if page in {"index", "index_zh"} else 2
    if parser.seo_events.count("language_switch") != expected_language_switches:
        checks.fail(
            f"{page}: expected {expected_language_switches} 'language_switch' "
            f"conversion hook(s), found {parser.seo_events.count('language_switch')}"
        )
    if parser.seo_events.count("github_star_intent") != 1:
        checks.fail(
            f"{page}: expected exactly one 'github_star_intent' conversion hook, "
            f"found {parser.seo_events.count('github_star_intent')}"
        )
    if page in {"index", "index_zh"}:
        for event_name in ("learning_path_start", "learning_path_accelerated"):
            if parser.seo_events.count(event_name) != 1:
                checks.fail(
                    f"{page}: expected exactly one {event_name!r} homepage hook, "
                    f"found {parser.seo_events.count(event_name)}"
                )

    og_site_name = checks.one(
        parser.meta_properties.get("og:site_name", []), "og:site_name", page
    )
    og_title = checks.one(parser.meta_properties.get("og:title", []), "og:title", page)
    og_description = checks.one(parser.meta_properties.get("og:description", []), "og:description", page)
    og_url = checks.one(parser.meta_properties.get("og:url", []), "og:url", page)
    og_type = checks.one(parser.meta_properties.get("og:type", []), "og:type", page)
    og_image = checks.one(parser.meta_properties.get("og:image", []), "og:image", page)
    og_image_width = checks.one(
        parser.meta_properties.get("og:image:width", []), "og:image:width", page
    )
    og_image_height = checks.one(
        parser.meta_properties.get("og:image:height", []), "og:image:height", page
    )
    og_image_alt = checks.one(
        parser.meta_properties.get("og:image:alt", []), "og:image:alt", page
    )
    og_locale = checks.one(parser.meta_properties.get("og:locale", []), "og:locale", page)
    og_locale_alternate = checks.one(
        parser.meta_properties.get("og:locale:alternate", []), "og:locale:alternate", page
    )

    if og_site_name != "DistilRL":
        checks.fail(f"{page}: og:site_name must be DistilRL")
    if title and og_title and compact(title) != compact(og_title):
        checks.fail(f"{page}: og:title must match the rendered title")
    if description and og_description and compact(description) != compact(og_description):
        checks.fail(f"{page}: og:description must match the rendered description")
    if og_url and normalized_url(og_url) != normalized_url(expected_canonical):
        checks.fail(f"{page}: og:url must match the canonical URL")
    if og_type and og_type != str(record.get("og_type", "")):
        checks.fail(f"{page}: og:type does not match the metadata registry")
    if og_image:
        image_url = urlsplit(og_image)
        if image_url.scheme.lower() != "https" or not image_url.netloc:
            checks.fail(f"{page}: og:image must be an absolute HTTPS URL")
        expected_image = registry_canonical(site_url, str(record.get("og_image", "")))
        if normalized_url(og_image) != normalized_url(expected_image):
            checks.fail(f"{page}: og:image does not match the metadata registry")
        site_parts = urlsplit(site_url)
        site_path = unquote(site_parts.path)
        image_path = unquote(image_url.path)
        if image_url.netloc.lower() == site_parts.netloc.lower() and image_path.startswith(site_path):
            local_image = build_dir / image_path[len(site_path) :]
            dimensions = png_dimensions(local_image)
            if dimensions is None:
                checks.fail(f"{page}: social image is missing or is not a valid PNG: {local_image}")
            elif dimensions != (1200, 630):
                checks.fail(f"{page}: social image must be 1200x630, found {dimensions}")
        else:
            checks.fail(f"{page}: social image must be hosted under the canonical site path")
    if og_image_width != "1200" or og_image_height != "630":
        checks.fail(f"{page}: Open Graph image dimensions must declare 1200x630")
    if not og_image_alt or "DistilRL" not in og_image_alt:
        checks.fail(f"{page}: og:image:alt must describe DistilRL")
    expected_locale = "en_US" if expected_lang == "en" else "zh_CN"
    if og_locale and og_locale != expected_locale:
        checks.fail(f"{page}: og:locale is {og_locale!r}, expected {expected_locale!r}")
    expected_alternate_locale = "zh_CN" if expected_lang == "en" else "en_US"
    if og_locale_alternate and og_locale_alternate != expected_alternate_locale:
        checks.fail(
            f"{page}: og:locale:alternate is {og_locale_alternate!r}, "
            f"expected {expected_alternate_locale!r}"
        )

    twitter_card = checks.one(parser.meta_names.get("twitter:card", []), "twitter:card", page)
    twitter_title = checks.one(parser.meta_names.get("twitter:title", []), "twitter:title", page)
    twitter_description = checks.one(
        parser.meta_names.get("twitter:description", []), "twitter:description", page
    )
    twitter_image = checks.one(parser.meta_names.get("twitter:image", []), "twitter:image", page)
    twitter_image_alt = checks.one(
        parser.meta_names.get("twitter:image:alt", []), "twitter:image:alt", page
    )
    if twitter_card and twitter_card.lower() != "summary_large_image":
        checks.fail(f"{page}: twitter:card must be summary_large_image")
    if title and twitter_title and compact(title) != compact(twitter_title):
        checks.fail(f"{page}: twitter:title must match the rendered title")
    if description and twitter_description and compact(description) != compact(twitter_description):
        checks.fail(f"{page}: twitter:description must match the rendered description")
    if og_image and twitter_image and normalized_url(og_image) != normalized_url(twitter_image):
        checks.fail(f"{page}: twitter:image must match og:image")
    if twitter_image_alt and og_image_alt and compact(twitter_image_alt) != compact(og_image_alt):
        checks.fail(f"{page}: twitter:image:alt must match og:image:alt")

    payloads = parse_json_ld(parser, page, checks)
    for index, payload in enumerate(payloads, start=1):
        if not isinstance(payload, dict) or payload.get("@context") != "https://schema.org":
            checks.fail(f"{page}: JSON-LD block {index} must declare https://schema.org context")
    types = {item_type for payload in payloads for item_type in json_types(payload)}
    required_types = {"WebSite"} if page in {"index", "index_zh"} else {
        "BreadcrumbList",
        "LearningResource",
    }
    missing_types = required_types - types
    if missing_types:
        checks.fail(f"{page}: JSON-LD is missing required types {sorted(missing_types)}")

    website_objects = [
        item for payload in payloads for item in json_objects_of_type(payload, "WebSite")
    ]
    if page in {"index", "index_zh"}:
        if len(website_objects) != 1:
            checks.fail(f"{page}: expected exactly one WebSite JSON-LD object")
        else:
            website = website_objects[0]
            expected_website_id = site_url.rstrip("/") + "/#website"
            if str(website.get("@id", "")).strip() != expected_website_id:
                checks.fail(f"{page}: WebSite.@id must use the canonical site identifier")
            if normalized_url(str(website.get("url", ""))) != normalized_url(site_url):
                checks.fail(f"{page}: WebSite.url must be the canonical site root")
            if website.get("name") != "DistilRL" or website.get("inLanguage") != expected_lang:
                checks.fail(f"{page}: WebSite name/language does not match the page")
            if compact(str(website.get("description", ""))) != compact(str(record["description"])):
                checks.fail(f"{page}: WebSite.description must match the metadata registry")
            if website.get("license") != "https://www.apache.org/licenses/LICENSE-2.0":
                checks.fail(f"{page}: WebSite.license must declare Apache-2.0")
            creator = website.get("creator")
            publisher = website.get("publisher")
            if not isinstance(creator, dict) or creator.get("url") != "https://github.com/Dong237":
                checks.fail(f"{page}: WebSite.creator must identify the author")
            if (
                not isinstance(publisher, dict)
                or publisher.get("url") != "https://github.com/datawhalechina"
            ):
                checks.fail(f"{page}: WebSite.publisher must identify Datawhale")

    if page not in {"index", "index_zh"}:
        resources = [
            item
            for payload in payloads
            for item in json_objects_of_type(payload, "LearningResource")
        ]
        if len(resources) != 1:
            checks.fail(f"{page}: expected exactly one LearningResource JSON-LD object")
        else:
            resource = resources[0]
            expected_resource = {
                "url": expected_canonical,
                "name": record["seo_title"],
                "description": record["description"],
                "inLanguage": expected_lang,
                "license": "https://www.apache.org/licenses/LICENSE-2.0",
            }
            for key, expected_value in expected_resource.items():
                actual_value = resource.get(key)
                if key == "url":
                    matches = normalized_url(str(actual_value or "")) == normalized_url(expected_value)
                else:
                    matches = compact(str(actual_value or "")) == compact(str(expected_value))
                if not matches:
                    checks.fail(f"{page}: LearningResource.{key} does not match the SEO contract")
            expected_website_id = site_url.rstrip("/") + "/#website"
            is_part_of = resource.get("isPartOf")
            if (
                not isinstance(is_part_of, dict)
                or str(is_part_of.get("@id", "")).strip() != expected_website_id
            ):
                checks.fail(f"{page}: LearningResource.isPartOf must reference the site WebSite")

        breadcrumb_objects = [
            item
            for payload in payloads
            for item in json_objects_of_type(payload, "BreadcrumbList")
        ]
        if len(breadcrumb_objects) != 1:
            checks.fail(f"{page}: expected exactly one BreadcrumbList JSON-LD object")
        else:
            elements = breadcrumb_objects[0].get("itemListElement", [])
            expected_crumbs = record.get("breadcrumb", [])
            if not isinstance(elements, list) or len(elements) != len(expected_crumbs):
                checks.fail(f"{page}: BreadcrumbList length does not match the metadata registry")
            else:
                for position, (element, crumb) in enumerate(
                    zip(elements, expected_crumbs, strict=True), start=1
                ):
                    crumb_page = str(crumb.get("pagename") or page)
                    expected_item_url = expected_url(site_url, crumb_page)
                    if (
                        element.get("position") != position
                        or compact(str(element.get("name", ""))) != compact(str(crumb.get("name", "")))
                        or normalized_url(str(element.get("item", "")))
                        != normalized_url(expected_item_url)
                    ):
                        checks.fail(f"{page}: BreadcrumbList item {position} is inconsistent")

    return title, description, canonical


def check_utility_pages(build_dir: Path, checks: Checks) -> None:
    for relative in REQUIRED_UTILITY_PAGES:
        path = build_dir / relative
        if not path.is_file():
            checks.fail(f"required utility page is missing: {relative}")
            continue
        parser = parse_html(path)
        values = parser.meta_names.get("robots", [])
        robots = checks.one(values, "robots noindex", relative)
        if robots:
            directives = {part.strip().lower() for part in robots.split(",")}
            if "noindex" not in directives or "follow" not in directives or "nofollow" in directives:
                checks.fail(f"{relative}: utility page robots must be noindex,follow")

    for relative in OPTIONAL_UTILITY_PAGES:
        path = build_dir / relative
        if not path.is_file():
            continue
        parser = parse_html(path)
        robots = checks.one(parser.meta_names.get("robots", []), "robots noindex", relative)
        if robots:
            directives = {part.strip().lower() for part in robots.split(",")}
            if "noindex" not in directives or "follow" not in directives or "nofollow" in directives:
                checks.fail(f"{relative}: utility page robots must be noindex,follow")


def check_sitemap(
    build_dir: Path,
    site_url: str,
    registry: dict[str, dict[str, Any]],
    checks: Checks,
) -> None:
    path = build_dir / "sitemap.xml"
    if not path.is_file():
        checks.fail("sitemap.xml is missing from the build output")
        return
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        checks.fail(f"sitemap.xml is invalid XML: {exc}")
        return
    if root.tag.rsplit("}", 1)[-1] != "urlset":
        checks.fail("sitemap.xml root must be <urlset>")
    locations = [
        compact(element.text or "")
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "loc"
    ]
    normalized_locations = [normalized_url(location) for location in locations]
    expected = {normalized_url(expected_url(site_url, page)) for page in EXPECTED_PAGENAMES}
    actual = set(normalized_locations)
    if len(locations) != 26 or len(actual) != 26:
        checks.fail(f"sitemap.xml must contain exactly 26 unique URLs; found {len(locations)} entries")
    if actual != expected:
        checks.fail(
            "sitemap.xml must contain only the canonical 26-page whitelist; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )

    page_by_location = {
        normalized_url(expected_url(site_url, pagename)): pagename
        for pagename in EXPECTED_PAGENAMES
    }
    for url_element in root:
        if url_element.tag.rsplit("}", 1)[-1] != "url":
            continue
        location_element = next(
            (
                child
                for child in url_element
                if child.tag.rsplit("}", 1)[-1] == "loc"
            ),
            None,
        )
        location = normalized_url(compact(location_element.text or "")) if location_element is not None else ""
        pagename = page_by_location.get(location)
        if not pagename:
            continue
        alternates: dict[str, list[str]] = {}
        for child in url_element:
            if child.tag.rsplit("}", 1)[-1] != "link":
                continue
            if child.attrib.get("rel") != "alternate" or not child.attrib.get("hreflang"):
                continue
            language = child.attrib["hreflang"]
            language = language if language == "x-default" else normalized_language(language)
            alternates.setdefault(language, []).append(child.attrib.get("href", ""))
        alternate_page = EXPECTED_ALTERNATE[pagename]
        expected_alternates = {
            EXPECTED_LANGUAGE[pagename]: expected_url(site_url, pagename),
            EXPECTED_LANGUAGE[alternate_page]: expected_url(site_url, alternate_page),
        }
        if pagename in {"index", "index_zh"}:
            expected_alternates["x-default"] = site_url
        if set(alternates) != set(expected_alternates):
            checks.fail(
                f"sitemap.xml {pagename}: hreflang set is {sorted(alternates)}, "
                f"expected {sorted(expected_alternates)}"
            )
        for language, expected_href in expected_alternates.items():
            values = alternates.get(language, [])
            if len(values) != 1 or normalized_url(values[0]) != normalized_url(expected_href):
                checks.fail(
                    f"sitemap.xml {pagename}: hreflang={language} must point to {expected_href}"
                )


def check_robots(build_dir: Path, site_url: str, checks: Checks) -> None:
    path = build_dir / "robots.txt"
    if not path.is_file():
        checks.fail("robots.txt is missing from the build output")
        return
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    normalized_lines = [line.casefold() for line in lines]
    normalized = set(normalized_lines)
    if "user-agent: *" not in normalized or "allow: /" not in normalized:
        checks.fail("robots.txt must allow all crawlers to access the site")
    if any(line.startswith("disallow:") and line != "disallow:" for line in normalized):
        checks.fail("robots.txt must not disallow indexable site paths")
    expected_sitemap = f"sitemap: {urljoin(site_url, 'sitemap.xml')}".casefold()
    sitemap_lines = [line for line in normalized_lines if line.startswith("sitemap:")]
    if sitemap_lines != [expected_sitemap]:
        checks.fail(f"robots.txt must declare exactly one canonical sitemap: {expected_sitemap}")


def human_html_files(build_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in build_dir.rglob("*.html"):
        relative = path.relative_to(build_dir)
        if any(part.startswith("_static") or part.startswith("_sphinx") for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


def check_stale_links(build_dir: Path, checks: Checks) -> dict[Path, RenderedHTML]:
    parsers: dict[Path, RenderedHTML] = {}
    old_repo = "github.com/dong237/distilrlintroduction"
    master_pattern = re.compile(
        r"github\.com/(?:datawhalechina/distil-rl-introduction|dong237/distilrlintroduction)/"
        r"(?:blob|edit|tree)/master(?:/|$)",
        flags=re.IGNORECASE,
    )
    for path in human_html_files(build_dir):
        parser = parse_html(path)
        parsers[path.resolve()] = parser
        relative = path.relative_to(build_dir).as_posix()
        for href in parser.anchors:
            decoded = unquote(href).lower()
            if old_repo in decoded:
                checks.fail(f"{relative}: link still targets the old personal repository: {href}")
            if master_pattern.search(unquote(href)):
                checks.fail(f"{relative}: repository link still targets the master branch: {href}")
            if "_sources/" in decoded or decoded.startswith("_sources"):
                checks.fail(f"{relative}: source-copy link must not be exposed: {href}")

    source_dir = build_dir / "_sources"
    if source_dir.exists() and any(source_dir.rglob("*")):
        checks.fail("build output still contains _sources copies; set html_copy_source=False")
    return parsers


def local_target(
    href: str,
    source: Path,
    build_dir: Path,
    site_url: str,
) -> tuple[Path, str] | None:
    parsed = urlsplit(href)
    site = urlsplit(site_url)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc and parsed.netloc.lower() != site.netloc.lower():
        return None

    path_text = unquote(parsed.path)
    site_prefix = unquote(site.path)
    if parsed.netloc or path_text.startswith("/"):
        if not path_text.startswith(site_prefix):
            return None
        path_text = path_text[len(site_prefix) :]
        target = build_dir / path_text
    elif path_text:
        target = source.parent / path_text
    else:
        target = source

    target = target.resolve()
    try:
        target.relative_to(build_dir.resolve())
    except ValueError:
        return None
    if path_text.endswith("/") or target.is_dir():
        target = target / "index.html"
    return target, unquote(parsed.fragment)


def check_internal_links(
    build_dir: Path,
    site_url: str,
    parsers: dict[Path, RenderedHTML],
    checks: Checks,
) -> None:
    # Iterate over a snapshot because a link can target an HTML file that was
    # not part of the initial human-page scan (for example, a generated helper
    # page); parsing that target adds it to the cache.
    for source, parser in list(parsers.items()):
        source_relative = source.relative_to(build_dir.resolve()).as_posix()
        for href in parser.anchors:
            target_info = local_target(href, source, build_dir, site_url)
            if target_info is None:
                continue
            target, fragment = target_info
            if not target.exists():
                checks.fail(f"{source_relative}: broken internal link {href!r}")
                continue
            if fragment and target.suffix.lower() == ".html":
                target_parser = parsers.get(target.resolve())
                if target_parser is None:
                    target_parser = parse_html(target)
                    parsers[target.resolve()] = target_parser
                if fragment not in target_parser.ids:
                    checks.fail(
                        f"{source_relative}: internal link fragment {fragment!r} does not exist in "
                        f"{target.relative_to(build_dir).as_posix()}"
                    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DistilRL's rendered technical SEO output.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("docs/_build/html"),
        help="Rendered Sphinx HTML directory (default: docs/_build/html).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("docs/_seo/pages.json"),
        help="Central page metadata registry (default: docs/_seo/pages.json).",
    )
    parser.add_argument(
        "--site-url",
        default=SITE_URL,
        help=f"Canonical project URL (default: {SITE_URL}).",
    )
    parser.add_argument(
        "--skip-internal-links",
        action="store_true",
        help="Skip local link/anchor verification (not used by CI).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks = Checks()
    build_dir = args.build_dir.resolve()
    metadata_path = args.metadata.resolve()
    site_url = args.site_url.rstrip("/") + "/"

    if not build_dir.is_dir():
        print(f"SEO check could not run: build directory does not exist: {build_dir}", file=sys.stderr)
        return 2
    parsed_site = urlsplit(site_url)
    if parsed_site.scheme.lower() != "https" or not parsed_site.netloc:
        print("SEO check could not run: --site-url must be an absolute HTTPS URL", file=sys.stderr)
        return 2

    registry = read_registry(metadata_path, checks)
    rendered: dict[str, RenderedHTML] = {}
    titles: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    canonicals: dict[str, str] = {}

    for page in sorted(EXPECTED_PAGENAMES):
        path = html_path(build_dir, page)
        if not path.is_file():
            checks.fail(f"{page}: expected rendered file is missing: {path.relative_to(build_dir)}")
            continue
        parser = parse_html(path)
        rendered[page] = parser
        record = registry.get(page)
        if record is None:
            continue
        title, description, canonical = check_content_page(
            page, parser, record, registry, build_dir, site_url, checks
        )
        if title:
            key = compact(title).casefold()
            if key in titles:
                checks.fail(f"{page}: title duplicates {titles[key]}")
            else:
                titles[key] = page
        if description:
            key = compact(description).casefold()
            if key in descriptions:
                checks.fail(f"{page}: description duplicates {descriptions[key]}")
            else:
                descriptions[key] = page
        if canonical:
            key = normalized_url(canonical)
            if key in canonicals:
                checks.fail(f"{page}: canonical duplicates {canonicals[key]}")
            else:
                canonicals[key] = page

    if len(rendered) != 26:
        checks.fail(f"expected 26 rendered content pages, found {len(rendered)}")
    if len(titles) != 26:
        checks.fail(f"expected 26 unique rendered titles, found {len(titles)}")
    if len(descriptions) != 26:
        checks.fail(f"expected 26 unique rendered descriptions, found {len(descriptions)}")
    if len(canonicals) != 26:
        checks.fail(f"expected 26 unique rendered canonicals, found {len(canonicals)}")

    check_utility_pages(build_dir, checks)
    check_sitemap(build_dir, site_url, registry, checks)
    check_robots(build_dir, site_url, checks)
    parsers = check_stale_links(build_dir, checks)
    if not args.skip_internal_links:
        check_internal_links(build_dir, site_url, parsers, checks)

    if checks.errors:
        print(f"SEO validation failed with {len(checks.errors)} error(s):", file=sys.stderr)
        for error in checks.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "SEO validation passed: 26 canonical content pages, 13 reciprocal language pairs, "
        "reciprocal prev/next chains, utility noindex, robots and sitemap hreflang, "
        "social metadata/assets, JSON-LD, breadcrumbs, conversion hooks, and internal "
        "links are valid."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
