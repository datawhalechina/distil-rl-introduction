"""Centralized, content-safe SEO support for the DistilRL Sphinx site.

All page metadata lives outside the learning Markdown files.  The extension
validates the bilingual registry, injects page context for Jinja templates,
and writes an allow-listed sitemap after a successful HTML build.
"""

from __future__ import annotations

from html import escape as html_escape
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.sax.saxutils import escape as xml_escape

from sphinx.errors import ConfigError


EXPECTED_PAGE_COUNT = 26
SUPPORTED_LANGUAGES = {"en", "zh-CN"}
DEFAULT_OG_IMAGE = "_static/seo/social-card.png"

ZH_NAV_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("I. 引言", ("chapter-0-zh", "chapter-1-zh")),
    ("II.1. 强化学习基础", ("chapter-2-zh", "chapter-3-zh", "chapter-4-zh")),
    ("II.2. 基于采样的学习方法", ("chapter-5-zh", "chapter-6-zh", "chapter-7-zh")),
    ("III.1. 价值函数近似", ("chapter-8-zh", "chapter-9-zh")),
    ("III.2. 策略近似", ("chapter-10-zh", "chapter-11-zh")),
)

# These aliases preserve historical inbound fragments and repair old internal
# links.  They are injected only into Sphinx's in-memory source during a build;
# the protected chapter Markdown remains byte-for-byte unchanged.
LEGACY_ANCHOR_ALIASES: dict[str, tuple[tuple[str, str], ...]] = {
    "Contents_zh/3_马尔可夫决策过程": (
        ("## 3.3 策略和价值函数", "33-策略和价值函数"),
    ),
    "Contents_zh/4_动态规划": (
        ("## 4.1 策略评估", "41-policy-evaluation-prediction-problem"),
    ),
    "Contents_zh/5_蒙特卡洛方法": (
        ("## 5.1 蒙特卡洛预测", "51-monte-carlo-prediction-evaluation"),
        (
            "### 5.1.1 状态值函数的蒙特卡洛预测",
            "511-mc-prediction-for-state-value-function",
        ),
    ),
    "Contents_zh/11_现代策略梯度方法": (
        ("## 11.1 优势函数的概念", "111-the-concept-of-advantage"),
        ("### 11.1.2 存在意义", "1112-justification-of-existence"),
    ),
}

LEGACY_LINK_REPAIRS: dict[str, dict[str, str]] = {
    "Contents_zh/10_策略梯度方法": {
        (
            "https://datawhalechina.github.io/distil-rl-introduction/Contents/"
            "9_on_policy_control_with_approximation.md#92-average-reward-a-new-"
            "way-of-formulating-control-problems"
        ): (
            "https://datawhalechina.github.io/distil-rl-introduction/Contents/"
            "9_on_policy_control_with_approximation.html#average-reward-a-new-"
            "way-of-formulating-control-problems"
        ),
    },
    "Contents_zh/11_现代策略梯度方法": {
        "(10_策略梯度方法.md#1051-一步演员-评论员)": (
            "(https://datawhalechina.github.io/distil-rl-introduction/"
            "Contents_zh/10_%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E6%96%B9"
            "%E6%B3%95.html#actor-critic-for-episodic-tasks)"
        ),
    },
}


def _repair_legacy_source_links(app, docname: str, source: list[str]) -> None:
    """Normalize historical fragment URLs in memory without touching chapters."""

    rendered_source = source[0]
    for heading_prefix, alias in LEGACY_ANCHOR_ALIASES.get(docname, ()):
        marker = next(
            (line for line in rendered_source.splitlines() if line.startswith(heading_prefix)),
            None,
        )
        if marker:
            rendered_source = rendered_source.replace(
                marker, f"({alias})=\n{marker}", 1
            )
    for old, new in LEGACY_LINK_REPAIRS.get(docname, {}).items():
        rendered_source = rendered_source.replace(old, new)
    source[0] = rendered_source


def _load_registry(app) -> dict[str, Any]:
    registry_path = Path(app.confdir) / app.config.seo_registry_path
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"SEO registry not found: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid SEO registry JSON: {exc}") from exc

    pages = registry.get("pages") if isinstance(registry, dict) else None
    if not isinstance(pages, list):
        raise ConfigError("SEO registry must contain a top-level 'pages' list")
    if len(pages) != EXPECTED_PAGE_COUNT:
        raise ConfigError(
            f"SEO registry must contain {EXPECTED_PAGE_COUNT} pages; found {len(pages)}"
        )

    required = {
        "id",
        "pagename",
        "language",
        "seo_title",
        "description",
        "canonical_path",
        "alternate_pagename",
        "og_type",
        "previous",
        "next",
        "breadcrumb",
    }
    by_name: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    titles: set[str] = set()
    descriptions: set[str] = set()
    canonicals: set[str] = set()

    for position, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ConfigError(f"SEO page at index {position} must be an object")
        missing = required.difference(page)
        if missing:
            raise ConfigError(
                f"SEO page at index {position} is missing: {', '.join(sorted(missing))}"
            )
        if page["language"] not in SUPPORTED_LANGUAGES:
            raise ConfigError(
                f"Unsupported language for {page['pagename']}: {page['language']}"
            )
        for field, seen in (
            ("pagename", by_name),
            ("id", by_id),
            ("seo_title", titles),
            ("description", descriptions),
            ("canonical_path", canonicals),
        ):
            value = page[field]
            if value in seen:
                raise ConfigError(f"Duplicate SEO {field}: {value}")
            if isinstance(seen, dict):
                seen[value] = page
            else:
                seen.add(value)
        breadcrumb = page["breadcrumb"]
        if not isinstance(breadcrumb, list) or not breadcrumb:
            raise ConfigError(f"Breadcrumb must be a non-empty list: {page['pagename']}")

    for page in pages:
        alternate = by_name.get(page["alternate_pagename"])
        if not alternate:
            raise ConfigError(
                f"Missing alternate page for {page['pagename']}: "
                f"{page['alternate_pagename']}"
            )
        if alternate["alternate_pagename"] != page["pagename"]:
            raise ConfigError(f"Non-reciprocal alternate for {page['pagename']}")
        if alternate["language"] == page["language"]:
            raise ConfigError(f"Alternate has same language for {page['pagename']}")
        for relation in ("previous", "next"):
            target_id = page[relation]
            if target_id is None:
                continue
            target = by_id.get(target_id)
            if not target:
                raise ConfigError(
                    f"Unknown {relation} id for {page['pagename']}: {target_id}"
                )
            if target["language"] != page["language"]:
                raise ConfigError(
                    f"Cross-language {relation} relation for {page['pagename']}"
                )

    registry["pages_by_name"] = by_name
    registry["pages_by_id"] = by_id
    return registry


def _absolute_url(site_url: str, path: str) -> str:
    base = site_url.rstrip("/") + "/"
    normalized = path.lstrip("/")
    if not normalized:
        return base
    return base + quote(normalized, safe="/._~-:")


def _short_title(page: dict[str, Any]) -> str:
    breadcrumb = page.get("breadcrumb", [])
    if breadcrumb and isinstance(breadcrumb[-1], dict):
        return str(breadcrumb[-1].get("name") or page["seo_title"])
    return page["seo_title"].split(" | ", 1)[0]


def _breadcrumb_items(
    page: dict[str, Any], by_name: dict[str, dict[str, Any]], site_url: str
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for crumb in page["breadcrumb"]:
        if not isinstance(crumb, dict) or not crumb.get("name"):
            raise ConfigError(f"Invalid breadcrumb for {page['pagename']}")
        crumb_name = str(crumb["name"])
        crumb_pagename = crumb.get("pagename")
        if crumb_pagename:
            crumb_page = by_name.get(str(crumb_pagename))
            if not crumb_page:
                raise ConfigError(
                    f"Unknown breadcrumb pagename for {page['pagename']}: {crumb_pagename}"
                )
            crumb_url = _absolute_url(site_url, crumb_page["canonical_path"])
        else:
            crumb_url = _absolute_url(site_url, page["canonical_path"])
        items.append({"name": crumb_name, "url": crumb_url})
    return items


def _json_ld(
    page: dict[str, Any],
    canonical_url: str,
    breadcrumbs: list[dict[str, str]],
    site_url: str,
) -> str:
    graph: list[dict[str, Any]] = []
    if page["pagename"] in {"index", "index_zh"}:
        graph.append(
            {
                "@type": "WebSite",
                "@id": site_url.rstrip("/") + "/#website",
                "url": site_url.rstrip("/") + "/",
                "name": "DistilRL",
                "description": page["description"],
                "inLanguage": page["language"],
                "creator": {
                    "@type": "Person",
                    "name": "Youxiang Dong",
                    "url": "https://github.com/Dong237",
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Datawhale",
                    "url": "https://github.com/datawhalechina",
                },
                "license": "https://www.apache.org/licenses/LICENSE-2.0",
            }
        )
    if len(breadcrumbs) > 1:
        graph.append(
            {
                "@type": "BreadcrumbList",
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "position": position,
                        "name": item["name"],
                        "item": item["url"],
                    }
                    for position, item in enumerate(breadcrumbs, start=1)
                ],
            }
        )
        graph.append(
            {
                "@type": "LearningResource",
                "@id": canonical_url + "#learning-resource",
                "url": canonical_url,
                "name": page["seo_title"],
                "description": page["description"],
                "inLanguage": page["language"],
                "isPartOf": {"@id": site_url.rstrip("/") + "/#website"},
                "creator": {
                    "@type": "Person",
                    "name": "Youxiang Dong",
                    "url": "https://github.com/Dong237",
                },
                "license": "https://www.apache.org/licenses/LICENSE-2.0",
                "educationalLevel": "Beginner to intermediate",
            }
        )
    payload = {"@context": "https://schema.org", "@graph": graph}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )


def _builder_inited(app) -> None:
    registry = _load_registry(app)
    app.env.distilrl_seo_registry = registry


def _page_context(
    app, pagename: str, templatename: str, context: dict[str, Any], doctree
) -> None:
    # Docutils' HTML5 writer and the theme both add a viewport tag in addition
    # to Sphinx's base template. Keep one authoritative tag in the final head.
    metatags = context.get("metatags")
    if isinstance(metatags, str):
        context["metatags"] = metatags.replace(
            '<meta name="viewport" content="width=device-width, initial-scale=1" />\n',
            "",
        )

    registry = getattr(app.env, "distilrl_seo_registry", None)
    if not registry:
        registry = _load_registry(app)
        app.env.distilrl_seo_registry = registry

    by_name = registry["pages_by_name"]
    by_id = registry["pages_by_id"]
    page = by_name.get(pagename)
    utility_pages = set(app.config.seo_utility_pages)

    context["seo_noindex"] = page is None or pagename in utility_pages
    context["seo_page"] = page
    if not page:
        return

    site_url = app.config.seo_site_url
    canonical_url = _absolute_url(site_url, page["canonical_path"])
    alternate = by_name[page["alternate_pagename"]]
    alternate_url = _absolute_url(site_url, alternate["canonical_path"])
    og_image = page.get("og_image") or DEFAULT_OG_IMAGE
    og_image_url = _absolute_url(site_url, og_image)
    breadcrumbs = _breadcrumb_items(page, by_name, site_url)

    hreflangs = [
        {"language": page["language"], "url": canonical_url},
        {"language": alternate["language"], "url": alternate_url},
    ]
    if pagename in {"index", "index_zh"}:
        hreflangs.append(
            {"language": "x-default", "url": _absolute_url(site_url, "")}
        )

    navigation = []
    for candidate in registry["pages"]:
        if candidate["language"] != page["language"]:
            continue
        if candidate["pagename"] in {"index", "index_zh"}:
            continue
        navigation.append(
            {
                "id": candidate["id"],
                "pagename": candidate["pagename"],
                "title": _short_title(candidate),
                "current": candidate["pagename"] == pagename,
            }
        )

    navigation_by_id = {item["id"]: item for item in navigation}
    navigation_groups = [
        {
            "title": title,
            "items": [navigation_by_id[item_id] for item_id in item_ids],
        }
        for title, item_ids in ZH_NAV_GROUPS
        if all(item_id in navigation_by_id for item_id in item_ids)
    ]

    context.update(
        {
            "language": page["language"],
            "seo_canonical_url": canonical_url,
            "seo_alternate_url": alternate_url,
            "seo_alternate_pagename": alternate["pagename"],
            "seo_alternate_language": alternate["language"],
            "seo_hreflangs": hreflangs,
            "seo_og_image_url": og_image_url,
            "seo_breadcrumbs": breadcrumbs,
            "seo_jsonld": _json_ld(page, canonical_url, breadcrumbs, site_url),
            "seo_navigation": navigation,
            "seo_navigation_groups": navigation_groups,
            "seo_home_pagename": "index_zh"
            if page["language"] == "zh-CN"
            else "index",
        }
    )

    for relation in ("prev", "next"):
        registry_field = "previous" if relation == "prev" else "next"
        target_id = page.get(registry_field)
        if not target_id:
            if pagename in {"index", "index_zh"} and relation == "next":
                first = next(
                    item
                    for item in navigation
                    if item["pagename"].endswith("0_prelude")
                    or item["pagename"].endswith("0_前言")
                )
                context[relation] = {
                    "link": app.builder.get_relative_uri(pagename, first["pagename"]),
                    "title": first["title"],
                }
            else:
                context[relation] = None
            continue
        target = by_id[target_id]
        context[relation] = {
            "link": app.builder.get_relative_uri(pagename, target["pagename"]),
            "title": _short_title(target),
        }


def _write_sitemap(app, exception) -> None:
    if exception or app.builder.name not in {"html", "dirhtml"}:
        return
    registry = getattr(app.env, "distilrl_seo_registry", None)
    if not registry:
        return

    site_url = app.config.seo_site_url
    by_name = registry["pages_by_name"]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
        'xmlns:xhtml="http://www.w3.org/1999/xhtml">',
    ]
    for page in registry["pages"]:
        canonical = _absolute_url(site_url, page["canonical_path"])
        alternate = by_name[page["alternate_pagename"]]
        alternate_url = _absolute_url(site_url, alternate["canonical_path"])
        lines.extend(
            [
                "  <url>",
                f"    <loc>{xml_escape(canonical)}</loc>",
                "    <xhtml:link rel=\"alternate\" "
                f"hreflang=\"{html_escape(page['language'])}\" "
                f"href=\"{xml_escape(canonical)}\" />",
                "    <xhtml:link rel=\"alternate\" "
                f"hreflang=\"{html_escape(alternate['language'])}\" "
                f"href=\"{xml_escape(alternate_url)}\" />",
            ]
        )
        if page["pagename"] in {"index", "index_zh"}:
            lines.append(
                "    <xhtml:link rel=\"alternate\" hreflang=\"x-default\" "
                f"href=\"{xml_escape(_absolute_url(site_url, ''))}\" />"
            )
        lines.append("  </url>")
    lines.append("</urlset>")

    output = Path(app.builder.outdir) / "sitemap.xml"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup(app):
    app.add_config_value("seo_site_url", "", "html")
    app.add_config_value("seo_registry_path", "_seo/pages.json", "html")
    app.add_config_value("seo_utility_pages", [], "html")
    app.connect("builder-inited", _builder_inited)
    app.connect("source-read", _repair_legacy_source_links)
    app.connect("html-page-context", _page_context)
    app.connect("build-finished", _write_sitemap)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
