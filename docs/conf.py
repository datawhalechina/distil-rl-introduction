"""Sphinx configuration for the DistilRL documentation site."""

from pathlib import Path
import sys


DOCS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))

# -- Project information -----------------------------------------------------

project = "DistilRL"
author = "Youxiang Dong"
copyright = "2025–2026, Youxiang Dong and DistilRL contributors"

# Keep the historical source version until the project owner publishes v1.0.
# The public HTML title is configured separately and does not expose this value.
version = "0.1"
release = "0.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "seo_metadata",
]

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "linkify",
    "substitution",
    "deflist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
language = "en"
myst_heading_anchors = 3
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "html",
    "requirements.txt",
    "robots.txt",
    "Thumbs.db",
    ".DS_Store",
]

todo_include_todos = True

# -- SEO configuration -------------------------------------------------------

seo_site_url = "https://datawhalechina.github.io/distil-rl-introduction/"
seo_registry_path = "_seo/pages.json"
seo_utility_pages = ["search", "genindex", "py-modindex", "blog", "blog/drafts"]

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "DistilRL"
html_short_title = "DistilRL"
# Canonicals are emitted by the registry-backed SEO extension.  Leaving this
# empty prevents Sphinx from adding a second, generic canonical URL.
html_baseurl = ""
html_static_path = ["_static"]
html_extra_path = ["robots.txt"]
html_logo = "_static/img/logo.png"
html_favicon = "_static/img/logo.png"
html_copy_source = False
html_show_sourcelink = False
html_last_updated_fmt = None

html_theme_options = {
    "show_navbar_depth": 3,
    "show_toc_level": 3,
    "toc_title": "Contents",
    "repository_url": "https://github.com/datawhalechina/distil-rl-introduction",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_source_button": False,
    "use_download_button": False,
    "use_fullscreen_button": True,
    "use_sidenotes": True,
}

html_sidebars = {
    "**": [
        "language-aware-logo.html",
        "language-aware-search.html",
        "language-switcher.html",
        "language-aware-nav.html",
        "github-cta.html",
    ],
}


def setup(app):
    """Register project-owned static assets."""

    app.add_css_file("seo/site.css")
    app.add_js_file("seo/site.js", loading_method="defer")
