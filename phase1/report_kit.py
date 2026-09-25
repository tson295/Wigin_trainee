"""Shared plumbing for the assignment HTML reports.

A report template carries two placeholders:

    <!--__CHROME__-->        the shared style block and chart helpers
    /*__REPORT_DATA__*/null  the JSON payload the page renders from

``write`` fills both in and emits the result twice, because the two places a report
gets viewed want different things:

``<name>.html``
    A complete document — doctype, ``<html>``, ``<head>``, ``<body>``. Open it from
    disk, serve it with VS Code's Live Server, or drop it on any static host. Live
    Server in particular refuses to start on a file with no ``<head>``/``<body>``.

``<name>.artifact.html``
    Body content only. The Artifact publisher wraps what it is given in its own
    doctype/head/body skeleton, so a full document there would nest one inside the
    other. This is the file to pass to the Artifact tool.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PHASE_ROOT = Path(__file__).resolve().parent
CHROME = PHASE_ROOT / "report_chrome.html"

DOCUMENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{title}
</head>
<body>
{body}
</body>
</html>
"""


def build(template: Path, data: dict) -> str:
    """Return the body-only page: template + shared chrome + data."""

    html = Path(template).read_text(encoding="utf-8")
    for placeholder in ("<!--__CHROME__-->", "/*__REPORT_DATA__*/null"):
        if placeholder not in html:
            raise ValueError(f"{template} is missing the {placeholder} placeholder.")

    payload = json.dumps(data, ensure_ascii=False)
    if "�" in payload:
        raise ValueError(
            "Report data contains U+FFFD. Something decoded bytes with "
            "errors='replace' and lost them; decode with errors='backslashreplace' "
            "so partial byte sequences stay readable."
        )
    html = html.replace("<!--__CHROME__-->", CHROME.read_text(encoding="utf-8"))
    return html.replace("/*__REPORT_DATA__*/null", payload)


def as_document(body: str) -> str:
    """Wrap body-only content in a full HTML document.

    The ``<title>`` and charset declaration live at the top of the body-only form
    (where the Artifact publisher looks for them); move them into a real ``<head>``.
    """

    title_match = re.search(r"<title>.*?</title>", body, re.S)
    title = title_match.group(0) if title_match else "<title>Report</title>"
    if title_match:
        body = body.replace(title_match.group(0), "", 1)
    body = re.sub(r"<meta\s+charset=[^>]*>\s*", "", body, count=1)
    return DOCUMENT.format(title=title, body=body.strip())


def write(out_path: Path, template: Path, data: dict) -> tuple[Path, Path]:
    """Write the standalone document and the publishable body-only twin."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = build(template, data)

    artifact_path = out_path.with_suffix(".artifact.html")
    out_path.write_text(as_document(body), encoding="utf-8")
    artifact_path.write_text(body, encoding="utf-8")
    (out_path.parent / f"{out_path.stem}_data.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    return out_path, artifact_path
