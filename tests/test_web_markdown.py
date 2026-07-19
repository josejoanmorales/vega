from vega.web.markdown import render_markdown


def test_headers() -> None:
    out = render_markdown("# Title\n## Sub\n### SubSub")
    assert "<h1>Title</h1>" in out
    assert "<h2>Sub</h2>" in out
    assert "<h3>SubSub</h3>" in out


def test_bold_and_italic() -> None:
    out = render_markdown("**bold** and _italic_")
    assert "<b>bold</b>" in out and "<i>italic</i>" in out


def test_table() -> None:
    out = render_markdown("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert "<table>" in out
    assert "<th>a</th>" in out and "<th>b</th>" in out
    assert "<td>1</td>" in out and "<td>2</td>" in out


def test_list_and_rule() -> None:
    out = render_markdown("- one\n- two\n\n---\n")
    assert "<li>one</li>" in out and "<li>two</li>" in out
    assert "<hr>" in out


def test_html_is_escaped() -> None:
    out = render_markdown("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_never_crashes_on_unknown_syntax() -> None:
    out = render_markdown("```python\nweird > fenced *stuff*\n```\n\t~~~")
    assert isinstance(out, str)


def test_real_briefing_renders_without_error() -> None:
    text = (
        "# Vega pre-market briefing — 2026-07-16\n\n## Regime\n\n"
        "**Composite: RISK_ON** — trend risk_on.\n\n"
        "## Ranked calls\n\n"
        "| rank | symbol | qty |\n|---|---|---|\n| 1 | CDW | 52.2 |\n"
    )
    out = render_markdown(text)
    assert "<h1>" in out and "<table>" in out and "CDW" in out
