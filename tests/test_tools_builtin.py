from __future__ import annotations

from nova.tools.builtin import _extract_duckduckgo_url, _parse_duckduckgo


def test_parse_duckduckgo_extracts_results():
    html = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">'
        "Example Title</a><a class=\"result__snippet\">A useful snippet here.</a>"
    )
    results = _parse_duckduckgo(html, max_results=5)
    assert results[0]["url"] == "https://example.com"
    assert results[0]["title"] == "Example Title"
    assert "useful" in results[0]["snippet"]


def test_extract_duckduckgo_url_decodes_and_falls_back():
    assert _extract_duckduckgo_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com") == "https://example.com"
    assert _extract_duckduckgo_url("https://plain.example.com") == "https://plain.example.com"
