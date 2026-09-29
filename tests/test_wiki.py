import pytest

from acs.tools.wiki import (
    LookupState,
    SqliteCache,
    clean_page,
    get_page_obs,
    lookup,
    not_found_observation,
    sentences_of,
)

PAGE = (
    "The Colorado orogeny was an episode of mountain building in Colorado.\n"
    "== History ==\n"
    "\n"
    "ok\n"
    "The eastern sector extends into the High Plains. It is called the Central Plains "
    "orogeny. The eastern sector is large.\n"
)


def test_clean_page_drops_headers_short_lines_and_blanks():
    cleaned = clean_page(PAGE)
    assert "== History ==" not in cleaned
    assert "\nok\n" not in f"\n{cleaned}\n"
    assert cleaned.startswith("The Colorado orogeny")


def test_sentences_and_five_sentence_observation():
    sentences = sentences_of(clean_page(PAGE))
    assert sentences[0].endswith(".")
    obs = get_page_obs(clean_page(PAGE))
    assert obs.startswith("The Colorado orogeny")
    assert len(obs.split(". ")) <= 5


def test_lookup_cycles_then_exhausts():
    page = clean_page(PAGE)
    state = LookupState(page_text=page)
    first = lookup(page, "eastern sector", state)
    second = lookup(page, "eastern sector", state)
    third = lookup(page, "eastern sector", state)
    assert first.startswith("(Result 1 / 2)")
    assert second.startswith("(Result 2 / 2)")
    assert third == "No more results."


def test_lookup_resets_on_a_new_keyword():
    page = clean_page(PAGE)
    state = LookupState(page_text=page)
    lookup(page, "eastern sector", state)
    again = lookup(page, "Colorado", state)
    assert again.startswith("(Result 1 /")


def test_lookup_without_a_page():
    assert lookup(None, "x", LookupState()) == "No page loaded. Use Search first."


def test_not_found_observation_uses_python_list_repr():
    assert not_found_observation("X", ["A", "B"]) == "Could not find X. Similar: ['A', 'B']."


def test_cache_key_is_stable_and_request_specific():
    a = SqliteCache.make_key("e", {"x": 1, "y": 2})
    b = SqliteCache.make_key("e", {"y": 2, "x": 1})
    c = SqliteCache.make_key("e", {"x": 2, "y": 2})
    assert a == b and a != c


@pytest.mark.network
async def test_live_search_lookup_and_cache(tmp_path):
    from acs.tools.wiki import WikiClient

    client = WikiClient(cache=SqliteCache(tmp_path / "wiki.sqlite"))
    try:
        obs, page = await client.search("Colorado orogeny")
        assert obs.startswith("The Colorado orogeny")
        assert page is not None

        state = LookupState(page_text=page)
        assert lookup(page, "eastern sector", state).startswith("(Result 1 /")

        miss, none_page = await client.search("zzqq nonsense entity 12345")
        assert miss.startswith("Could not find")
        assert none_page is None

        before = client.http_requests
        await client.search("Colorado orogeny")
        assert client.http_requests == before  # served from the disk cache
    finally:
        await client.aclose()
        client.cache.close()
