from app.ingest.chunker import chunk_pages, overlap_tail, split_long, split_table
from app.ingest.parser import Block, Page


def words(s: str) -> int:  # stands in for tiktoken
    return len(s.split())


def para(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def chunks(pages, max_tokens=50, overlap=10):
    return chunk_pages(pages, "Acme Inc.", count=words, max_tokens=max_tokens, overlap=overlap)


def test_header_has_company_page_and_section():
    page = Page(7, [Block("text", "Revenue was 406.1 million.", "Results")])
    [c] = chunks([page])
    assert c.page == 7
    assert c.content == "[Acme Inc. | p.7 | Results]\nRevenue was 406.1 million."


def test_header_without_section():
    [c] = chunks([Page(1, [Block("text", "Hello there.")])])
    assert c.content.startswith("[Acme Inc. | p.1]\n")


def test_chunks_never_cross_pages():
    pages = [Page(1, [Block("text", para(5, "a"))]), Page(2, [Block("text", para(5, "b"))])]
    assert [(c.page, c.text) for c in chunks(pages)] == [(1, para(5, "a")), (2, para(5, "b"))]


def test_paragraphs_are_packed_up_to_the_limit_with_overlap():
    blocks = [Block("text", para(20, t)) for t in "abcd"]  # 4 x 20 words, limit 50
    result = chunks([Page(1, blocks)], max_tokens=50, overlap=25)
    assert [c.text.split("\n\n") for c in result] == [
        [para(20, "a"), para(20, "b")],
        [para(20, "b"), para(20, "c"), ],
        [para(20, "c"), para(20, "d")],
    ]


def test_no_chunk_made_of_overlap_only():
    blocks = [Block("text", para(20, "a")), Block("text", para(20, "b"))]
    result = chunks([Page(1, blocks)], max_tokens=40, overlap=25)
    assert [c.text for c in result] == [para(20, "a") + "\n\n" + para(20, "b")]


def test_heading_starts_a_new_chunk_without_old_tail():
    blocks = [
        Block("text", para(30, "a"), "One"),
        Block("heading", "Two", "Two"),
        Block("text", para(5, "b"), "Two"),
    ]
    result = chunks([Page(1, blocks)], max_tokens=500, overlap=40)
    # MIN_TOKENS is 80, so a 30-word chunk is too small to be closed by a heading
    assert len(result) == 1
    blocks[0] = Block("text", para(90, "a"), "One")
    first, second = chunks([Page(1, blocks)], max_tokens=500, overlap=40)
    assert first.section == "One" and second.section == "Two"
    assert second.text == "Two\n\n" + para(5, "b")


def test_tables_are_their_own_chunks():
    table = "| Item | 2023 |\n| Revenue | 5 |"
    page = Page(3, [Block("text", "Intro text.", "S"), Block("table", table, "S")])
    kinds = [(c.kind, c.text) for c in chunks([page])]
    assert ("table", table) in kinds and ("text", "Intro text.") in kinds


def test_split_table_repeats_header():
    table = "\n".join(["| h1 | h2 |"] + [f"| r{i} | {i} |" for i in range(6)])
    parts = split_table(table, words, max_tokens=15)
    assert len(parts) > 1
    assert all(p.startswith("| h1 | h2 |\n") for p in parts)
    rows = [r for p in parts for r in p.split("\n")[1:]]
    assert rows == table.split("\n")[1:]  # nothing lost, nothing doubled


def test_split_long_cuts_at_sentences():
    text = "One two three. Four five six. Seven eight nine."
    assert split_long(text, words, max_tokens=6) == ["One two three. Four five six.", "Seven eight nine."]


def test_split_long_falls_back_to_words():
    assert split_long(para(10, "w"), words, max_tokens=4) == [para(10, "w")[:11], "w4 w5 w6 w7", "w8 w9"]


def test_overlap_tail_takes_whole_paragraphs_only():
    assert overlap_tail(["a b c", "d e", "f"], words, overlap=3) == ["d e", "f"]
    assert overlap_tail(["a b c d"], words, overlap=3) == []
