from app.ingest.parser import body_font_size, is_heading, merge_columns, repeated_lines, table_to_markdown


def line(text, size=10.0, bold=False):
    return {"text": text, "size": size, "bold": bold, "chars": len(text)}


def test_body_font_size_is_weighted_by_characters():
    lines = [line("Big Title", 20), line("x" * 500, 9.5), line("y" * 100, 11)]
    assert body_font_size(lines) == 9.5


def test_heading_needs_bigger_or_bold_font():
    assert is_heading(line("Risk Factors", 14), body_size=10)
    assert is_heading(line("Risk Factors", 10, bold=True), body_size=10)
    assert not is_heading(line("Risk Factors", 10), body_size=10)


def test_heading_rejects_sentences_and_numbers():
    assert not is_heading(line("Revenue grew strongly.", 14), body_size=10)
    assert not is_heading(line("2023 2024", 14), body_size=10)


def test_repeated_lines_finds_running_headers_but_keeps_numbers():
    pages = [["Annual Report 2023", "42", f"paragraph {chr(97 + i)}"] for i in range(12)]
    assert repeated_lines(pages) == {"Annual Report #"}


def test_repeated_lines_masks_page_numbers():
    pages = [[f"Page | {i}", f"body {i} text"] for i in range(12)]
    assert repeated_lines(pages) == {"Page | #", "body # text"}


def test_repeated_lines_ignores_short_documents():
    assert repeated_lines([["Header"]] * 3) == set()


def test_merge_columns_joins_currency_sign_and_duplicate_columns():
    rows = [["Revenue", "$", "406.1", "406.1"], ["Costs", "", "200.0", "200.0"]]
    assert merge_columns(rows) == [["Revenue", "406.1"], ["Costs", "200.0"]]


def test_merge_columns_keeps_columns_that_disagree():
    rows = [["", "2023", "2024"], ["Revenue", "1", "2"]]
    assert merge_columns(rows) == rows


def test_table_to_markdown_cleans_cells_and_drops_empty_rows():
    raw = [["Item", None, "2023"], [None, None, None], ["Net\nincome", "$", " 5 "]]
    assert table_to_markdown(raw) == "| Item | 2023 |\n| Net income | 5 |"


def test_table_to_markdown_rejects_non_tables():
    assert table_to_markdown([["only one row", "x"]]) == ""
    assert table_to_markdown([["a"], ["b"]]) == ""
