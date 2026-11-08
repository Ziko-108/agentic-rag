from app.retrieval.companies import core_name, detect_documents

FILES = ["1-800-FLOWERS.COM, INC..pdf", "Origin Bancorp, Inc..pdf", "RWE AG.pdf", "Brave Bison Group plc.pdf", "Datalogic.pdf"]


def test_core_name_drops_legal_suffixes():
    assert core_name("Origin Bancorp, Inc..pdf") == "origin bancorp"
    assert core_name("Brave Bison Group plc.pdf") == "brave bison"
    assert core_name("1-800-FLOWERS.COM, INC..pdf") == "1 800 flowers com"


def test_detect_with_and_without_suffix():
    assert detect_documents("For Origin Bancorp, Inc., what was ...", FILES) == ["Origin Bancorp, Inc..pdf"]
    assert detect_documents("Did origin bancorp grow?", FILES) == ["Origin Bancorp, Inc..pdf"]


def test_detect_several_and_none():
    q = 'Which had the lowest income: "RWE AG", "Datalogic"?'
    assert detect_documents(q, FILES) == ["Datalogic.pdf", "RWE AG.pdf"]
    assert detect_documents("What is the weather?", FILES) == []


def test_no_match_inside_other_words():
    assert detect_documents("Is the rwexyz model new?", FILES) == []
