from hw2.chunk_corpus import CHUNK_SIZE, OVERLAP, chunks


def article(text):
    return {"page": "Timeline", "url": "https://example.org/wiki/Timeline", "text": text}


def test_character_chunks_overlap_without_repeating_the_tail():
    source = article("a" * (2 * CHUNK_SIZE - OVERLAP))
    result = list(chunks(source, "char"))
    assert [(row["start"], row["end"]) for row in result] == [
        (0, CHUNK_SIZE), (CHUNK_SIZE - OVERLAP, len(source["text"])),
    ]


def test_structured_chunks_keep_hierarchy_and_verifiable_offsets():
    paragraph = "Example " * 201
    source = article(f"Intro\n\n== Events ==\r\n=== May ===\n  {paragraph}\n== References ==\nLocal citation.")
    result = list(chunks(source, "structure"))
    assert result[0]["section"] == ""
    assert result[-1]["section"] == "References"
    entries = result[1:-1]
    assert all(row["section"] == "Events / May" for row in entries)
    assert " ".join(row["text"] for row in entries) == paragraph.strip()
    for row in result:
        assert row["text"] == source["text"][row["start"]:row["end"]]
        assert 0 < len(row["text"]) <= CHUNK_SIZE
    ids = {row["id"] for row in result}
    assert len(ids) == len(result)
    assert ids.isdisjoint(row["id"] for row in chunks(source, "char"))
