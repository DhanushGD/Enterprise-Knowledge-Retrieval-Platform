from ingestion.chunking import Chunk, chunk_document, chunk_text, count_tokens
from ingestion.parsing import ParsedDocument, Section


def doc(*sections, title="Doc"):
    return ParsedDocument(title, [Section(list(p), t, pg) for p, t, pg in sections])


def sentence_text(n_sentences, tag=None):
    """tag makes every sentence unique to its block, so tests can tell blocks apart."""
    if tag is None:
        return " ".join(f"This is sentence number {i} about refunds and billing." for i in range(n_sentences))
    return " ".join(f"Block {tag} sentence {i} covers refunds and billing." for i in range(n_sentences))


def test_short_section_is_one_chunk_with_heading_path():
    chunks = chunk_document(doc((["Billing", "Refunds"], "Refunds take 5 days.", None)))
    assert len(chunks) == 1
    assert chunks[0].heading_path == ["Billing", "Refunds"]
    assert chunks[0].section == "Billing > Refunds"


def test_long_section_is_split_and_every_chunk_respects_max():
    text = "\n\n".join(sentence_text(8, k) for k in range(12))
    parts = chunk_text(text, max_tokens=120, overlap_tokens=20)
    assert len(parts) > 3
    assert all(count_tokens(p) <= 120 for p in parts)


def test_consecutive_chunks_overlap():
    text = "\n\n".join(sentence_text(8, k) for k in range(6))
    parts = chunk_text(text, max_tokens=120, overlap_tokens=20)
    for prev, nxt in zip(parts, parts[1:]):
        tail_words = prev.split()[-20:]
        assert " ".join(tail_words) in nxt


def test_no_overlap_when_disabled():
    text = "\n\n".join(sentence_text(8, k) for k in range(6))
    parts = chunk_text(text, max_tokens=120, overlap_tokens=0)
    assert len(parts) > 1
    assert " ".join(parts[0].split()[-10:]) not in parts[1]


def test_code_block_kept_whole_when_it_fits():
    code = "```bash\n" + "\n".join(f"echo line{i}" for i in range(10)) + "\n\nsudo reboot\n```"
    text = sentence_text(10) + "\n\n" + code + "\n\n" + sentence_text(10)
    parts = chunk_text(text, max_tokens=200, overlap_tokens=0)
    holder = [p for p in parts if "echo line0" in p]
    assert len(holder) == 1 and "sudo reboot" in holder[0]     # blank line inside fence didn't split it


def test_oversized_paragraph_without_sentences_is_hard_split():
    blob = " ".join(f"word{i}" for i in range(2000))
    parts = chunk_text(blob, max_tokens=100, overlap_tokens=0)
    assert len(parts) > 5 and all(count_tokens(p) <= 100 for p in parts)
    assert " ".join(parts).split() == blob.split()             # nothing lost


def test_oversized_code_block_split_on_lines():
    code = "```\n" + "\n".join(f"line {i} of a very long script" for i in range(300)) + "\n```"
    parts = chunk_text(code, max_tokens=100, overlap_tokens=0)
    assert len(parts) > 3 and all(count_tokens(p) <= 100 for p in parts)


def test_tiny_following_section_is_absorbed():
    big = sentence_text(10)
    chunks = chunk_document(doc(
        (["A"], big, None),
        (["A", "Note"], "Short note.", None),
    ), max_tokens=400, min_tokens=60)
    assert len(chunks) == 1
    assert "Note\nShort note." in chunks[0].text


def test_tiny_section_not_absorbed_when_it_would_overflow():
    chunks = chunk_document(doc(
        (["A"], sentence_text(30), None),
        (["B"], "Short note.", None),
    ), max_tokens=302, min_tokens=60)      # 300-token section + 4-token note > 302
    assert [c.heading_path for c in chunks] == [["A"], ["B"]]


def test_page_numbers_preserved_and_not_merged_across_pages():
    chunks = chunk_document(doc(
        ([], sentence_text(10), 1),
        ([], "tiny page", 2),
    ))
    assert [c.page for c in chunks] == [1, 2]


def test_indices_are_contiguous_and_token_counts_correct():
    chunks = chunk_document(doc(
        (["A"], sentence_text(60), None), (["B"], sentence_text(60), None),
    ), max_tokens=150, overlap_tokens=15)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count == count_tokens(c.text) for c in chunks)


def test_context_header_and_embed_text():
    c = Chunk(0, ["Runner", "Install"], "Run it.", 3)
    assert c.context_header("Guide") == "Guide > Runner > Install"
    assert c.context_header("Runner") == "Runner > Install"     # no duplicated title
    assert c.embed_text("Guide").startswith("Guide > Runner > Install\n\nRun it.")


def test_deterministic():
    d = doc((["A"], sentence_text(60), None))
    assert [c.text for c in chunk_document(d)] == [c.text for c in chunk_document(d)]


def test_token_heavy_blocks_never_exceed_the_ceiling():
    """Regression: the word-count fallback split gave ~600-token chunks against a 400 limit
    when the 'words' were URLs and table rows (many tokens per whitespace-delimited word)."""
    from ingestion.chunking import chunk_text, count_tokens

    row = "|https://gitlab.com/gitlab-org/support/team/-/issues/12345?a=1&b=2|x.y.z/q|"
    table = " ".join(row for _ in range(120))                 # one block, no sentence breaks
    blob = "A" * 5000                                          # unbreakable single word
    for text in (table, blob, f"{table}\n\n{blob}"):
        chunks = chunk_text(text, max_tokens=320, overlap_tokens=50)
        assert chunks and max(count_tokens(c) for c in chunks) <= 320


def test_long_identifiers_cost_more_than_plain_words():
    from ingestion.chunking import count_tokens

    assert count_tokens("the quick brown fox") == 4
    assert count_tokens("a" * 64) >= 12                    # hash / encoded blob is many WordPieces


def test_hash_heavy_text_is_split_by_real_cost():
    from ingestion.chunking import chunk_text, count_tokens

    text = " ".join(f"{i:040x}" for i in range(400))      # 400 forty-char ids
    chunks = chunk_text(text, max_tokens=320, overlap_tokens=0)
    assert len(chunks) > 10 and max(count_tokens(c) for c in chunks) <= 320


def test_absorbing_a_tiny_section_never_exceeds_the_ceiling():
    from ingestion.chunking import chunk_document
    from ingestion.parsing import ParsedDocument, Section

    big = " ".join(["word"] * 95)
    doc = ParsedDocument(title="T", sections=[
        Section(heading_path=["T", "A"], text=big),
        Section(heading_path=["T", "A very long heading label here"], text="tiny " * 4),
    ])
    for c in chunk_document(doc, max_tokens=100, overlap_tokens=0, min_tokens=60):
        assert c.token_count <= 100
