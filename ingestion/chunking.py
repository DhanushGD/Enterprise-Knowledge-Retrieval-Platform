"""Step 6: split parsed sections into retrieval-sized chunks.

Rules, in order of priority:
  1. Never cross a section boundary (a chunk belongs to exactly one heading path),
     except to absorb a tiny section into the previous chunk.
  2. Pack whole paragraphs / code blocks up to `max_tokens`.
  3. Oversized paragraphs split on sentences, then on words as a last resort.
  4. Consecutive chunks of one long section overlap by ~`overlap_tokens` words.

Token counts are an approximation (words + punctuation, long identifiers charged by length). That is close enough for
sizing and keeps the chunker free of model downloads; swap `count_tokens` for a real
tokenizer if you want exact budgets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ingestion.parsing import ParsedDocument

_TOKEN = re.compile(r"\w+|[^\w\s]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FENCE = re.compile(r"^\s*(```+|~~~+)")


def _word_cost(token: str) -> int:
    """Ordinary words and punctuation cost 1. WordPiece shreds long identifiers, hashes and
    encoded strings into many pieces, so a long run of letters/digits is charged by length."""
    n = len(token)
    if n <= 8:
        return 1
    if n <= 12:
        return 2
    return -(-n // 4)


def count_tokens(text: str) -> int:
    return sum(_word_cost(t) for t in _TOKEN.findall(text))


@dataclass
class Chunk:
    chunk_index: int
    heading_path: list[str]
    text: str
    token_count: int
    page: int | None = None

    @property
    def section(self) -> str:
        return " > ".join(self.heading_path)

    def context_header(self, doc_title: str) -> str:
        """'Title > Section > Subsection' - prepended to the text before embedding."""
        path = self.heading_path
        parts = path if path and path[0] == doc_title else [doc_title, *path]
        return " > ".join(parts)

    def embed_text(self, doc_title: str) -> str:
        return build_embed_text(doc_title, self.section, self.text)


def build_embed_text(doc_title: str, section: str, text: str) -> str:
    """What actually gets embedded: 'Title > Section' context header, then the text.

    The header means a chunk saying only "Run the installer." still embeds with
    "Runner > Install > Linux" attached, which is what makes it findable.
    """
    if not section:
        header = doc_title
    elif section == doc_title or section.startswith(f"{doc_title} > "):
        header = section
    else:
        header = f"{doc_title} > {section}"
    return f"{header}\n\n{text}"


def _blocks(text: str) -> list[str]:
    """Paragraph-level blocks. A fenced code block is always one block."""
    blocks, current, in_fence = [], [], False
    for line in text.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
            current.append(line)
            continue
        if not in_fence and not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


_MAX_WORD_CHARS = 200   # a single "word" longer than this (base64, minified JSON) is sliced


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last resort: split on words, but budget by *tokens*, not word count.

    One whitespace-delimited word (a URL, a table row, a path) can be dozens of tokens,
    so a fixed number of words per piece can blow far past `max_tokens`. Every piece
    returned here is <= max_tokens by `count_tokens`.
    """
    pieces, current, used = [], [], 0
    for word in text.split():
        # An unbreakable giant (base64 blob, one-line JSON) is sliced by characters.
        parts = [word[i:i + _MAX_WORD_CHARS] for i in range(0, len(word), _MAX_WORD_CHARS)]
        for part in parts:
            size = count_tokens(part)
            if current and used + size > max_tokens:
                pieces.append(" ".join(current))
                current, used = [], 0
            current.append(part)
            used += size
    if current:
        pieces.append(" ".join(current))
    return pieces


def _pack(units: list[str], max_tokens: int, sep: str) -> list[str]:
    """Greedy-pack units into groups <= max_tokens; hard-split any unit that is too big."""
    out, current, used = [], [], 0
    for unit in units:
        size = count_tokens(unit)
        if size > max_tokens:
            if current:
                out.append(sep.join(current))
                current, used = [], 0
            out.extend(_hard_split(unit, max_tokens))
            continue
        if current and used + size > max_tokens:
            out.append(sep.join(current))
            current, used = [], 0
        current.append(unit)
        used += size
    if current:
        out.append(sep.join(current))
    return out


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    if _FENCE.match(block.split("\n", 1)[0]):
        return _pack(block.split("\n"), max_tokens, "\n")       # code: split on lines
    sentences = _SENTENCE_SPLIT.split(block)
    if len(sentences) == 1:
        sentences = block.split("\n")                           # lists / tables: split on lines
    return _pack(sentences, max_tokens, " ")


def _tail(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    words = text.split()
    return " ".join(words[-overlap_tokens:]) if len(words) > overlap_tokens else ""


def chunk_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    pieces: list[str] = []
    for block in _blocks(text):
        if count_tokens(block) <= max_tokens:
            pieces.append(block)
        else:
            pieces.extend(_split_oversized(block, max_tokens))

    chunks, current, used = [], [], 0
    for piece in pieces:
        size = count_tokens(piece)
        if current and used + size > max_tokens:
            chunks.append("\n\n".join(current))
            tail = _tail(chunks[-1], overlap_tokens)
            if tail and count_tokens(tail) + size <= max_tokens:
                current, used = [tail], count_tokens(tail)
            else:
                current, used = [], 0
        current.append(piece)
        used += size
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(
    doc: ParsedDocument,
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    min_tokens: int = 60,
) -> list[Chunk]:
    candidates: list[Chunk] = []
    for section in doc.sections:
        for text in chunk_text(section.text, max_tokens, overlap_tokens):
            candidates.append(
                Chunk(0, list(section.heading_path), text, count_tokens(text), section.page)
            )

    # Absorb tiny sections into the previous chunk instead of indexing 20-word scraps.
    merged: list[Chunk] = []
    for cand in candidates:
        prev = merged[-1] if merged else None
        if (
            prev
            and cand.token_count < min_tokens
            and cand.heading_path != prev.heading_path
            and cand.page == prev.page
            and prev.token_count + cand.token_count <= max_tokens
        ):
            label = cand.heading_path[-1] if cand.heading_path else ""
            merged_text = prev.text + "\n\n" + (f"{label}\n" if label else "") + cand.text
            if count_tokens(merged_text) > max_tokens:     # the label itself must fit too
                merged.append(cand)
                continue
            prev.text = merged_text
            prev.token_count = count_tokens(prev.text)
        else:
            merged.append(cand)

    for index, chunk in enumerate(merged):
        chunk.chunk_index = index
    return merged
