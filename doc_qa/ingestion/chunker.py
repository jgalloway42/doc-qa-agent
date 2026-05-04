from pathlib import Path

from doc_qa.store.base import Chunk

ParsedPage = tuple[str, int]


def chunk_pages(
    pages: list[ParsedPage],
    filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_chars: int = 100,
) -> list[Chunk]:
    """
    Concatenate all page text with page-break markers, then apply a sliding
    window with overlap to produce Chunk objects.

    chunk_size and chunk_overlap are in characters (not tokens).
    chunk_index is 0-based and sequential across accepted chunks.
    page_or_line is set to the page/line of the first character in each chunk.
    chunk_id is f"{filename}::{chunk_index:04d}".
    """
    if not pages:
        return []

    # Build full text and track which character positions belong to which page
    parts: list[str] = []
    intervals: list[tuple[int, int, int]] = []  # (start, end, page_or_line)
    current_pos = 0

    for text, page_num in pages:
        if parts:
            current_pos += 2  # "\n\n" separator
        start = current_pos
        end = current_pos + len(text)
        intervals.append((start, end, page_num))
        parts.append(text)
        current_pos = end

    full_text = "\n\n".join(parts)

    step = max(1, chunk_size - chunk_overlap)
    base_name = Path(filename).name
    chunks: list[Chunk] = []
    chunk_index = 0
    pos = 0

    while pos < len(full_text):
        window = full_text[pos : pos + chunk_size]
        stripped = window.strip()
        if len(stripped) >= min_chunk_chars:
            chunks.append(
                Chunk(
                    chunk_id=f"{base_name}::{chunk_index:04d}",
                    text=stripped,
                    filename=base_name,
                    page_or_line=_page_at(pos, intervals),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        pos += step

    return chunks


def _page_at(pos: int, intervals: list[tuple[int, int, int]]) -> int:
    """Return the page/line number for the given character position."""
    for start, end, page_num in intervals:
        if start <= pos < end:
            return page_num
    # pos falls in a separator or past the end — use the closest preceding page
    best = intervals[0][2]
    for start, _, page_num in intervals:
        if start <= pos:
            best = page_num
    return best
