import re

from doc_qa.ingestion.chunker import chunk_pages


def _pages(text: str, page_num: int = 1) -> list[tuple[str, int]]:
    return [(text, page_num)]


class TestChunkCount:
    def test_chunk_produces_correct_count(self):
        # 1000-char text, chunk_size=200, overlap=20 → step=180
        # windows start at: 0, 180, 360, 540, 720, 900 → 6 windows
        # last window = full_text[900:1100] = 100 chars (≥ min_chunk_chars=100)
        text = "A" * 1000
        chunks = chunk_pages(_pages(text), "doc.txt", chunk_size=200, chunk_overlap=20, min_chunk_chars=100)
        assert len(chunks) == 6

    def test_empty_pages_returns_empty(self):
        assert chunk_pages([], "doc.txt") == []

    def test_single_short_page_below_min_returns_empty(self):
        chunks = chunk_pages(_pages("Hi"), "doc.txt", chunk_size=512, chunk_overlap=64, min_chunk_chars=100)
        assert chunks == []

    def test_single_page_at_min_length_included(self):
        text = "X" * 100
        chunks = chunk_pages(_pages(text), "doc.txt", chunk_size=512, chunk_overlap=64, min_chunk_chars=100)
        assert len(chunks) == 1


class TestChunkIds:
    def test_chunk_ids_are_unique(self):
        text = "Word " * 300  # ~1500 chars
        chunks = chunk_pages(_pages(text), "doc.txt", chunk_size=200, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_format(self):
        text = "A" * 600
        chunks = chunk_pages(_pages(text), "path/to/my_doc.pdf", chunk_size=300, chunk_overlap=30, min_chunk_chars=10)
        pattern = re.compile(r"^my_doc\.pdf::\d{4}$")
        for chunk in chunks:
            assert pattern.match(chunk.chunk_id), f"Bad chunk_id: {chunk.chunk_id}"

    def test_chunk_id_uses_basename_not_full_path(self):
        text = "A" * 300
        chunks = chunk_pages(_pages(text), "some/nested/dir/report.pdf", chunk_size=300, chunk_overlap=10, min_chunk_chars=10)
        assert all(c.chunk_id.startswith("report.pdf::") for c in chunks)

    def test_chunk_index_is_sequential(self):
        text = "B" * 1000
        chunks = chunk_pages(_pages(text), "doc.txt", chunk_size=200, chunk_overlap=20)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


class TestChunkOverlap:
    def test_chunk_overlap_content(self):
        # Chunks should share the overlapping region
        text = "A" * 500
        chunks = chunk_pages(_pages(text), "doc.txt", chunk_size=100, chunk_overlap=20, min_chunk_chars=10)
        assert len(chunks) >= 2
        # chunk[0] ends at text[100], chunk[1] starts at text[80]
        # so chunk[0][-20:] == chunk[1][:20]
        assert chunks[0].text[-20:] == chunks[1].text[:20]


class TestShortChunkDropping:
    def test_chunk_drops_short_chunks(self):
        # Make a text where the last window is very short
        # chunk_size=100, overlap=0, text=105 chars
        # windows: [0:100] (100 chars), [100:200] (5 chars — dropped)
        text = "A" * 105
        chunks = chunk_pages(_pages(text), "doc.txt", chunk_size=100, chunk_overlap=0, min_chunk_chars=100)
        assert len(chunks) == 1
        assert len(chunks[0].text) == 100

    def test_min_chunk_chars_boundary(self):
        # Exactly min_chunk_chars should be included
        text = "A" * 50
        chunks = chunk_pages(_pages(text), "doc.txt", chunk_size=512, chunk_overlap=64, min_chunk_chars=50)
        assert len(chunks) == 1


class TestPageAssignment:
    def test_chunk_page_assignment_first_page(self):
        pages = [
            ("A" * 300, 1),
            ("B" * 300, 2),
        ]
        chunks = chunk_pages(pages, "doc.pdf", chunk_size=200, chunk_overlap=0, min_chunk_chars=10)
        # First chunk starts at pos 0 → page 1
        assert chunks[0].page_or_line == 1

    def test_chunk_page_assignment_second_page(self):
        pages = [
            ("A" * 100, 1),
            ("B" * 300, 2),
        ]
        # full_text = "AAA...AAA\n\nBBB...BBB" → page 2 starts at 102
        chunks = chunk_pages(pages, "doc.pdf", chunk_size=200, chunk_overlap=0, min_chunk_chars=10)
        # A chunk whose start position is ≥ 102 should get page 2
        page2_chunks = [c for c in chunks if c.page_or_line == 2]
        assert len(page2_chunks) >= 1

    def test_chunk_spanning_pages_gets_first_page(self):
        # page 1 is 150 chars, chunk_size=200 → first chunk spans both pages
        pages = [
            ("A" * 150, 1),
            ("B" * 150, 2),
        ]
        chunks = chunk_pages(pages, "doc.pdf", chunk_size=200, chunk_overlap=0, min_chunk_chars=10)
        # First chunk starts at pos 0 → page 1
        assert chunks[0].page_or_line == 1

    def test_filename_stored_as_basename(self):
        text = "A" * 300
        chunks = chunk_pages(_pages(text), "nested/path/report.pdf", chunk_size=300, chunk_overlap=0, min_chunk_chars=10)
        assert all(c.filename == "report.pdf" for c in chunks)
