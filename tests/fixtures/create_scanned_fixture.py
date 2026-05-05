"""
Generate tests/fixtures/scanned_promissory_note.pdf — an image-only PDF
with zero extractable text layer. Used to trigger OCR fallback in test_parsers.py.

Usage:
    python tests/fixtures/create_scanned_fixture.py
"""

import importlib
import io
import random
import subprocess
import sys
from pathlib import Path


def _ensure_deps() -> None:
    missing = []
    for pkg in ("PIL", "reportlab", "numpy"):
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append("Pillow" if pkg == "PIL" else pkg)
    if missing:
        print(f"Installing missing deps: {missing}")
        # Try pip first; fall back to uv pip if pip module is absent
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install"] + missing, capture_output=True
        )
        if result.returncode != 0:
            subprocess.check_call(["uv", "pip", "install"] + missing)


_ensure_deps()

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

OUTPUT_PATH = Path(__file__).parent / "scanned_promissory_note.pdf"

PAGE_W, PAGE_H = 850, 1100
BG_COLOR = (245, 243, 238)
TEXT_COLOR = (0, 0, 0)
LEFT_MARGIN = 72
TOP_MARGIN = 80
LINE_HEIGHT = 22
FONT_SIZE = 14

PAGE_1_LINES = [
    "PROMISSORY NOTE",
    "",
    "Date: January 1, 2025",
    "Loan Amount: $150,000.00",
    "Interest Rate: 6.75% per annum",
    "",
    "Borrower: Test Borrower",
    "Address: 123 Main Street, Springfield, IL 62701",
    "",
    "Lender: Meridian Bank of Springfield N.A.",
    "Address: 456 Commerce Avenue, Springfield, IL 62702",
    "",
    "PROMISE TO PAY",
    "",
    "For value received, the undersigned Borrower hereby promises to pay to the",
    'order of Meridian Bank of Springfield N.A. ("Lender"), the principal sum of',
    "ONE HUNDRED FIFTY THOUSAND AND 00/100 DOLLARS ($150,000.00), together",
    "with interest on the unpaid principal balance from the date of this Note,",
    "until paid, at the annual rate of 6.75%.",
    "",
    "PAYMENT SCHEDULE",
    "",
    "Borrower shall make monthly payments of principal and interest beginning",
    "February 1, 2025, and continuing on the first day of each month thereafter",
    "until the Note is paid in full. The monthly payment amount is $972.90.",
    "",
    "Payments are due on the first (1st) day of each month.",
    "A late charge of 5% will be assessed on payments received after the 15th.",
    "",
    "MATURITY DATE",
    "",
    "The entire unpaid principal balance, together with all accrued and unpaid",
    'interest, shall be due and payable on January 1, 2055 ("Maturity Date").',
    "",
    "PREPAYMENT",
    "",
    "Borrower may prepay this Note in full or in part at any time without penalty.",
    "Partial prepayments shall not postpone the due date of any subsequent monthly",
    "payment unless Lender agrees in writing.",
]

PAGE_2_LINES = [
    "PROMISSORY NOTE (continued)",
    "",
    "ACCELERATION",
    "",
    "If Borrower fails to make any payment when due, or breaches any covenant",
    "herein, Lender may declare the entire unpaid principal balance, together",
    "with accrued interest, immediately due and payable.",
    "",
    "GOVERNING LAW",
    "",
    "This Note shall be governed by and construed in accordance with the laws",
    "of the State of Illinois, without regard to its conflict of law provisions.",
    "",
    "WAIVER",
    "",
    "Borrower waives presentment, demand, protest, and notice of dishonor.",
    "No failure by Lender to exercise any right hereunder shall constitute a",
    "waiver of such right.",
    "",
    "ENTIRE AGREEMENT",
    "",
    "This Note constitutes the entire agreement between Borrower and Lender",
    "with respect to the subject matter hereof and supersedes all prior",
    "negotiations, representations, warranties, and undertakings.",
    "",
    "",
    "SIGNATURES",
    "",
    "Borrower:",
    "",
    "_______________________________          Date: _______________",
    "Test Borrower",
    "",
    "",
    "Lender:",
    "",
    "_______________________________          Date: _______________",
    "Meridian Bank of Springfield N.A.",
    "Authorized Representative",
    "",
    "",
    "Loan Number: MB-2025-150000-001",
    "Note Date: January 1, 2025",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _render_page(lines: list[str], seed: int) -> Image.Image:
    rng = random.Random(seed)
    image = Image.new("RGB", (PAGE_W, PAGE_H), BG_COLOR)
    draw = ImageDraw.Draw(image)
    font = _load_font(FONT_SIZE)

    y = TOP_MARGIN
    for line in lines:
        draw.text((LEFT_MARGIN, y), line, font=font, fill=TEXT_COLOR)
        y += LINE_HEIGHT

    # Slight Gaussian blur to simulate scan softness
    image = image.filter(ImageFilter.GaussianBlur(radius=0.4))

    # Low-level random noise
    arr = np.array(image, dtype=np.int16)
    noise = np.random.default_rng(seed).normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr)

    # Slight random rotation per page
    angle = rng.uniform(-0.3, 0.3)
    image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=BG_COLOR)

    return image


def create_fixture(output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pages = [
        _render_page(PAGE_1_LINES, seed=42),
        _render_page(PAGE_2_LINES, seed=43),
    ]

    # Points per pixel at 72 DPI (1:1 mapping)
    page_w_pts = PAGE_W
    page_h_pts = PAGE_H

    c = canvas.Canvas(str(output_path), pagesize=(page_w_pts, page_h_pts))
    for page_img in pages:
        buf = io.BytesIO()
        page_img.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=page_w_pts, height=page_h_pts)
        c.showPage()
    c.save()

    _verify(output_path)


def _verify(path: Path) -> None:
    try:
        import pypdf
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
        import pypdf

    reader = pypdf.PdfReader(str(path))
    total_chars = sum(len(page.extract_text().strip()) for page in reader.pages)
    if total_chars > 0:
        print(f"WARNING: {total_chars} extractable chars found — text layer may be present")
    else:
        print(
            f"✓ {path.name} created — {len(reader.pages)} pages, "
            f"0 extractable chars (OCR fixture ready)"
        )


if __name__ == "__main__":
    create_fixture()
