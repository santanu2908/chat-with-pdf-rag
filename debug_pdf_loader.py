from app.pdf_loader import load_pdf

# Put a PDF in data/test.pdf, then run:
#   uv run python debug_pdf_loader.py

with open("data/test.pdf", "rb") as f:
    file_bytes = f.read()
    print(file_bytes[:100])  # Show the first 100 bytes to confirm we read the file

pages = load_pdf(file_bytes)

print(f"Total pages with text: {len(pages)}\n")
for text, page_num in pages:
    print(f"--- Page {page_num} ---")
    print(text[:200])
    print()
