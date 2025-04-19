import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    with fitz.open(stream=pdf_file, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text
