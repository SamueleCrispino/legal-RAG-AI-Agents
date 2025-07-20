import fitz  # PyMuPDF

def extract_text_pymupdf(pdf_path):
    """
    Extracts clean text from a PDF using PyMuPDF.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted clean text.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text("text") # "text" extracts plain text
            # You can also use "words" to get a list of words with coordinates,
            # or "blocks" for text blocks, which can be useful for more structured extraction.
        doc.close()
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {e}")
    return text