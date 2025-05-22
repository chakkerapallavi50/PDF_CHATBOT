from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
import os

def extract_text_with_page_numbers(file_path):
    reader = PdfReader(file_path)
    chunks = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        sentences = sent_tokenize(text)
        chunk = ""
        for sent in sentences:
            if len(chunk) + len(sent) <= 500:
                chunk += " " + sent
            else:
                chunks.append({
                    "content": chunk.strip(),
                    "source": os.path.basename(file_path),
                    "page": page_num + 1
                })
                chunk = sent
        if chunk:
            chunks.append({
                "content": chunk.strip(),
                "source": os.path.basename(file_path),
                "page": page_num + 1
            })
    return chunks

def process_all_pdfs(directory):
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            path = os.path.join(directory, filename)
            pdf_chunks = extract_text_with_page_numbers(path)
            all_chunks.extend(pdf_chunks)
    return all_chunks
