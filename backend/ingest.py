from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

def extract_pdf(pdf_path: str, out_img_dir: str):
    """
    Extrae texto por página e imágenes del PDF.
    Devuelve un dict con listas: text (page,text) e images (page,file).
    """
    doc = fitz.open(pdf_path)
    out = {"text": [], "images": []}
    Path(out_img_dir).mkdir(parents=True, exist_ok=True)

    for pno, page in enumerate(doc):
        # Texto plano por página
        text = page.get_text("text")
        out["text"].append({"page": pno + 1, "text": text})

        # Imágenes de la página
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.alpha:  # elimina canal alpha si existe
                pix = fitz.Pixmap(pix, 0)
            img_bytes = pix.tobytes("png")
            fname = f"page{pno+1}_img{idx+1}.png"
            (Path(out_img_dir) / fname).write_bytes(img_bytes)
            # Asegura que el archivo es válido
            Image.open(io.BytesIO(img_bytes)).verify()
            out["images"].append({"page": pno + 1, "file": fname})
    return out


def chunk_text(text: str, chunk_size=800, overlap=100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += max(1, chunk_size - overlap)
    return chunks


def build_corpus(parsed):
    corpus = []
    for item in parsed["text"]:
        for ch in chunk_text(item["text"]):
            corpus.append({"type": "text", "page": item["page"], "content": ch})
    for im in parsed["images"]:
        corpus.append({"type": "image", "page": im["page"], "file": im["file"]})
    return corpus
