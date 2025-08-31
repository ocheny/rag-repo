from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import re
import pytesseract
from typing import Dict, List


def _clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def extract_pdf(pdf_path: str, out_img_dir: str, debug: bool = True) -> Dict[str, List[dict]]:
    """
    Extrae TEXTO (con OCR si es necesario) + IMÁGENES embebidas del PDF.
    Devuelve {"text":[{page,text}], "images":[{page,file}]}.
    """
    doc = fitz.open(pdf_path)
    out = {"text": [], "images": []}
    out_dir = Path(out_img_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pno, page in enumerate(doc, start=1):
        # --- TEXTO ---
        text = _clean_text(page.get_text("text") or "")

        # fallback por bloques si está vacío
        if not text.strip():
            blocks = page.get_text("blocks") or []
            txt_blocks = []
            for b in blocks:
                if len(b) >= 5 and isinstance(b[4], str):
                    txt_blocks.append(b[4])
            text = _clean_text("\n".join(txt_blocks))

        # OCR si sigue vacío
        if not text.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            try:
                text = _clean_text(pytesseract.image_to_string(img, lang="spa+eng"))
                if debug:
                    print(f"[OCR] Página {pno}: {len(text)} caracteres extraídos")
            except Exception as e:
                print(f"[WARN] OCR falló en página {pno}: {e}")
                text = ""

        if debug:
            preview = text[:200].replace("\n", " ") + ("…" if len(text) > 200 else "")
            print(f"[DEBUG] Página {pno}: '{preview}'")

        out["text"].append({"page": pno, "text": text or ""})

        # --- IMÁGENES embebidas ---
        for idx, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)  # elimina canal alpha
                img_bytes = pix.tobytes("png")
                fname = f"page{pno}_img{idx+1}.png"
                (out_dir / fname).write_bytes(img_bytes)
                # Validación mínima
                Image.open(io.BytesIO(img_bytes)).verify()
                out["images"].append({"page": pno, "file": fname})
                if debug:
                    print(f"[DEBUG] Imagen extraída: {fname} (página {pno})")
            except Exception as e:
                print(f"[WARN] Imagen corrupta en página {pno}: {e}")

    doc.close()
    return out


def chunk_text(text: str, chunk_size=800, overlap=100):
    words = text.split()
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += step
    return chunks


def build_corpus(parsed: Dict[str, List[dict]]):
    corpus = []
    # Texto
    for item in parsed["text"]:
        for ch in chunk_text(item.get("text", "")):
            corpus.append({"type": "text", "page": item["page"], "content": ch})
    # Imágenes
    for im in parsed["images"]:
        corpus.append({"type": "image", "page": im["page"], "file": im["file"]})
    return corpus
