from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import re
from typing import Dict, List
import pytesseract


def _clean_text(t: str) -> str:
    # Limpieza suave para evitar basura y huecos
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def extract_pdf(pdf_path: str, out_img_dir: str, debug: bool = True) -> Dict[str, List[dict]]:
    """
    Extrae TEXTO + IMÁGENES del PDF.
    Devuelve {"text":[{page,content}], "images":[{page,file}]}
    - Usa OCR si no hay texto digital.
    - Guarda imágenes embebidas y también un render completo por página.
    """
    doc = fitz.open(pdf_path)
    out = {"text": [], "images": []}
    out_dir = Path(out_img_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pno, page in enumerate(doc, start=1):
        # === 1) Extraer texto ===
        text = _clean_text(page.get_text("text") or "")

        if not text:
            # fallback por bloques
            blocks = page.get_text("blocks") or []
            txt_blocks = []
            for b in blocks:
                if len(b) >= 5 and isinstance(b[4], str):
                    txt_blocks.append(b[4])
            text = _clean_text("\n".join(txt_blocks))

        if not text.strip():
            # fallback final con OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
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

        out["text"].append({"page": pno, "content": text or ""})

        # === 2) Guardar render completo de la página ===
        pix_full = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        fname_full = f"page{pno}.png"
        (out_dir / fname_full).write_bytes(pix_full.tobytes("png"))
        out["images"].append({"page": pno, "file": fname_full})

        # === 3) Guardar imágenes embebidas en la página ===
        for idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.alpha:  # elimina canal alpha si existe
                pix = fitz.Pixmap(pix, 0)
            img_bytes = pix.tobytes("png")
            fname = f"page{pno}_img{idx+1}.png"
            (out_dir / fname).write_bytes(img_bytes)
            # verificación rápida
            try:
                Image.open(io.BytesIO(img_bytes)).verify()
            except Exception:
                print(f"[WARN] Imagen corrupta en página {pno}")
            out["images"].append({"page": pno, "file": fname})

    doc.close()
    return out


def chunk_text(text: str, chunk_size=800, overlap=100):
    # Chunking por PALABRAS (robusto a idiomas); evita trozos vacíos
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
    # Primero texto
    for item in parsed["text"]:
        for ch in chunk_text(item.get("content", "")):
            corpus.append({"type": "text", "page": item["page"], "content": ch})
    # Luego imágenes
    for im in parsed["images"]:
        corpus.append({"type": "image", "page": im["page"], "file": im["file"]})
    return corpus
