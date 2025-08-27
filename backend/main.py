from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv
from backend.ingest import extract_pdf, build_corpus
from backend.retriever import Retriever
from backend.llm import answer_with_llm
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG Multimodal (PDF)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

load_dotenv()  # lee .env en la raíz del proyecto

DATA = Path("backend/store")
ASSETS = Path("backend/assets")
DATA.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# Servir imágenes extraídas del PDF
app.mount("/images", StaticFiles(directory=str(ASSETS)), name="images")

RET = None
CORPUS = []

@app.post("/ingest")
async def ingest(pdf: UploadFile = File(...)):
    """
    Sube y procesa el PDF: extrae texto e imágenes, crea índices.
    """
    global RET, CORPUS
    pdf_path = DATA / pdf.filename
    pdf_bytes = await pdf.read()
    pdf_path.write_bytes(pdf_bytes)

    parsed = extract_pdf(str(pdf_path), str(ASSETS))
    CORPUS = build_corpus(parsed)

    RET = Retriever(str(ASSETS))
    RET.build(CORPUS)

    pages = len({t["page"] for t in parsed["text"]})
    return {"pages": pages, "images": len(parsed["images"])}

@app.post("/query")
async def query(q: str = Form(...), k_text: int = Form(5), k_img: int = Form(3)):
    """
    Busca pasajes e imágenes relevantes y responde con Groq + citas.
    """
    if RET is None:
        return {"error": "Primero ingesta el PDF en /ingest"}

    res = RET.search(q, k_text=k_text, k_img=k_img)
    contexts = [r["content"] for r in res["text"]]
    answer = answer_with_llm(q, contexts)

    imgs = [{"page":i["page"], "url": f"/images/{i['file']}", "score": i["score"]} for i in res["images"]]
    cites = [{"page":t["page"], "preview": t["content"][:140]+"…"} for t in res["text"]]

    return {"answer": answer, "citations": cites, "images": imgs}
