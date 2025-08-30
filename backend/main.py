# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
from uuid import uuid4
from typing import Dict, Any

from backend.ingest import extract_pdf, build_corpus
from backend.retriever import Retriever
from backend.llm import answer_with_llm

load_dotenv()

app = FastAPI(title="RAG Multimodal (PDF)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

DATA = Path("backend/store")
ASSETS = Path("backend/assets")
DATA.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# Sirve las imágenes extraídas
app.mount("/images", StaticFiles(directory=str(ASSETS)), name="images")

# Estado global simple (memoria del proceso)
RET: Retriever | None = None
CORPUS = []
JOBS: Dict[str, Dict[str, Any]] = {}   # job_id -> {status, detail, result|error}


@app.get("/")
def root():
    return {"ok": True, "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok"}


def _process_pdf(pdf_path: Path, job_id: str):
    """Trabajo pesado: extraer, indexar y dejar listo el retriever."""
    global RET, CORPUS
    try:
        JOBS[job_id] = {"status": "running", "detail": "Parsing PDF…"}
        parsed = extract_pdf(str(pdf_path), str(ASSETS))

        JOBS[job_id] = {"status": "running", "detail": "Building corpus…"}
        CORPUS = build_corpus(parsed)

        # Crea (o reutiliza) el retriever y construye índices (texto + imágenes)
        if RET is None:
            RET = Retriever(str(ASSETS))
        JOBS[job_id] = {"status": "running", "detail": "Indexing embeddings…"}
        RET.build(CORPUS)

        pages = len({t["page"] for t in parsed["text"]})
        result = {"pages": pages, "images": len(parsed["images"])}
        JOBS[job_id] = {"status": "done", "result": result}
    except Exception as e:
        JOBS[job_id] = {"status": "error", "error": str(e)}


@app.post("/ingest")
async def ingest(background: BackgroundTasks, pdf: UploadFile = File(...)):
    """
    Sube y procesa el PDF en segundo plano para evitar timeouts de Railway.
    Devuelve un job_id para consultar /status.
    """
    job_id = uuid4().hex
    pdf_path = DATA / pdf.filename
    pdf_path.write_bytes(await pdf.read())

    JOBS[job_id] = {"status": "queued", "detail": "Scheduled"}
    background.add_task(_process_pdf, pdf_path, job_id)

    return {
        "status": "processing",
        "job_id": job_id,
        "note": "Consulta /status?job_id=... para ver progreso",
    }


@app.get("/status")
def status(job_id: str):
    """Consulta el progreso de /ingest (queued|running|done|error)."""
    data = JOBS.get(job_id)
    if not data:
        return {"status": "unknown", "error": "job_id no encontrado"}
    return data
    
@app.get("/debug/images")
def debug_images():
    global RET
    if RET is None:
        return {"error": "Retriever no inicializado"}
    return {
        "indexed_images": len(RET.image_meta),
        "files": [im["file"] for im in RET.image_meta]
    }


@app.post("/query")
async def query(q: str = Form(...), k_text: int = Form(5), k_img: int = Form(3)):
    """
    Busca pasajes e imágenes relevantes y responde con Groq + citas.
    """
    global RET
    if RET is None:
        return {"error": "Primero procesa un PDF en /ingest"}

    # Limita rangos para evitar abusos
    k_text = max(1, min(int(k_text), 10))
    k_img = max(0, min(int(k_img), 6))

    res = RET.search(q, k_text=k_text, k_img=k_img)
    text_hits = res.get("text", [])
    img_hits = res.get("images", [])

    contexts = [r["content"] for r in text_hits if r.get("content")]
    answer = (
        answer_with_llm(q, contexts)
        if contexts
        else "No encontré pasajes de texto relevantes en el documento para responder."
    )

    imgs = [
        {"page": i["page"], "url": f"/images/{i['file']}", "score": i["score"]}
        for i in img_hits
    ]
    cites = [
        {"page": t["page"], "preview": t["content"][:140] + "…"} for t in text_hits
    ]

    return {"answer": answer, "citations": cites, "images": imgs}

