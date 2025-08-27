import os, time, requests, json
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
BACKEND = (os.getenv("BACKEND_URL") or os.getenv("API") or "http://127.0.0.1:8000").rstrip("/")
print("Frontend apuntando a BACKEND:", BACKEND)  # mira esto en los logs de Railway

def _to_json_or_text(resp: requests.Response):
    ct = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        return True, resp.json()
    # devuelve texto (o parte) si no es JSON
    body = resp.text
    return False, {"status": resp.status_code, "content_type": ct, "text": body[:800]}

def ping_backend():
    try:
        r = requests.get(f"{BACKEND}/", timeout=10)
        ok, payload = _to_json_or_text(r)
        return f"Ping: {r.status_code} | {payload}"
    except Exception as e:
        return f"Ping ERROR: {e}"

def ingest_pdf(file):
    if file is None:
        return "⚠️ Sube un PDF primero.", gr.update(visible=False)
    try:
        with open(file, "rb") as f:
            r = requests.post(f"{BACKEND}/ingest",
                              files={"pdf": (os.path.basename(file), f, "application/pdf")},
                              timeout=300)
        ok, payload = _to_json_or_text(r)
        if not r.ok or not ok:
            return f"❌ /ingest ERROR {r.status_code}: {payload}", gr.update(visible=False)

        # formato esperado: {"pages": X, "images": Y}  o job_id/queue
        if "job_id" in payload:
            job_id = payload["job_id"]
            for _ in range(120):
                st = requests.get(f"{BACKEND}/status", params={"job_id": job_id}, timeout=10)
                ok2, p2 = _to_json_or_text(st)
                if not st.ok or not ok2:
                    return f"❌ /status ERROR {st.status_code}: {p2}", gr.update(visible=False)
                if p2.get("status") == "done":
                    r2 = p2.get("result", {})
                    return f"✅ Ingestado: {r2.get('pages','?')} páginas, {r2.get('images','?')} imágenes.", gr.update(visible=True)
                if p2.get("status") == "error":
                    return f"❌ Error en ingest: {p2.get('error')}", gr.update(visible=False)
                time.sleep(1)
            return "⌛ Timeout consultando estado.", gr.update(visible=False)
        else:
            pages = payload.get("pages", "?"); images = payload.get("images", "?")
            return f"✅ Ingestado: {pages} páginas, {images} imágenes.", gr.update(visible=True)
    except Exception as e:
        return f"❌ Error subiendo/ingestando: {e}", gr.update(visible=False)

def ask_query(q, k_text, k_img):
    if not q or not q.strip():
        return "⚠️ Escribe una pregunta.", "", []
    try:
        r = requests.post(f"{BACKEND}/query",
                          data={"q": q, "k_text": int(k_text), "k_img": int(k_img)},
                          headers={"Content-Type": "application/x-www-form-urlencoded"},
                          timeout=120)
        ok, payload = _to_json_or_text(r)
        if not r.ok or not ok:
            # muestra el error crudo para depurar
            return f"❌ /query ERROR {r.status_code}: {payload}", "", []
        answer = payload.get("answer", "(sin respuesta)")
        cites = payload.get("citations", [])
        if cites:
            cites_txt = "\n".join([f"• p.{c['page']}: {c['preview']}" for c in cites])
        else:
            cites_txt = "(sin citas de texto relevantes)"
        imgs = payload.get("images", [])
        gallery = []
        for im in imgs:
            url_path = im.get("url", "")
            if not (url_path.startswith("http://") or url_path.startswith("https://")):
                url_path = f"{BACKEND}{url_path}"
            gallery.append((url_path, f"p.{im['page']} — score: {im['score']:.3f}"))
        return answer, cites_txt, gallery
    except Exception as e:
        return f"❌ Error en /query: {e}", "", []

with gr.Blocks(title="RAG Multimodal (PDF)") as demo:
    gr.Markdown(f"## RAG Multimodal (PDF)\nBackend: **{BACKEND}**")
    ping = gr.Markdown(ping_backend())
    with gr.Row():
        with gr.Column(scale=1):
            pdf = gr.File(label="Sube tu PDF", file_types=[".pdf"])
            ingest_btn = gr.Button("Ingestar PDF", variant="primary")
            ingest_msg = gr.Markdown("")
            query_box = gr.Textbox(label="Tu pregunta", placeholder="Ej: resume los objetivos…")
            with gr.Row():
                k_text = gr.Slider(1, 10, value=5, step=1, label="k_text")
                k_img = gr.Slider(0, 6, value=3, step=1, label="k_img")
            ask_btn = gr.Button("Preguntar", variant="secondary")
        with gr.Column(scale=1, visible=False) as results_col:
            answer = gr.Markdown(label="Respuesta")
            citations = gr.Textbox(label="Citas (texto)", lines=8)
            gallery = gr.Gallery(label="Imágenes relevantes", columns=2, height=400)
    ingest_btn.click(fn=ingest_pdf, inputs=[pdf], outputs=[ingest_msg, results_col])
    ask_btn.click(fn=ask_query, inputs=[query_box, k_text, k_img], outputs=[answer, citations, gallery])

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    demo.queue().launch(server_name="0.0.0.0", server_port=port, show_api=False)

