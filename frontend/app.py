import os
import gradio as gr
import requests

API = os.getenv("API", "https://rag-repo-production.up.railway.app")

def up(pdf):
    files = {"pdf": (pdf.name, open(pdf.name, "rb"), "application/pdf")}
    r = requests.post(f"{API}/ingest", files=files, timeout=120)
    return r.json()

def ask(q, k_text, k_img):
    data = {"q": q, "k_text": int(k_text), "k_img": int(k_img)}
    r = requests.post(f"{API}/query", data=data, timeout=120)
    js = r.json()
    if "error" in js:
        return js["error"], []
    answer = js.get("answer", "")
    cites = "\n".join([f"• p.{c['page']}: {c['preview']}" for c in js.get("citations", [])])
    imgs = [API + i["url"] for i in js.get("images", [])]
    return (answer + ("\n\nCitas:\n" + cites if cites else "")), imgs

with gr.Blocks(title="Chatbot RAG Multimodal") as demo:
    gr.Markdown("# Chatbot RAG Multimodal")
    with gr.Tab("1) Ingesta"):
        up_pdf = gr.File(label="Sube el PDF", file_types=[".pdf"])
        out_ing = gr.JSON()
        btn = gr.Button("Procesar")
        btn.click(up, inputs=up_pdf, outputs=out_ing)
    with gr.Tab("2) Consulta"):
        q = gr.Textbox(label="Pregunta")
        k1 = gr.Slider(1, 10, 5, step=1, label="Top-K texto")
        k2 = gr.Slider(0, 6, 3, step=1, label="Top-K imágenes")
        out_txt = gr.Textbox(label="Respuesta + citas")
        out_imgs = gr.Gallery(label="Imágenes relevantes", columns=3, height=300)
        ask_btn = gr.Button("Preguntar")
        ask_btn.click(ask, inputs=[q, k1, k2], outputs=[out_txt, out_imgs])


demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
