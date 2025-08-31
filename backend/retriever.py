import os
from pathlib import Path

# Nunca enviar tokens de HF por error
for k in [
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
]:
    os.environ.pop(k, None)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# 🔹 Forzar HuggingFace a usar /tmp como caché (gratis en Railway)
CACHE_DIR = "/tmp/hf_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch

ENABLE_IMAGES = os.getenv("ENABLE_IMAGES", "1").lower() not in ("0", "false", "no")

# Carga perezosa de CLIP (solo si ENABLE_IMAGES=1)
if ENABLE_IMAGES:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image


class Retriever:
    def __init__(self, assets_dir: str):
        self.assets = Path(assets_dir)

        # ---- Embeddings de texto (primero local, luego online) ----
        model_local_path = Path("backend/models/all-MiniLM-L6-v2")
        try:
            if model_local_path.exists():
                print(f"[INFO] Cargando modelo local desde {model_local_path}")
                self.text_model = SentenceTransformer(str(model_local_path))
            else:
                print("[INFO] Cargando modelo desde HuggingFace Hub")
                self.text_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder=CACHE_DIR,
                    token=None,
                )
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo: {e}")
            raise

        self.text_index = None
        self.text_meta = []
        self.bm25 = None

        # ---- CLIP para imágenes (solo si está activado) ----
        self.use_images = ENABLE_IMAGES
        if self.use_images:
            try:
                self.clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    cache_dir=CACHE_DIR,
                    token=None,
                )
                self.clip_proc = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    cache_dir=CACHE_DIR,
                    token=None,
                )
                self.image_index = None
                self.image_meta = []
            except Exception as e:
                print(f"[WARN] No se pudo cargar CLIP: {e}")
                self.use_images = False
                self.image_index = None
                self.image_meta = []
        else:
            self.image_index = None
            self.image_meta = []

    def _embed_text(self, texts):
        embs = self.text_model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embs.astype("float32")

    # Métodos de imagen solo si está activado
    if ENABLE_IMAGES:
        def _embed_image_paths(self, paths):
            images = [Image.open(self.assets / p).convert("RGB") for p in paths]
            inputs = self.clip_proc(images=images, return_tensors="pt")
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype("float32")

        def _embed_text_clip(self, texts):
            inputs = self.clip_proc(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            with torch.no_grad():
                feats = self.clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            return feats.detach().cpu().numpy().astype("float32")

    def build(self, corpus):
        # Texto
        text_chunks = [c for c in corpus if c["type"] == "text" and c.get("content", "").strip()]
        texts = [c["content"] for c in text_chunks]
        if texts:
            X = self._embed_text(texts)
            self.text_index = faiss.IndexFlatIP(X.shape[1])
            self.text_index.add(X)
            self.text_meta = text_chunks
            self.bm25 = BM25Okapi([t.split() for t in texts])

        # Imágenes
        if self.use_images:
            images = [c for c in corpus if c["type"] == "image"]
            if images:
                img_paths = [c["file"] for c in images]
                XI = self._embed_image_paths(img_paths)
                self.image_index = faiss.IndexFlatIP(XI.shape[1])
                self.image_index.add(XI)
                self.image_meta = images

    def search(self, query: str, k_text=5, k_img=3):
        out = {"text": [], "images": []}

        # Búsqueda vectorial de texto
        if self.text_index is not None:
            q = self._embed_text([query])
            D, I = self.text_index.search(q, k_text)
            for d, idx in zip(D[0], I[0]):
                out["text"].append({"score": float(d), **self.text_meta[idx]})

        # BM25 para recall lexical
        if self.bm25 is not None:
            bm = self.bm25.get_top_n(query.split(), [t["content"] for t in self.text_meta], n=3)
            for b in bm:
                i = [t["content"] for t in self.text_meta].index(b)
                out["text"].append({"score": 0.0, **self.text_meta[i], "bm25": True})

        # CLIP texto→imagen
        if self.use_images and self.image_index is not None and k_img > 0:
            tq = self._embed_text_clip([query])
            D, I = self.image_index.search(tq, k_img)
            for d, idx in zip(D[0], I[0]):
                out["images"].append({"score": float(d), **self.image_meta[idx]})

        # Dedupe + limitar
        seen, deduped = set(), []
        for t in out["text"]:
            key = (t["page"], t["content"])
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        out["text"] = deduped[:k_text]
        return out
