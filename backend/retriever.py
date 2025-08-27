import os
# --- Blindaje para que NUNCA se envíe un token HF por error ---
for k in [
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
]:
    os.environ.pop(k, None)

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import faiss, numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
import torch

CACHE_DIR = str(Path("backend/store/hf_cache").absolute())

class Retriever:
    def __init__(self, assets_dir: str):
        self.assets = Path(assets_dir)

        # ---- Embeddings de texto (forzar token=None y cache local) ----
        try:
            self.text_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=CACHE_DIR,  # cache local
                # Para versiones nuevas de sentence-transformers/transformers:
                # 'token' es respetado por transformers internamente
                token=None,
            )
        except Exception as e:
            # Reintento limpio
            for k in ["TRANSFORMERS_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE"]:
                os.environ.pop(k, None)
            self.text_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=CACHE_DIR,
                token=None,
            )

        self.text_index = None
        self.text_meta = []
        self.bm25 = None

        # ---- CLIP para imágenes (forzar token=None y cache local) ----
        try:
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=CACHE_DIR,
                token=None,
            )
            self.clip_proc  = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=CACHE_DIR,
                token=None,
            )
        except Exception:
            self.clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=CACHE_DIR,
                token=None,
            )
            self.clip_proc  = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=CACHE_DIR,
                token=None,
            )

        self.image_index = None
        self.image_meta = []

    def _embed_text(self, texts):
        embs = self.text_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embs.astype("float32")

    def _embed_image_paths(self, paths):
        images = [Image.open(self.assets / p).convert("RGB") for p in paths]
        inputs = self.clip_proc(images=images, return_tensors="pt")
        with torch.no_grad():
            feats = self.clip_model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype("float32")

    def _embed_text_clip(self, texts):
        inputs = self.clip_proc(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            feats = self.clip_model.get_text_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype("float32")

    def build(self, corpus):
        # Texto
        text_chunks = [c for c in corpus if c["type"] == "text" and c["content"].strip()]
        texts = [c["content"] for c in text_chunks]
        if texts:
            X = self._embed_text(texts)
            self.text_index = faiss.IndexFlatIP(X.shape[1])
            self.text_index.add(X)
            self.text_meta = text_chunks
            self.bm25 = BM25Okapi([t.split() for t in texts])

        # Imágenes
        images = [c for c in corpus if c["type"] == "image"]
        if images:
            img_paths = [c["file"] for c in images]
            XI = self._embed_image_paths(img_paths)
            self.image_index = faiss.IndexFlatIP(XI.shape[1])
            self.image_index.add(XI)
            self.image_meta = images

    def search(self, query: str, k_text=5, k_img=3):
        out = {"text": [], "images": []}

        # Vectores texto
        if self.text_index is not None:
            q = self._embed_text([query])
            D, I = self.text_index.search(q, k_text)
            for d, idx in zip(D[0], I[0]):
                out["text"].append({"score": float(d), **self.text_meta[idx]})

        # BM25
        if self.bm25 is not None:
            bm = self.bm25.get_top_n(query.split(), [t["content"] for t in self.text_meta], n=3)
            for b in bm:
                i = [t["content"] for t in self.text_meta].index(b)
                out["text"].append({"score": 0.0, **self.text_meta[i], "bm25": True})

        # CLIP texto→imagen
        if self.image_index is not None:
            tq = self._embed_text_clip([query])
            D, I = self.image_index.search(tq, k_img)
            for d, idx in zip(D[0], I[0]):
                out["images"].append({"score": float(d), **self.image_meta[idx]})

        # dedupe
        seen, deduped = set(), []
        for t in out["text"]:
            key = (t["page"], t["content"])
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        out["text"] = deduped[:k_text]
        return out
