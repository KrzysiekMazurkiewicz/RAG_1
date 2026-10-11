"""Budowa i wczytywanie indeksu: jednostki (JSONL) + BM25 (JSON) + embeddingi (NPY).

Katalog indeksu:
    jednostki.jsonl   wszystkie jednostki (także rodzice i uchylone – do rozszerzania)
    bm25.json         indeks leksykalny po jednostkach szukalnych
    emb.npy           macierz embeddingów (float32, znormalizowane) – opcjonalnie
    meta.json         który model, prefiksy, liczby
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np

from .bm25 import BM25
from .config import KONFIG
from .models import Jednostka, wczytaj_jsonl, zapisz_jsonl
from .tokenize import lematyzacja_dostepna, tokenizuj

log = logging.getLogger("lexrag")

_MODEL_EMB = None


def urzadzenie() -> str:
    """cpu | cuda wg LEXRAG_DEVICE; "auto" wybiera kartę NVIDIA, gdy torch ją widzi."""
    wybor = KONFIG.urzadzenie.strip().lower()
    if wybor and wybor != "auto":
        return wybor
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def argumenty_modelu() -> dict:
    """Na karcie wagi w fp16: o połowę mniej VRAM (reranker 1,1 GB zamiast 2,3 GB) – na współdzielonej
    karcie (Ollama + aplikacja) to różnica między stabilną pracą a wywrotką runnera z braku pamięci."""
    if urzadzenie() != "cuda":
        return {}
    try:
        import torch  # type: ignore
        return {"torch_dtype": torch.float16}
    except Exception:
        return {}


def model_embeddingow():
    """Leniwe ładowanie sentence-transformers; None, gdy niedostępne/wyłączone."""
    global _MODEL_EMB
    if KONFIG.emb_wylacz:
        return None
    if _MODEL_EMB is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            log.warning("sentence-transformers niezainstalowane – działa tylko BM25 "
                        "(pip install -r requirements-ml.txt).")
            return None
        try:
            _MODEL_EMB = SentenceTransformer(KONFIG.emb_model, device=urzadzenie(), model_kwargs=argumenty_modelu())
            log.info("Embeddingi: %s na %s", KONFIG.emb_model, urzadzenie())
        except Exception as e:  # brak modelu offline itp.
            log.warning("Nie udało się załadować modelu embeddingów %s: %s", KONFIG.emb_model, e)
            return None
    return _MODEL_EMB


def embedduj(teksty: list[str], prefiks: str) -> Optional[np.ndarray]:
    m = model_embeddingow()
    if m is None:
        return None
    wek = m.encode([prefiks + t for t in teksty], normalize_embeddings=True,
                   batch_size=16, show_progress_bar=len(teksty) > 64)
    return np.asarray(wek, dtype=np.float32)


class Indeks:
    def __init__(self, jednostki: list[Jednostka], bm25: BM25, szukalne: list[str],
                 emb: Optional[np.ndarray], meta: dict):
        self.jednostki = {j.id: j for j in jednostki}
        self.bm25 = bm25
        self.szukalne = szukalne            # kolejność == wiersze bm25 / emb
        self.emb = emb
        self.meta = meta
        self._pozycja = {ident: i for i, ident in enumerate(szukalne)}
        # indeks: termin zdefiniowany -> id definicji (do rozszerzania kontekstu)
        self.definicje: dict[str, str] = {}
        for j in jednostki:
            for t in j.terminy:
                self.definicje[t.lower()] = j.id

    def __len__(self) -> int:
        return len(self.szukalne)

    def jednostka(self, ident: str) -> Optional[Jednostka]:
        return self.jednostki.get(ident)

    # --- budowa ----------------------------------------------------------
    @classmethod
    def zbuduj(cls, pliki_jsonl: list[str], katalog: str, z_embeddingami: bool = True) -> "Indeks":
        jednostki: list[Jednostka] = []
        for p in pliki_jsonl:
            jednostki.extend(wczytaj_jsonl(p))
        if not jednostki:
            raise SystemExit("Brak jednostek do zindeksowania.")
        duplikaty = len(jednostki) - len({j.id for j in jednostki})
        if duplikaty:
            raise SystemExit(f"Zduplikowane id jednostek: {duplikaty} (użyj innych prefiksów --kod).")

        szukalne = [j.id for j in jednostki if j.szukalny and j.tekst.strip()]
        mapa = {j.id: j for j in jednostki}
        teksty = [mapa[i].tekst_do_indeksu() for i in szukalne]

        log.info("BM25: %d jednostek (lematyzacja: %s)", len(teksty), lematyzacja_dostepna())
        bm25 = BM25().dopasuj([tokenizuj(t) for t in teksty])

        emb = None
        if z_embeddingami:
            log.info("Embeddingi: %s", KONFIG.emb_model)
            emb = embedduj(teksty, KONFIG.emb_prefix_dokumentu)

        os.makedirs(katalog, exist_ok=True)
        zapisz_jsonl(jednostki, os.path.join(katalog, "jednostki.jsonl"))
        bm25.zapisz(os.path.join(katalog, "bm25.json"))
        meta = {
            "liczba_jednostek": len(jednostki), "liczba_szukalnych": len(szukalne),
            "szukalne": szukalne, "lematyzacja": lematyzacja_dostepna(),
            "emb_model": KONFIG.emb_model if emb is not None else None,
            "emb_prefix_zapytania": KONFIG.emb_prefix_zapytania,
            "emb_prefix_dokumentu": KONFIG.emb_prefix_dokumentu,
        }
        if emb is not None:
            np.save(os.path.join(katalog, "emb.npy"), emb)
        elif os.path.exists(os.path.join(katalog, "emb.npy")):
            os.remove(os.path.join(katalog, "emb.npy"))
        with open(os.path.join(katalog, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        return cls(jednostki, bm25, szukalne, emb, meta)

    @classmethod
    def wczytaj(cls, katalog: str) -> "Indeks":
        jednostki = wczytaj_jsonl(os.path.join(katalog, "jednostki.jsonl"))
        bm25 = BM25.wczytaj(os.path.join(katalog, "bm25.json"))
        with open(os.path.join(katalog, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        emb = None
        p = os.path.join(katalog, "emb.npy")
        if os.path.exists(p) and not KONFIG.emb_wylacz:
            emb = np.load(p)
        return cls(jednostki, bm25, meta["szukalne"], emb, meta)
