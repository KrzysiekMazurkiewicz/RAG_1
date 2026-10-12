"""Wyszukiwanie: hybryda BM25 + embeddingi (RRF) → reranker → rozszerzanie kontekstu.

    szukaj()   – dla każdego wariantu zapytania: top-k z BM25 i top-k z wektorów,
                 scalone przez Reciprocal Rank Fusion; filtr na datę obowiązywania
    rerank()   – cross-encoder ocenia (pytanie, jednostka); bez modelu – kolejność z RRF
    rozszerz() – do trafień dokłada artykuł-rodzic, odesłania (1 skok) i definicje,
                 w granicach budżetu znaków (CPU: prefill promptu jest wąskim gardłem)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import KONFIG
from .index import BLOKADA_MODELI, Indeks, embedduj
from .models import Jednostka
from .tokenize import tokenizuj

log = logging.getLogger("lexrag")

_RERANKER = None


@dataclass
class Trafienie:
    id: str
    wynik: float
    zrodlo: str  # "rrf" | "rerank"


# --- hybryda ----------------------------------------------------------------
def _rrf(listy: list[list[str]], k: int) -> dict[str, float]:
    wynik: dict[str, float] = {}
    for lista in listy:
        for poz, ident in enumerate(lista):
            wynik[ident] = wynik.get(ident, 0.0) + 1.0 / (k + poz + 1)
    return wynik


def szukaj(indeks: Indeks, zapytania: list[str], k_kazde: Optional[int] = None,
           na_dzien: Optional[str] = None, uzyj_emb: bool = True,
           uzyj_bm25: bool = True) -> list[Trafienie]:
    k_kazde = k_kazde or KONFIG.k_kazde
    dozwolone = None
    if na_dzien:
        dozwolone = {i for i in indeks.szukalne if indeks.jednostki[i].obowiazuje(na_dzien)}

    def filtr(ids: list[str]) -> list[str]:
        return [i for i in ids if dozwolone is None or i in dozwolone]

    listy: list[list[str]] = []
    for z in zapytania:
        if uzyj_bm25:
            top = indeks.bm25.top(tokenizuj(z), k=k_kazde * 2)
            listy.append(filtr([indeks.szukalne[i] for i, _ in top])[:k_kazde])
        if uzyj_emb and indeks.emb is not None:
            q = embedduj([z], indeks.meta.get("emb_prefix_zapytania", KONFIG.emb_prefix_zapytania))
            if q is not None:
                sim = indeks.emb @ q[0]
                kolejnosc = np.argsort(-sim)[: k_kazde * 2]
                listy.append(filtr([indeks.szukalne[int(i)] for i in kolejnosc])[:k_kazde])
    scal = _rrf(listy, KONFIG.rrf_k)
    return [Trafienie(i, s, "rrf") for i, s in sorted(scal.items(), key=lambda x: -x[1])]


# --- reranker ---------------------------------------------------------------
def _reranker():
    global _RERANKER
    if KONFIG.reranker_wylacz:
        return None
    if _RERANKER is None:
        with BLOKADA_MODELI:
            if _RERANKER is None:               # drugi wątek mógł już załadować model
                try:
                    from sentence_transformers import CrossEncoder  # type: ignore
                    from .index import argumenty_modelu, urzadzenie
                    _RERANKER = CrossEncoder(KONFIG.reranker_model, device=urzadzenie(), max_length=512,
                                             model_kwargs=argumenty_modelu())
                    log.info("Reranker: %s na %s", KONFIG.reranker_model, urzadzenie())
                except Exception as e:
                    log.warning("Reranker niedostępny (%s) – zostaje kolejność z RRF.", e)
                    _RERANKER = False
    return _RERANKER or None


def rerank(indeks: Indeks, pytanie: str, kandydaci: list[Trafienie],
           top_n: Optional[int] = None, ile_ocenic: int = 50) -> list[Trafienie]:
    top_n = top_n or KONFIG.top_po_reranku
    kandydaci = kandydaci[:ile_ocenic]
    m = _reranker()
    if m is None or not kandydaci:
        return kandydaci[:top_n]
    pary = [(pytanie, indeks.jednostki[t.id].tekst_do_indeksu()[:2000]) for t in kandydaci]
    with BLOKADA_MODELI:                                  # jedno wywołanie modelu naraz (MPS nie jest wielowątkowe)
        oceny = m.predict(pary, batch_size=8, show_progress_bar=False)
    out = [Trafienie(t.id, float(o), "rerank") for t, o in zip(kandydaci, oceny)]
    return sorted(out, key=lambda x: -x.wynik)[:top_n]


# --- rozszerzanie kontekstu -------------------------------------------------
def rozszerz(indeks: Indeks, trafienia: list[Trafienie], max_znakow: Optional[int] = None,
             z_rodzicem: bool = True, z_odwolaniami: bool = True, z_definicjami: bool = True,
             na_dzien: Optional[str] = None) -> list[Jednostka]:
    """Zwraca jednostki do promptu w kolejności ważności, bez duplikatów, w budżecie znaków.

    Kolejność: trafienia → ich rodzice (cały artykuł) → odesłania 1 skok → definicje terminów
    występujących w trafieniach. Gdy do promptu wchodzi rodzic, sam ustęp i tak zostaje
    (model ma widzieć, który fragment był trafieniem). `na_dzien` odrzuca jednostki nieobowiązujące
    w tym dniu także wśród rodziców, odesłań i definicji (trafienia filtruje już szukaj()).
    """
    max_znakow = max_znakow or KONFIG.max_znakow_kontekstu
    wybrane: list[Jednostka] = []
    widziane: set[str] = set()
    uzyte = 0

    def dodaj(ident: Optional[str]) -> None:
        nonlocal uzyte
        if not ident or ident in widziane:
            return
        j = indeks.jednostki.get(ident)
        if j is None or not j.tekst.strip():
            return
        if na_dzien and not j.obowiazuje(na_dzien):
            return
        if uzyte + len(j.tekst) > max_znakow:
            return
        widziane.add(ident)
        wybrane.append(j)
        uzyte += len(j.tekst)

    for t in trafienia:
        dodaj(t.id)
    if z_rodzicem:
        for t in trafienia:
            j = indeks.jednostki.get(t.id)
            if j and j.rodzic:
                dodaj(j.rodzic)
    if z_odwolaniami:
        for t in trafienia:
            j = indeks.jednostki.get(t.id)
            if not j:
                continue
            for o in j.odwolania[:6]:
                dodaj(o)
    if z_definicjami and indeks.definicje:
        tekst_traf = " ".join(indeks.jednostki[t.id].tekst.lower() for t in trafienia if t.id in indeks.jednostki)
        for termin, ident in indeks.definicje.items():
            # całe słowo z końcówką fleksyjną (do 3 liter), a nie dowolny podciąg ("emeryt" w "emerytura")
            if re.search(r"\b" + re.escape(termin) + r"\w{0,3}\b", tekst_traf):
                dodaj(ident)
    return wybrane


def szukaj_pelnie(indeks: Indeks, pytanie: str, warianty: Optional[list[str]] = None,
                  na_dzien: Optional[str] = None, uzyj_emb: bool = True,
                  uzyj_reranker: bool = True, top_n: Optional[int] = None) -> list[Trafienie]:
    """Pełna ścieżka: warianty → hybryda → (reranker) → top-n."""
    zapytania = [pytanie] + [w for w in (warianty or []) if w and w != pytanie]
    kandydaci = szukaj(indeks, zapytania, na_dzien=na_dzien, uzyj_emb=uzyj_emb)
    if uzyj_reranker:
        return rerank(indeks, pytanie, kandydaci, top_n=top_n)
    return kandydaci[: (top_n or KONFIG.top_po_reranku)]
