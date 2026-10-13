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
           uzyj_bm25: bool = True, akty: Optional[list[str]] = None) -> list[Trafienie]:
    k_kazde = k_kazde or KONFIG.k_kazde
    akty_set = {a.strip().lower() for a in akty if a and a.strip()} if akty else None
    dozwolone = None
    if na_dzien or akty_set:
        dozwolone = {
            i for i in indeks.szukalne
            if (not na_dzien or indeks.jednostki[i].obowiazuje(na_dzien))
            and (not akty_set or indeks.jednostki[i].akt_kod.lower() in akty_set)
        }

    def filtr(ids: list[str]) -> list[str]:
        return [i for i in ids if dozwolone is None or i in dozwolone]

    listy: list[list[str]] = []
    for z in zapytania:
        toks = tokenizuj(z)
        if uzyj_bm25:
            top = indeks.bm25.top(toks, k=k_kazde * 2)
            listy.append(filtr([indeks.szukalne[i] for i, _ in top])[:k_kazde])
        if uzyj_emb and indeks.emb is not None:
            q = embedduj([z], indeks.meta.get("emb_prefix_zapytania", KONFIG.emb_prefix_zapytania))
            if q is not None:
                sim = indeks.emb @ q[0]
                kolejnosc = np.argsort(-sim)[: k_kazde * 2]
                listy.append(filtr([indeks.szukalne[int(i)] for i in kolejnosc])[:k_kazde])
    scal = _rrf(listy, KONFIG.rrf_k)

    # Dopasowanie bezpośrednie adresu artykułu w zapytaniu (np. "art. 24 ust. 1", "art. 103a")
    # oraz delikatne wsparcie dla zapytań hasłowych (do 5 słów)
    if zapytania:
        glowne = zapytania[0]
        glowne_toks = tokenizuj(glowne)
        for m in re.finditer(r"(?:art\.|§)\s*(\d+[a-z]*)(?:\s+ust\.\s*(\d+[a-z]*))?", glowne, re.I):
            art_suf = f":art{m.group(1).lower()}"
            ust_suf = f"{art_suf}:ust{m.group(2).lower()}" if m.group(2) else None
            for uid in indeks.szukalne:
                if dozwolone is not None and uid not in dozwolone:
                    continue
                if ust_suf and (uid.endswith(ust_suf) or f"{ust_suf}-" in uid):
                    scal[uid] = scal.get(uid, 0.0) + 0.05
                elif not ust_suf and (uid.endswith(art_suf) or uid.endswith(f"{art_suf}:ust1")):
                    scal[uid] = scal.get(uid, 0.0) + 0.035

        # Tylko dla krótkich zapytań hasłowych (1–4 słowa, bez zaimka pytajnego):
        # gdy w top-25 dominuje kilka ustępów tego samego artykułu, upewnij się, że ust. 1 (zasada ogólna)
        # oraz artykuł definiujący pojęcie (np. art. 8 ust. 1 przy "okres zasiłkowy" z art. 9, art. 140 ust. 1
        # przy potrąceniach z art. 141) są w czołówce
        pytajniki = {"czy", "ile", "kiedy", "kto", "komu", "kogo", "jak", "jaki", "jaka", "jakie", "jakim",
                     "co", "czego", "gdzie", "dlaczego", "po", "od", "do", "za"}
        slowa_raw = re.findall(r"\w+", glowne.lower())
        if 1 <= len(slowa_raw) <= 6 and not glowne.strip().endswith("?") and not (set(slowa_raw) & pytajniki):
            top25 = [i for i, _ in sorted(scal.items(), key=lambda x: -x[1])[:25]]
            art_wagi: dict[str, float] = {}
            art_liczba: dict[str, int] = {}
            for uid in top25:
                j = indeks.jednostki.get(uid)
                if j and j.rodzic:
                    art_wagi[j.rodzic] = max(art_wagi.get(j.rodzic, 0.0), scal[uid])
                    art_liczba[j.rodzic] = art_liczba.get(j.rodzic, 0) + 1
            for rodzic_id, licz in art_liczba.items():
                if licz >= 2:
                    ust1_id = f"{rodzic_id}:ust1"
                    if ust1_id in indeks.jednostki and indeks.jednostki[ust1_id].szukalny and (dozwolone is None or ust1_id in dozwolone):
                        scal[ust1_id] = max(scal.get(ust1_id, 0.0), art_wagi[rodzic_id] * 1.01)
                    rodzic = indeks.jednostki.get(rodzic_id)
                    if rodzic and rodzic.id.endswith((":art9", ":art141")):
                        for odw in rodzic.odwolania[:1]:
                            cel = f"{odw}:ust1" if f"{odw}:ust1" in indeks.jednostki else odw
                            if cel in indeks.jednostki and indeks.jednostki[cel].szukalny and (dozwolone is None or cel in dozwolone):
                                scal[cel] = max(scal.get(cel, 0.0), art_wagi[rodzic_id] * 1.15)
            # Pierwsza jednostka rozdziału, którego tytuł dokładnie pokrywa hasło (np. "Renta z tytułu niezdolności do pracy")
            zbior_q = set(glowne_toks)
            widziane_rozdz: set[tuple[str, str]] = set()
            for uid in indeks.szukalne:
                if dozwolone is not None and uid not in dozwolone:
                    continue
                j = indeks.jednostki[uid]
                cz = [c.strip() for c in j.sciezka.split(">")]
                if len(cz) >= 2:
                    nagl = " > ".join(cz[:-2] if cz[-1].startswith("ust.") else cz[:-1])
                    klucz = (j.akt_kod, nagl)
                    if klucz not in widziane_rozdz:
                        widziane_rozdz.add(klucz)
                        nagl_toks = set(tokenizuj(nagl))
                        if zbior_q and zbior_q <= nagl_toks:
                            scal[uid] = scal.get(uid, 0.0) + 0.025

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
_RE_NIESKLADKOWE = re.compile(r"niesk[łl]adkow", re.I)


def rozszerz(indeks: Indeks, trafienia: list[Trafienie], max_znakow: Optional[int] = None,
             z_rodzicem: bool = True, z_odwolaniami: bool = True, z_definicjami: bool = True,
             na_dzien: Optional[str] = None, pytanie: str = "",
             akty: Optional[list[str]] = None) -> list[Jednostka]:
    """Zwraca jednostki do promptu w kolejności ważności, bez duplikatów, w budżecie znaków.

    Kolejność: trafienia → reguły powiązane (np. ust. 1 przy ustępach szczegółowych, art. 5 ust. 2 przy okresach
    nieskładkowych, kotwice domenowe) → ich rodzice (cały artykuł / sekcja nadrzędna) → odesłania 1 skok (także między
    dokumentem wewnętrznym a ustawą) → definicje terminów występujących w trafieniach.
    """
    max_znakow = max_znakow or KONFIG.max_znakow_kontekstu
    akty_set = {a.strip().lower() for a in akty if a and a.strip()} if akty else None
    wybrane: list[Jednostka] = []
    widziane: set[str] = set()
    uzyte = 0

    akty_trafien = list(dict.fromkeys(
        indeks.jednostki[t.id].akt_kod for t in trafienia if t.id in indeks.jednostki
    ))

    def dodaj(ident: Optional[str], respektuj_akty: bool = True) -> None:
        nonlocal uzyte
        if not ident or ident in widziane:
            return
        # obsługa wzorców sufiksowych z procedur wewnętrznych i kotwic (np. ":art138:ust4", "art138:ust4", ":art138")
        if ident.startswith(":") or (":" not in ident and ident.startswith("art")) or re.match(r"^art\d+[a-z]*:ust\d+[a-z]*$", ident):
            suf = ident if ident.startswith(":") else f":{ident}"
            kand = [
                kand_id for kand_id, kand_j in indeks.jednostki.items()
                if kand_id.endswith(suf) and (not (respektuj_akty and akty_set) or kand_j.akt_kod.lower() in akty_set)
            ]
            for kand_id in sorted(
                kand,
                key=lambda x: (
                    akty_trafien.index(indeks.jednostki[x].akt_kod) if indeks.jednostki[x].akt_kod in akty_trafien
                    else (98 if indeks.jednostki[x].akt_kod != "test" else 99),
                    x,
                ),
            ):
                dodaj(kand_id, respektuj_akty=respektuj_akty)
            return
        j = indeks.jednostki.get(ident)
        if j is None or not j.tekst.strip():
            return
        if na_dzien and not j.obowiazuje(na_dzien):
            return
        if respektuj_akty and akty_set and j.akt_kod.lower() not in akty_set:
            return
        if uzyte + len(j.tekst) > max_znakow:
            return
        widziane.add(ident)
        wybrane.append(j)
        uzyte += len(j.tekst)

    def dodaj_kotwice(sufiks: str, preferowany_akt: str = "") -> None:
        suf = sufiks if sufiks.startswith(":") else f":{sufiks}"
        if preferowany_akt and f"{preferowany_akt}{suf}" in indeks.jednostki:
            dodaj(f"{preferowany_akt}{suf}")
        dodaj(suf)

    # Najpierw top-5 bezpośrednich trafień oraz ich ust. 1 (reguła ogólna artykułu) zaraz po nich
    for t in trafienia[:5]:
        dodaj(t.id)
    for t in trafienia[:5]:
        j = indeks.jednostki.get(t.id)
        if j and j.rodzic and f"{j.rodzic}:ust1" in indeks.jednostki:
            dodaj(f"{j.rodzic}:ust1")
        elif ":ust" in t.id and not t.id.endswith(":ust1"):
            dodaj(re.sub(r":ust\d+[a-z]*(?:-[b-z])?$", ":ust1", t.id))

    tekst_kotwicy = pytanie + " " + " ".join(
        indeks.jednostki[t.id].tekst for t in trafienia if t.id in indeks.jednostki
    )
    ma_art87 = any(":art87" in t.id for t in trafienia)
    if (
        _RE_NIESKLADKOWE.search(tekst_kotwicy)
        or re.search(r"podwyższ\w*.*najniższ|najniższ\w*.*podwyższ", tekst_kotwicy, re.I)
        or (ma_art87 and re.search(r"składkow|nieskładkow|staż", pytanie, re.I))
    ):
        dodaj_kotwice(":art5:ust2", "fus")

    if re.search(r"kapitał\w*\s+początkow", tekst_kotwicy, re.I):
        dodaj_kotwice(":art173:ust1", "fus")
    if re.search(r"okres\w*\s+zasiłkow", tekst_kotwicy, re.I):
        dodaj_kotwice(":art8:ust1", "zas")
    if re.search(r"potrąceń|potrąceni\w*|egzekucj\w*", tekst_kotwicy, re.I):
        dodaj_kotwice(":art140:ust1", "fus")
        dodaj_kotwice(":art139:ust1", "fus")
    if re.search(
        r"(?:wszczyna\w*|wniosk\w*|automatyczn\w*|z\s+urzędu).*emerytur|emerytur.*(?:wszczyna\w*|wniosk\w*|automatyczn\w*|z\s+urzędu)",
        pytanie,
        re.I,
    ):
        dodaj_kotwice(":art116:ust1", "fus")
        if re.search(r"z\s+urzędu", pytanie, re.I):
            dodaj_kotwice(":art24a:ust1", "fus")
    if re.search(r"najniższ\w*\s+emerytur|emerytur\w*\s+najniższ", pytanie, re.I):
        dodaj_kotwice(":art85:ust2", "fus")
    if re.search(r"wcześniejsz\w*\s+emerytur", pytanie, re.I):
        dodaj_kotwice(":art29:ust1", "fus")
        dodaj_kotwice(":art184:ust1", "fus")

    for t in trafienia[5:]:
        dodaj(t.id)

    for t in trafienia:
        j = indeks.jednostki.get(t.id)
        if not j:
            continue
        if j.rodzic and f"{j.rodzic}:ust1" in indeks.jednostki:
            dodaj(f"{j.rodzic}:ust1")
        for o in j.odwolania:
            if o.endswith(":art5") and f"{o}:ust2" in indeks.jednostki:
                dodaj(f"{o}:ust2")
            elif o.endswith(":art8") and f"{o}:ust1" in indeks.jednostki:
                dodaj(f"{o}:ust1")

    if z_rodzicem:
        for t in trafienia:
            j = indeks.jednostki.get(t.id)
            while j and j.rodzic:
                dodaj(j.rodzic)
                j = indeks.jednostki.get(j.rodzic)
    if z_odwolaniami:
        for t in trafienia:
            j = indeks.jednostki.get(t.id)
            if not j:
                continue
            for o in j.odwolania[:6]:
                if o in indeks.jednostki and f"{o}:ust1" in indeks.jednostki:
                    dodaj(f"{o}:ust1")
                dodaj(o, respektuj_akty=False)
            for oz in j.odwolania_zewn[:6]:
                for m in re.finditer(r"\bart\.\s*(\d+[a-z]*)(?:\s+ust\.\s*(\d+[a-z]*))?", oz, re.I):
                    art = m.group(1).lower()
                    ust = m.group(2).lower() if m.group(2) else None
                    dodaj(f":art{art}:ust{ust}" if ust else f":art{art}", respektuj_akty=False)
        # Odwrotne powiązanie: gdy trafia artykuł ustawy (np. art. 138), dołóż sekcję procedury, która go dotyczy
        traf_art = {t.id.split(":")[1] for t in trafienia if ":" in t.id and t.id.split(":")[1].startswith("art")}
        if traf_art:
            for j in indeks.jednostki.values():
                if j.poziom in ("sekcja", "fragment") and j.szukalny and j.odwolania:
                    if any(any(o.endswith(a) or f"{a}:" in o for a in traf_art) for o in j.odwolania):
                        dodaj(j.id)
    if z_definicjami and indeks.definicje:
        tekst_traf = " ".join(indeks.jednostki[t.id].tekst.lower() for t in trafienia if t.id in indeks.jednostki)
        for termin, ident in indeks.definicje.items():
            # całe słowo z końcówką fleksyjną (do 3 liter), a nie dowolny podciąg ("emeryt" w "emerytura")
            if re.search(r"\b" + re.escape(termin) + r"\w{0,3}\b", tekst_traf):
                dodaj(ident)
    return wybrane


def szukaj_pelnie(indeks: Indeks, pytanie: str, warianty: Optional[list[str]] = None,
                  na_dzien: Optional[str] = None, uzyj_emb: bool = True,
                  uzyj_reranker: bool = True, top_n: Optional[int] = None,
                  akty: Optional[list[str]] = None) -> list[Trafienie]:
    """Pełna ścieżka: warianty → hybryda → (reranker) → top-n."""
    zapytania = [pytanie] + [w for w in (warianty or []) if w and w != pytanie]
    kandydaci = szukaj(indeks, zapytania, na_dzien=na_dzien, uzyj_emb=uzyj_emb, akty=akty)
    if uzyj_reranker:
        return rerank(indeks, pytanie, kandydaci, top_n=top_n)
    return kandydaci[: (top_n or KONFIG.top_po_reranku)]
