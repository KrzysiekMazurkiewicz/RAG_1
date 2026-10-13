"""Interfejs WWW i API (FastAPI): pytania z cytatami, wgrywanie aktów i dokumentów, stan indeksu.

    uvicorn lexrag.web:app --host 0.0.0.0 --port 8090

Zmienne: LEXRAG_DATA_DIR (domyślnie data), LEXRAG_INDEX_DIR (domyślnie data/index) + reszta z config.py.
Jeden wspólny indeks dla wszystkich wgranych aktów (data/units/*.jsonl); wgranie przebudowuje go w tle.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import secrets
import shutil
import threading
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import __version__
from .config import KONFIG, ustaw
from .generate import odpowiedz
from .index import Indeks
from .llm import llm_dostepny, resetuj_obsluge, tryb_api
from .models import zapisz_jsonl

log = logging.getLogger("lexrag")
KATALOG_DANYCH = os.environ.get("LEXRAG_DATA_DIR", "data")
KATALOG_INDEKSU = os.environ.get("LEXRAG_INDEX_DIR", os.path.join(KATALOG_DANYCH, "index"))
Z_EMBEDDINGAMI = not KONFIG.emb_wylacz
ROZSZERZENIA = (".pdf", ".txt", ".html", ".htm", ".docx", ".md")
PLIK_MODELU = "model.json"   # ustawienia serwera modelu zapisane z panelu (nadpisują zmienne środowiskowe)


class Stan:
    def __init__(self) -> None:
        self.indeks: Optional[Indeks] = None
        self.blokada = threading.Lock()
        self.zadanie: dict = {"status": "bezczynny", "komunikat": "", "kod": ""}

    def wczytaj(self) -> Optional[Indeks]:
        if os.path.exists(os.path.join(KATALOG_INDEKSU, "meta.json")):
            self.indeks = Indeks.wczytaj(KATALOG_INDEKSU)
        else:
            self.indeks = None
        return self.indeks


STAN = Stan()


@asynccontextmanager
async def _cykl(app: FastAPI):
    _wczytaj_ustawienia_modelu()
    try:
        STAN.wczytaj()
    except Exception as e:  # uszkodzony indeks nie może położyć serwera
        log.warning("Nie wczytano indeksu z %s: %s", KATALOG_INDEKSU, e)
    yield


app = FastAPI(title="LexRAG", version=__version__, lifespan=_cykl)


# --- serwer modelu: ustawienia z panelu i sprawdzanie połączenia ---------------------
_POLACZENIE: dict = {"sprawdzono": 0.0, "ok": False, "komunikat": ""}


def _wczytaj_ustawienia_modelu() -> None:
    sciezka = os.path.join(KATALOG_DANYCH, PLIK_MODELU)
    if not os.path.exists(sciezka):
        return
    try:
        with open(sciezka, encoding="utf-8") as f:
            d = json.load(f)
        ustaw(**{k: v for k, v in d.items() if k in ("llm_base_url", "llm_model", "llm_api") and isinstance(v, str)})
        log.info("Ustawienia serwera modelu z %s: %s", sciezka, d)
    except Exception as e:
        log.warning("Nie wczytano %s: %s", sciezka, e)


def _sprawdz_polaczenie(wymus: bool = False) -> dict:
    """Czy serwer modelu odpowiada i ma model: Ollama GET /api/tags, serwer OpenAI GET /models; wynik buforowany 60 s."""
    if not KONFIG.llm_model:
        return {"ok": False, "komunikat": "brak modelu (LEXRAG_LLM_MODEL)", "sprawdzono": time.time()}
    if not wymus and time.time() - _POLACZENIE["sprawdzono"] < 60:
        return dict(_POLACZENIE)
    baza = KONFIG.llm_base_url.rstrip("/")
    ollama = tryb_api() == "ollama"
    url = (re.sub(r"/v1$", "", baza) + "/api/tags") if ollama else (baza + "/models")
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {KONFIG.llm_api_key}"})
        with urllib.request.urlopen(req, timeout=8) as r:
            dane = json.loads(r.read().decode("utf-8", "replace") or "{}")
        if ollama:
            nazwy = {m.get("name", "") for m in dane.get("models", [])} | {m.get("model", "") for m in dane.get("models", [])}
            szukane = {KONFIG.llm_model, KONFIG.llm_model + ":latest"}
            if nazwy & szukane:
                wynik = {"ok": True, "komunikat": f"serwer odpowiada, model {KONFIG.llm_model} jest na nim"}
            else:
                wynik = {"ok": False, "komunikat": f"serwer odpowiada, ale nie ma modelu {KONFIG.llm_model} "
                                                   f"(ma: {', '.join(sorted(n for n in nazwy if n))[:120] or 'nic'})"}
        else:
            wynik = {"ok": True, "komunikat": "serwer odpowiada"}
    except Exception as e:
        wynik = {"ok": False, "komunikat": f"{type(e).__name__}: {str(e)[:160]}"}
    wynik["sprawdzono"] = time.time()
    _POLACZENIE.update(wynik)
    return dict(wynik)


# --- pomocnicze ----------------------------------------------------------------------
def _jednostka_json(j) -> dict:
    d = asdict(j)
    d.pop("opis", None)
    return d


def _akty(idx: Indeks) -> list[dict]:
    akty: dict[str, dict] = {}
    for j in idx.jednostki.values():
        a = akty.setdefault(j.akt_kod, {"kod": j.akt_kod, "akt": j.akt, "jednostek": 0, "szukalnych": 0, "opisy": 0,
                                         "obowiazuje_od": j.obowiazuje_od, "obowiazuje_do": j.obowiazuje_do,
                                         "wersja": ""})
        a["jednostek"] += 1
        if j.wersja and not a["wersja"]:
            a["wersja"] = j.wersja
        if j.szukalny:
            a["szukalnych"] += 1
            if j.opis:
                a["opisy"] += 1
    return sorted(akty.values(), key=lambda a: a["kod"])


def _opisz_jednostki(kod: str, out: str) -> int:
    """Opisy kontekstowe modelem (contextual retrieval) dla jednostek bez opisu; licznik w STAN.zadanie."""
    from .contextualize import kontekstualizuj_plik

    def postep(n: int, razem: int) -> None:
        STAN.zadanie.update(komunikat=f"{kod}: opisy kontekstowe {n}/{razem} (model {KONFIG.llm_model})")

    return kontekstualizuj_plik(out, postep=postep)


def _przebuduj_indeks(kod: str, komunikat: str) -> None:
    katalog_units = os.path.join(KATALOG_DANYCH, "units")
    STAN.zadanie.update(komunikat=f"{kod}: {komunikat}, buduję indeks (embeddingi)…")
    pliki = sorted(glob.glob(os.path.join(katalog_units, "*.jsonl")))
    idx = Indeks.zbuduj(pliki, KATALOG_INDEKSU, z_embeddingami=Z_EMBEDDINGAMI)
    with STAN.blokada:
        STAN.indeks = idx
    STAN.zadanie = {"status": "gotowe", "kod": kod,
                    "komunikat": f"{kod}: {komunikat}; indeks: {len(idx.jednostki)} jednostek z {len(pliki)} aktów"}


def _zadanie_opisz(kod: str) -> None:
    try:
        out = os.path.join(KATALOG_DANYCH, "units", f"{kod}.jsonl")
        n = _opisz_jednostki(kod, out)
        _przebuduj_indeks(kod, f"{n} opisów kontekstowych")
    except BaseException as e:
        log.exception("Opisy %s nieudane", kod)
        STAN.zadanie = {"status": "blad", "kod": kod, "komunikat": f"{type(e).__name__}: {e}"}


def _zadanie_wgraj(sciezka: str, kod: str, akt: str, typ: str,
                   od: Optional[str], do: Optional[str], wersja: str, opisy: bool = False) -> None:
    try:
        katalog_units = os.path.join(KATALOG_DANYCH, "units")
        os.makedirs(katalog_units, exist_ok=True)
        out = os.path.join(katalog_units, f"{kod}.jsonl")
        if typ == "dokument":
            from .parse_doc import parsuj_markdown
            plik = sciezka
            if not plik.lower().endswith((".md", ".txt", ".markdown")):
                from .convert import do_markdown
                plik = do_markdown(plik, os.path.splitext(plik)[0] + ".md")
            with open(plik, encoding="utf-8") as f:
                tekst = f.read()
            jednostki = parsuj_markdown(tekst, akt or os.path.basename(sciezka), kod,
                                        obowiazuje_od=od, obowiazuje_do=do, wersja=wersja)
            st: dict = {"jednostek": len(jednostki)}
        else:
            from .fetch import wczytaj_dowolny
            from .parse_ustawa import parsuj_ustawe, statystyki
            tekst = wczytaj_dowolny(sciezka)
            jednostki = parsuj_ustawe(tekst, akt, kod, od, do, wersja=wersja)
            st = statystyki(jednostki)
        if len(jednostki) < 3:
            raise ValueError("za mało jednostek – ustawa musi mieć 'Art. N.' na początku linii, "
                             "dokument wewnętrzny nagłówki Markdown")
        zapisz_jsonl(jednostki, out)
        opis_info = ""
        if opisy:
            if llm_dostepny():
                opis_info = f", {_opisz_jednostki(kod, out)} opisów kontekstowych"
            else:
                opis_info = ", opisy pominięte (brak modelu)"
        _przebuduj_indeks(kod, f"{len(jednostki)} jednostek ({st}){opis_info}")
    except BaseException as e:
        log.exception("Wgrywanie %s nieudane", kod)
        STAN.zadanie = {"status": "blad", "kod": kod, "komunikat": f"{type(e).__name__}: {e}"}


# --- API --------------------------------------------------------------------------------
class Pytanie(BaseModel):
    pytanie: str
    na_dzien: Optional[str] = None
    analiza: Optional[bool] = None
    slad: Optional[bool] = None      # zapisz pełny ślad diagnostyczny tej jednej odpowiedzi (domyślnie: wg LEXRAG_TRACE)
    akty: Optional[list[str]] = None # opcjonalny filtr po kodach aktów (np. ["fus", "pnps"])


@app.get("/api/stan")
def stan() -> dict:
    idx = STAN.indeks
    return {
        "wersja": __version__,
        "indeks": {"katalog": KATALOG_INDEKSU, "jednostek": len(idx.jednostki) if idx else 0,
                   "szukalnych": len(idx) if idx else 0, "embeddingi": bool(idx is not None and idx.emb is not None),
                   "akty": _akty(idx) if idx else []},
        "llm": {"model": KONFIG.llm_model, "baza": KONFIG.llm_base_url, "dostepny": llm_dostepny(),
                "polaczenie": _sprawdz_polaczenie()},
        "zadanie": STAN.zadanie,
        "slad": {"wlaczony": KONFIG.slad_wlaczony, "dni": KONFIG.slad_dni},
    }


class UstawieniaModelu(BaseModel):
    base_url: str
    model: str
    api: str = "auto"


@app.get("/api/model")
def model_get() -> dict:
    return {"base_url": KONFIG.llm_base_url, "model": KONFIG.llm_model, "api": KONFIG.llm_api,
            "polaczenie": _sprawdz_polaczenie()}


@app.post("/api/model")
def model_post(u: UstawieniaModelu) -> dict:
    """Zmiana serwera modelu bez restartu (nowy adres tunelu, inny model); zapis do data/model.json, sprawdzenie połączenia."""
    base, model, api = u.base_url.strip(), u.model.strip(), (u.api.strip() or "auto")
    if not re.match(r"^https?://", base):
        raise HTTPException(400, "Adres serwera musi zaczynać się od http:// albo https://")
    if not model:
        raise HTTPException(400, "Podaj nazwę modelu (np. gemma4:12b).")
    if api not in ("auto", "ollama", "openai"):
        raise HTTPException(400, "api: auto, ollama albo openai.")
    ustaw(llm_base_url=base, llm_model=model, llm_api=api)
    resetuj_obsluge()
    with STAN.blokada:
        with open(_plik_danych(PLIK_MODELU), "w", encoding="utf-8") as f:
            json.dump({"llm_base_url": base, "llm_model": model, "llm_api": api}, f, ensure_ascii=False, indent=1)
    return {"base_url": base, "model": model, "api": api, "polaczenie": _sprawdz_polaczenie(wymus=True)}


@app.post("/api/pytanie")
def pytanie(p: Pytanie) -> dict:
    idx = STAN.indeks
    if idx is None:
        raise HTTPException(409, "Brak indeksu – wgraj najpierw akt.")
    tresc = p.pytanie.strip()
    if not tresc:
        raise HTTPException(400, "Puste pytanie.")
    # bez daty pytamy o stan prawny na dziś – przy kilku wersjach aktu w indeksie wybiera to bieżącą;
    # ostrzeżenia o stanie prawnym dostaje tylko ten, kto datę wskazał sam
    wskazana = (p.na_dzien or "").strip()
    na_dzien = wskazana or date.today().isoformat()
    akty_filtr = [a.strip().lower() for a in (p.akty or []) if a and a.strip()] or None
    w = odpowiedz(idx, tresc, na_dzien=na_dzien, analiza=p.analiza, data_wskazana=bool(wskazana), akty=akty_filtr)
    ident = w.slad.get("trace_id") or uuid.uuid4().hex
    slad_zapisany = bool(KONFIG.slad_wlaczony or p.slad)
    if slad_zapisany:
        _zapisz_slad(ident, w.slad)
    _dopisz("odpowiedzi.jsonl", {
        "id": ident, "czas": datetime.now().isoformat(timespec="seconds"), "pytanie": _maskuj(tresc),
        "na_dzien": na_dzien, "akty": akty_filtr, "tryb": w.tryb, "tekst": w.tekst, "cytowane": [j.id for j in w.cytaty],
        "kontekst": [j.id for j in w.kontekst], "model": KONFIG.llm_model, "indeks": KATALOG_INDEKSU,
        "przeglad": w.przeglad, "czasy": w.czasy, "status_weryfikacji": w.status_weryfikacji,
        "ostrzezenia": [o["kod"] for o in w.ostrzezenia], "naprawa": w.naprawa, "uwagi": w.uwagi,
        "zdania_usuniete": w.zdania_usuniete, "warianty": w.warianty,
        "brakujace_parametry": w.brakujace_parametry, "dopytanie": w.dopytanie, "slad_zapisany": slad_zapisany})
    akty = {a["kod"]: a for a in _akty(idx)}
    zrodla = [akty[k] for k in dict.fromkeys(j.akt_kod for j in w.cytaty) if k in akty]
    slad_token_param = f"?token={KONFIG.slad_token}" if (KONFIG.slad_token and slad_zapisany) else ""
    return {
        "id_odpowiedzi": ident, "trace_id": ident,
        "slad_url": f"/api/slad/{ident}{slad_token_param}" if slad_zapisany else None,
        "tryb": w.tryb, "tekst": w.tekst, "uwagi": w.uwagi, "czasy": w.czasy, "warianty": w.warianty,
        "brakujace_parametry": w.brakujace_parametry, "dane_do_pelnej_oceny": w.dane_do_pelnej_oceny,
        "zdania_usuniete": w.zdania_usuniete, "przeglad": w.przeglad, "dopytanie": w.dopytanie,
        "status_weryfikacji": w.status_weryfikacji, "ostrzezenia": w.ostrzezenia, "naprawa": w.naprawa,
        "zrodla": zrodla,
        "cytaty": [_jednostka_json(j) for j in w.cytaty],
        "trafienia": [{"id": t.id, "wynik": round(float(t.wynik), 3), "zrodlo": t.zrodlo,
                       "sciezka": idx.jednostki[t.id].sciezka, "tekst": idx.jednostki[t.id].tekst}
                      for t in w.trafienia[:8] if t.id in idx.jednostki],
        "kontekst": [j.id for j in w.kontekst],
    }


def _dostep_do_sladu(request: Request) -> bool:
    """Ślad zawiera pełną treść pytań i prompty (dane osobowe): czytać go może tylko żądanie z adresu lokalnego
    albo z poprawnym tokenem LEXRAG_TRACE_TOKEN w nagłówku X-Lexrag-Token lub w parametrze URL ?token=
    (np. link w przeglądarce lub warstwa C# po uwierzytelnieniu)."""
    token = KONFIG.slad_token
    if token:
        naglowek = request.headers.get("x-lexrag-token", "")
        parametr = request.query_params.get("token", "")
        if (naglowek and secrets.compare_digest(naglowek, token)) or (parametr and secrets.compare_digest(parametr, token)):
            return True
    host = request.client.host if request.client else ""
    return host in ("127.0.0.1", "::1", "localhost")


@app.get("/api/slad/{ident}")
def slad(ident: str, request: Request) -> dict:
    """Pełny ślad diagnostyczny odpowiedzi (jeśli został zapisany): analiza, zapytania, ranking, kontekst, wszystkie
    wiadomości do modelu i jego surowe odpowiedzi, werdykty kontroli, naprawa, wynik, konfiguracja bez sekretów."""
    if not _dostep_do_sladu(request):
        raise HTTPException(403, "Ślad diagnostyczny: dostęp tylko z adresu lokalnego albo z tokenem "
                                 "(nagłówek X-Lexrag-Token).")
    if not re.fullmatch(r"[0-9a-f]{32}", ident):
        raise HTTPException(400, "Zły identyfikator.")
    sciezka = os.path.join(KATALOG_DANYCH, "slady", f"{ident}.json")
    if not os.path.exists(sciezka):
        raise HTTPException(404, "Nie ma śladu o tym identyfikatorze (diagnostyka jest domyślnie wyłączona albo ślad "
                                 "wygasł).")
    with open(sciezka, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/jednostka/{ident}")
def jednostka(ident: str) -> dict:
    idx = STAN.indeks
    if idx is None or ident not in idx.jednostki:
        raise HTTPException(404, "Nie ma takiej jednostki.")
    return asdict(idx.jednostki[ident])


@app.get("/api/przepisy")
@app.get("/api/szukaj")
def przepisy(q: str = "", akt: str = "", limit: int = 50) -> list[dict]:
    """Błyskawiczne przeglądanie i wyszukiwanie jednostek w indeksie (dla zakładki 'Baza przepisów i procedur')."""
    idx = STAN.indeks
    if idx is None:
        return []
    limit = max(1, min(limit, 200))
    akt_kod = akt.strip().lower()
    akty_lista = [akt_kod] if akt_kod else None
    zap = q.strip()
    wynik_ids: list[str] = []
    widziane: set[str] = set()

    if zap:
        # 1. Dokładne lub prefiksowe dopasowanie po ID albo ścieżce (np. "art. 24", "fus:art24")
        zap_low = zap.lower()
        for uid, j in idx.jednostki.items():
            if akt_kod and j.akt_kod.lower() != akt_kod:
                continue
            if zap_low == uid.lower() or zap_low in uid.lower() or zap_low in j.sciezka.lower():
                if j.szukalny and uid not in widziane:
                    widziane.add(uid)
                    wynik_ids.append(uid)
                    if len(wynik_ids) >= limit:
                        break
        # 2. Wyszukiwanie BM25 w indeksie
        if len(wynik_ids) < limit:
            from .retrieve import szukaj as _szukaj_bm25
            traf = _szukaj_bm25(idx, [zap], k_kazde=limit, uzyj_emb=False, akty=akty_lista)
            for t in traf:
                if t.id not in widziane and t.id in idx.jednostki:
                    widziane.add(t.id)
                    wynik_ids.append(t.id)
                    if len(wynik_ids) >= limit:
                        break
        # 3. Dopasowanie podciągu w treści lub opisie
        if len(wynik_ids) < limit:
            for uid in idx.szukalne:
                if uid in widziane:
                    continue
                j = idx.jednostki[uid]
                if akt_kod and j.akt_kod.lower() != akt_kod:
                    continue
                if zap_low in j.tekst.lower() or (j.opis and zap_low in j.opis.lower()):
                    widziane.add(uid)
                    wynik_ids.append(uid)
                    if len(wynik_ids) >= limit:
                        break
    else:
        for uid in idx.szukalne:
            j = idx.jednostki[uid]
            if akt_kod and j.akt_kod.lower() != akt_kod:
                continue
            wynik_ids.append(uid)
            if len(wynik_ids) >= limit:
                break

    return [asdict(idx.jednostki[uid]) for uid in wynik_ids[:limit] if uid in idx.jednostki]


@app.get("/api/historia")
def historia(limit: int = 30) -> list[dict]:
    """Ostatnie odpowiedzi z dziennika odpowiedzi.jsonl (od najnowszych)."""
    limit = max(1, min(limit, 100))
    wpisy = _wczytaj("odpowiedzi.jsonl")
    return list(reversed(wpisy[-limit:]))


@app.post("/api/wgraj")
async def wgraj(plik: UploadFile = File(...), kod: str = Form(...), akt: str = Form(""),
                typ: str = Form("ustawa"), obowiazuje_od: str = Form(""), obowiazuje_do: str = Form(""),
                wersja: str = Form(""), opisy: str = Form("")) -> dict:
    kod = kod.strip().lower()
    if not re.fullmatch(r"[a-z0-9_]{1,24}", kod):
        raise HTTPException(400, "Kod aktu: małe litery, cyfry i podkreślenie, do 24 znaków.")
    if typ not in ("ustawa", "dokument"):
        raise HTTPException(400, "Typ: ustawa albo dokument.")
    if STAN.zadanie.get("status") == "w toku":
        raise HTTPException(409, "Trwa poprzednie wgrywanie – poczekaj, aż się skończy.")
    rozsz = os.path.splitext(plik.filename or "")[1].lower()
    if rozsz not in ROZSZERZENIA:
        raise HTTPException(400, "Dozwolone pliki: PDF, TXT, HTML, DOCX, MD.")
    katalog_raw = os.path.join(KATALOG_DANYCH, "raw")
    os.makedirs(katalog_raw, exist_ok=True)
    sciezka = os.path.join(katalog_raw, f"{kod}{rozsz}")
    with open(sciezka, "wb") as f:
        shutil.copyfileobj(plik.file, f)
    STAN.zadanie = {"status": "w toku", "kod": kod, "komunikat": f"parsowanie {plik.filename}…", "start": time.time()}
    threading.Thread(target=_zadanie_wgraj, daemon=True,
                     args=(sciezka, kod, akt.strip(), typ, obowiazuje_od.strip() or None,
                           obowiazuje_do.strip() or None, wersja.strip(),
                           opisy.strip().lower() in ("1", "true", "on", "tak"))).start()
    return {"status": "w toku", "kod": kod}


@app.post("/api/opisz/{kod}")
def opisz(kod: str) -> dict:
    """Opisy kontekstowe modelem dla wgranego już aktu (jednostki bez opisu), potem przebudowa indeksu."""
    kod = kod.strip().lower()
    if not os.path.exists(os.path.join(KATALOG_DANYCH, "units", f"{kod}.jsonl")):
        raise HTTPException(404, "Nie ma takiego aktu.")
    if not llm_dostepny():
        raise HTTPException(409, "Brak modelu – opisy kontekstowe pisze model (LEXRAG_LLM_MODEL).")
    if STAN.zadanie.get("status") == "w toku":
        raise HTTPException(409, "Trwa inne zadanie – poczekaj, aż się skończy.")
    STAN.zadanie = {"status": "w toku", "kod": kod, "komunikat": f"{kod}: opisy kontekstowe…", "start": time.time()}
    threading.Thread(target=_zadanie_opisz, daemon=True, args=(kod,)).start()
    return {"status": "w toku", "kod": kod}


# --- opinie użytkowników (dziennik odpowiedzi, zgłoszenia, przegląd referenta) ----------------
_RE_PESEL = re.compile(r"(?<!\d)\d{11}(?!\d)")
OCENY = ("dobrze", "zle", "nie_wiem")
STATUSY = ("nowa", "zatwierdzona", "odrzucona")


def _maskuj(tekst: str) -> str:
    """Pytania mogą zawierać dane osobowe: ciągi 11 cyfr (PESEL) nie trafiają do dziennika."""
    return _RE_PESEL.sub("***********", tekst)


def _maskuj_strukture(obj: Any) -> Any:
    """Rekurencyjnie maskuje PESEL (ciągi 11 cyfr) w wartościach tekstowych wewnątrz struktur,
    pozostawiając liczby, wartości bool itp. nienaruszone, aby uniknąć uszkodzenia składni JSON."""
    if isinstance(obj, str):
        return _maskuj(obj)
    if isinstance(obj, dict):
        return {k: _maskuj_strukture(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_maskuj_strukture(v) for v in obj]
    return obj


def _plik_danych(nazwa: str) -> str:
    os.makedirs(KATALOG_DANYCH, exist_ok=True)
    return os.path.join(KATALOG_DANYCH, nazwa)


def _czysc_slady() -> int:
    """Retencja: usuwa ślady starsze niż LEXRAG_TRACE_DAYS dni (liczone od czasu zapisu pliku)."""
    katalog = os.path.join(KATALOG_DANYCH, "slady")
    if not os.path.isdir(katalog):
        return 0
    granica = time.time() - KONFIG.slad_dni * 86400
    usuniete = 0
    for nazwa in os.listdir(katalog):
        sciezka = os.path.join(katalog, nazwa)
        if nazwa.endswith(".json") and os.path.getmtime(sciezka) < granica:
            os.remove(sciezka)
            usuniete += 1
    return usuniete


def _zapisz_slad(ident: str, slad: dict) -> None:
    """Osobny plik JSON na odpowiedź (prompty, surowe wyjścia modelu, werdykty). Ciągi 11 cyfr są maskowane w zapisie,
    a nie w żądaniu do modelu – model dostaje dokładnie to, co wpisał użytkownik. Przy każdym zapisie działa retencja."""
    katalog = os.path.join(KATALOG_DANYCH, "slady")
    os.makedirs(katalog, exist_ok=True)
    _czysc_slady()
    slad_bezpieczny = _maskuj_strukture(slad)
    with open(os.path.join(katalog, f"{ident}.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(slad_bezpieczny, ensure_ascii=False, indent=1, default=str))


def _dopisz(nazwa: str, rekord: dict) -> None:
    with STAN.blokada:
        with open(_plik_danych(nazwa), "a", encoding="utf-8") as f:
            f.write(json.dumps(rekord, ensure_ascii=False) + "\n")


def _wczytaj(nazwa: str) -> list[dict]:
    sciezka = _plik_danych(nazwa)
    if not os.path.exists(sciezka):
        return []
    with open(sciezka, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _zapisz_wszystkie(nazwa: str, rekordy: list[dict]) -> None:
    with STAN.blokada:
        with open(_plik_danych(nazwa), "w", encoding="utf-8") as f:
            for r in rekordy:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Opinia(BaseModel):
    id_odpowiedzi: str
    ocena: str
    wlasciwy_przepis: list[str] = []
    komentarz: str = ""
    uzytkownik: str = ""


class Przeglad(BaseModel):
    status: str
    wlasciwy_przepis: Optional[list[str]] = None
    musi_zawierac: list[str] = []
    przegladajacy: str = ""


@app.post("/api/opinia")
def opinia(o: Opinia) -> dict:
    """Ocena odpowiedzi: dobrze / zle / nie_wiem; przy „zle” trzeba wskazać właściwy przepis albo dopisać komentarz."""
    if o.ocena not in OCENY:
        raise HTTPException(400, "Ocena: dobrze, zle albo nie_wiem.")
    odp = next((r for r in _wczytaj("odpowiedzi.jsonl") if r["id"] == o.id_odpowiedzi), None)
    if odp is None:
        raise HTTPException(404, "Nie ma odpowiedzi o tym identyfikatorze.")
    przepisy = [p.strip() for p in o.wlasciwy_przepis if p.strip()]
    if o.ocena == "zle" and not przepisy and not o.komentarz.strip():
        raise HTTPException(400, "Przy ocenie „zle” wskaż właściwy przepis albo dopisz komentarz.")
    idx = STAN.indeks
    if idx is not None:
        brak = [p for p in przepisy if p not in idx.jednostki]
        if brak:
            raise HTTPException(400, f"Nieznane id przepisów: {', '.join(brak)}")
    rekord = {"id": uuid.uuid4().hex, "czas": datetime.now().isoformat(timespec="seconds"),
              "uzytkownik": o.uzytkownik.strip(), "id_odpowiedzi": o.id_odpowiedzi, "ocena": o.ocena,
              "wlasciwy_przepis": przepisy, "komentarz": _maskuj(o.komentarz.strip()), "status": "nowa",
              "musi_zawierac": [], "przegladajacy": "", "czas_przegladu": ""}
    _dopisz("opinie.jsonl", rekord)
    return {"id": rekord["id"], "status": "nowa"}


@app.get("/api/opinie")
def opinie(status: str = "nowa") -> list[dict]:
    """Kolejka przeglądu: opinie o danym statusie (domyślnie nowe) razem z pytaniem i odpowiedzią."""
    if status not in STATUSY + ("wszystkie",):
        raise HTTPException(400, "Status: nowa, zatwierdzona, odrzucona albo wszystkie.")
    odpowiedzi = {r["id"]: r for r in _wczytaj("odpowiedzi.jsonl")}
    out = []
    for o in reversed(_wczytaj("opinie.jsonl")):
        if status != "wszystkie" and o["status"] != status:
            continue
        odp = odpowiedzi.get(o["id_odpowiedzi"], {})
        out.append({**o, "pytanie": odp.get("pytanie", ""), "odpowiedz": odp.get("tekst", ""),
                    "cytowane": odp.get("cytowane", []), "tryb": odp.get("tryb", ""), "model": odp.get("model", "")})
    return out


@app.post("/api/opinie/{ident}/przeglad")
def przeglad(ident: str, p: Przeglad) -> dict:
    """Referent zatwierdza albo odrzuca opinię, może poprawić przepis i dopisać frazy, które muszą paść."""
    if p.status not in ("zatwierdzona", "odrzucona"):
        raise HTTPException(400, "Status: zatwierdzona albo odrzucona.")
    rekordy = _wczytaj("opinie.jsonl")
    rekord = next((r for r in rekordy if r["id"] == ident), None)
    if rekord is None:
        raise HTTPException(404, "Nie ma takiej opinii.")
    if p.wlasciwy_przepis is not None:
        rekord["wlasciwy_przepis"] = [x.strip() for x in p.wlasciwy_przepis if x.strip()]
    if p.status == "zatwierdzona" and rekord["ocena"] == "zle" and not rekord["wlasciwy_przepis"]:
        raise HTTPException(400, "Zatwierdzenie zgłoszenia „zle” wymaga wskazania właściwego przepisu.")
    rekord.update(status=p.status, musi_zawierac=[f.strip() for f in p.musi_zawierac if f.strip()],
                  przegladajacy=p.przegladajacy.strip(), czas_przegladu=datetime.now().isoformat(timespec="seconds"))
    _zapisz_wszystkie("opinie.jsonl", rekordy)
    return {"id": ident, "status": p.status}


@app.get("/api/opinie/eksport")
def eksport_opinii() -> dict:
    """Zatwierdzone opinie jako wiersze złotego zbioru: „zle” daje wskazany przepis, „dobrze” – cytowane przepisy."""
    odpowiedzi = {r["id"]: r for r in _wczytaj("odpowiedzi.jsonl")}
    wiersze = []
    for o in _wczytaj("opinie.jsonl"):
        if o["status"] != "zatwierdzona":
            continue
        odp = odpowiedzi.get(o["id_odpowiedzi"])
        if not odp:
            continue
        oczekiwane = o["wlasciwy_przepis"] or (odp.get("cytowane", []) if o["ocena"] == "dobrze" else [])
        if not oczekiwane:
            continue
        wiersze.append({"pytanie": odp["pytanie"], "oczekiwane": oczekiwane, "musi_zawierac": o.get("musi_zawierac", [])})
    return {"liczba": len(wiersze), "jsonl": "\n".join(json.dumps(w, ensure_ascii=False) for w in wiersze)}


# --- strona -----------------------------------------------------------------------------
STRONA = """<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>LexRAG</title>
<style>
:root{--bg:#f3f5f7;--fg:#1b2430;--muted:#5b6675;--panel:#fff;--line:#d5dbe3;--kod:#0f6e6e;--odm:#a33a2f;--ask:#b8741a}
@media (prefers-color-scheme:dark){:root{--bg:#121820;--fg:#e6e9ee;--muted:#9aa5b4;--panel:#1a222d;--line:#2d3744;--kod:#5fbdb9;--odm:#e07a6c;--ask:#e0a04a}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:26px;margin:8px 0 2px}h2{font-size:18px;margin:28px 0 8px}
.muted{color:var(--muted);font-size:14px}
textarea,input,select,button{font:inherit}
textarea{width:100%;box-sizing:border-box;min-height:72px;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--fg)}
input,select{padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--panel);color:var(--fg)}
button{padding:9px 16px;border:1px solid var(--kod);border-radius:8px;background:var(--kod);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:13px;font-weight:600;color:#fff;background:var(--kod)}
.badge.odmowa{background:var(--odm)}.badge.dopytanie{background:var(--ask)}.badge.retrieval-only{background:var(--muted)}
blockquote{margin:6px 0 10px;padding:8px 12px;border-left:3px solid var(--kod);background:var(--bg);border-radius:0 6px 6px 0;white-space:pre-wrap;font-size:15px}
.cyt .id{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:var(--muted)}
details{margin-top:8px}summary{cursor:pointer;color:var(--muted)}
table{border-collapse:collapse;font-size:14px}td,th{padding:4px 10px;border-bottom:1px solid var(--line);text-align:left}
#stan{font-size:14px}
</style></head><body><div class="wrap">
<h1>LexRAG</h1>
<div id="stan" class="muted">Łączę…</div>

<h2>Pytanie</h2>
<textarea id="pytanie" placeholder="np. za jaki okres wstecz można żądać zwrotu nienależnie pobranych świadczeń"></textarea>
<div class="row">
  <label>na dzień <input type="date" id="na_dzien"></label>
  <label><input type="checkbox" id="analiza"> analiza pytania (dopytanie, warianty)</label>
  <button id="zapytaj">Zapytaj</button>
  <span id="czeka" class="muted"></span>
</div>
<div id="wynik"></div>

<h2>Wgraj akt lub dokument</h2>
<form id="wgraj" class="card">
  <div class="row">
    <input type="file" id="plik" accept=".pdf,.txt,.html,.htm,.docx,.md" required>
    <select id="typ"><option value="ustawa">ustawa / rozporządzenie (Art., §)</option><option value="dokument">dokument wewnętrzny, instrukcja (nagłówki)</option></select>
  </div>
  <div class="row">
    <input id="kod" placeholder="kod, np. fus" pattern="[a-z0-9_]{1,24}" required>
    <input id="akt" placeholder="pełna nazwa aktu lub dokumentu" style="flex:1;min-width:240px">
  </div>
  <div class="row">
    <label>obowiązuje od <input type="date" id="od"></label>
    <label>do <input type="date" id="do"></label>
    <input id="wersja" placeholder="wersja (dokumenty)" style="width:140px">
    <button type="submit" id="wyslij">Wgraj i zbuduj indeks</button>
  </div>
  <div id="zadanie" class="muted"></div>
</form>
<p class="muted">Wgrywaj tekst jednolity z ISAP jako PDF. Po wgraniu sprawdź wyrywkowo kilka ustępów i zadaj pytania kontrolne.</p>
</div>
<script>
const $=id=>document.getElementById(id);
function esc(s){return String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function stan(){
  try{const r=await fetch("/api/stan");const s=await r.json();
    const akty=s.indeks.akty.map(a=>`<b>${esc(a.kod)}</b> ${esc(a.akt||"")} (${a.jednostek})`).join(" · ")||"brak aktów – wgraj pierwszy";
    const llm=s.llm.dostepny?`model ${esc(s.llm.model)}`:"bez modelu językowego (tylko wyszukiwanie)";
    $("stan").innerHTML=`Indeks: ${s.indeks.jednostek} jednostek, ${akty}. LLM: ${llm}.`;
    const z=s.zadanie||{};
    $("zadanie").textContent=z.status&&z.status!=="bezczynny"?`${z.status}: ${z.komunikat||""}`:"";
    if(z.status==="w toku"){setTimeout(stan,3000)}
    $("wyslij").disabled=z.status==="w toku";
  }catch(e){$("stan").textContent="Serwer nie odpowiada."}
}
function renderuj(w){
  let h=`<div class="card"><span class="badge ${esc(w.tryb)}">${esc(w.tryb)}</span>`;
  if(w.tekst)h+=`<p>${esc(w.tekst)}</p>`;
  if(w.przeglad)h+=`<p class="muted">odpowiedź przeglądowa – pytanie ogólne albo bez danych o sytuacji</p>`;
  if(w.brakujace_parametry&&w.brakujace_parametry.length)h+=`<p class="muted">brakuje: ${esc(w.brakujace_parametry.join(", "))}</p>`;
  if(w.dopytanie)h+=`<p class="muted">dopytanie: ${esc(w.dopytanie)}</p>`;
  if(w.cytaty&&w.cytaty.length){h+=`<h3 style="font-size:15px;margin:10px 0 4px">Podstawa prawna</h3>`;
    for(const c of w.cytaty)h+=`<div class="cyt"><div class="id">${esc(c.id)} · ${esc(c.sciezka)}</div><blockquote>${esc(c.tekst)}</blockquote></div>`}
  if(w.tryb==="retrieval-only"&&w.trafienia){h+=`<h3 style="font-size:15px;margin:10px 0 4px">Najlepsze trafienia</h3>`;
    for(const t of w.trafienia)h+=`<div class="cyt"><div class="id">${esc(t.id)} · ${esc(t.sciezka)} · ${t.wynik}</div><blockquote>${esc(t.tekst)}</blockquote></div>`}
  if(w.zdania_usuniete&&w.zdania_usuniete.length)h+=`<details><summary>zdania usunięte przez bramki (${w.zdania_usuniete.length})</summary><ul>${w.zdania_usuniete.map(z=>`<li>${esc(z)}</li>`).join("")}</ul></details>`;
  if(w.czasy)h+=`<p class="muted">czasy [s]: ${esc(Object.entries(w.czasy).map(([k,v])=>k+" "+v).join(", "))}</p>`;
  if(w.uwagi)h+=`<p class="muted">${esc(w.uwagi)}</p>`;
  $("wynik").innerHTML=h+"</div>";
}
$("zapytaj").onclick=async()=>{
  const p=$("pytanie").value.trim();if(!p)return;
  $("zapytaj").disabled=true;$("czeka").textContent="szukam i pytam model…";const t0=Date.now();
  try{const r=await fetch("/api/pytanie",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({pytanie:p,na_dzien:$("na_dzien").value||null,analiza:$("analiza").checked?true:null})});
    const w=await r.json();if(!r.ok){$("wynik").innerHTML=`<div class="card">${esc(w.detail||r.statusText)}</div>`}else renderuj(w);
  }catch(e){$("wynik").innerHTML=`<div class="card">Błąd: ${esc(e)}</div>`}
  $("zapytaj").disabled=false;$("czeka").textContent=`${((Date.now()-t0)/1000).toFixed(1)} s`;
};
$("wgraj").onsubmit=async e=>{
  e.preventDefault();const f=new FormData();f.append("plik",$("plik").files[0]);f.append("kod",$("kod").value);
  f.append("akt",$("akt").value);f.append("typ",$("typ").value);f.append("obowiazuje_od",$("od").value);
  f.append("obowiazuje_do",$("do").value);f.append("wersja",$("wersja").value);
  $("wyslij").disabled=true;$("zadanie").textContent="wysyłam…";
  const r=await fetch("/api/wgraj",{method:"POST",body:f});const w=await r.json();
  $("zadanie").textContent=r.ok?"w toku: parsowanie i indeks (embeddingi mogą potrwać kilka minut)":(w.detail||r.statusText);
  setTimeout(stan,2000);
};
stan();
</script></body></html>
"""


KATALOG_UI = os.environ.get("LEXRAG_UI_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                           "ui", "dist"))
if os.path.isfile(os.path.join(KATALOG_UI, "index.html")):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=KATALOG_UI, html=True), name="ui")   # React (ui/): montowane na końcu, po /api
else:
    @app.get("/", response_class=HTMLResponse)
    def strona() -> str:
        return STRONA
