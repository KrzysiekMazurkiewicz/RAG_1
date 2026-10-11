"""Contextual retrieval: do każdej jednostki szukalnej dopisujemy jedno zdanie opisu,
które umiejscawia ją w akcie i tłumaczy, o co pyta ktoś, kto jej szuka –
w potocznych słowach pracownika. Opis trafia do tekstu embeddowanego i do BM25.

Jednorazowa robota wsadowa: dla ustaw (tekst publiczny) można ją zrobić dużym
modelem na Colabie, dla procedur wewnętrznych – u siebie małym modelem.
Wynik zapisywany jest z powrotem do JSONL (pole `opis`), więc indeks budujesz potem.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .config import KONFIG
from .llm import BrakLLM, czat_json
from .models import Jednostka, wczytaj_jsonl, zapisz_jsonl

log = logging.getLogger("lexrag")

PROMPT = """Akt / dokument: {akt}
Położenie fragmentu: {sciezka}

Kontekst (cały artykuł lub sekcja nadrzędna):
{rodzic}

Fragment, który opisujesz:
{tekst}

Napisz JEDNO zdanie (maks. 30 słów), które:
- mówi, czego dotyczy ten fragment i kogo / jakiej sytuacji,
- używa potocznych słów, jakich użyłby pracownik szukający tego przepisu (np. "kiedy można", "ile wynosi", "kto musi"),
- nie powtarza numeru artykułu ani treści dosłownie.

Zwróć JSON: {{"opis": "..."}}
"""


def opisz(j: Jednostka, rodzic: Optional[Jednostka]) -> str:
    model = KONFIG.llm_model_maly or None
    w = czat_json([{"role": "user", "content": PROMPT.format(
        akt=j.akt, sciezka=j.sciezka,
        rodzic=(rodzic.tekst[:3000] if rodzic else "(brak)"),
        tekst=j.tekst[:3000])}], model=model, max_tokens=120)
    return (w.get("opis") or "").strip()


def kontekstualizuj_plik(sciezka: str, limit: Optional[int] = None, nadpisz: bool = False,
                         postep: Optional[Callable[[int, int], None]] = None) -> int:
    """Dopisuje opisy do jednostek szukalnych bez opisu (albo wszystkich przy nadpisz); `postep(zrobione, razem)`
    woła się po każdej jednostce, żeby interfejs mógł pokazać licznik."""
    jednostki = wczytaj_jsonl(sciezka)
    mapa = {j.id: j for j in jednostki}
    do_zrobienia = [j for j in jednostki if j.szukalny and (nadpisz or not j.opis)]
    razem = len(do_zrobienia) if limit is None else min(limit, len(do_zrobienia))
    zrobione = 0
    for j in do_zrobienia:
        if limit is not None and zrobione >= limit:
            break
        try:
            j.opis = opisz(j, mapa.get(j.rodzic) if j.rodzic else None)
            zrobione += 1
            if postep:
                postep(zrobione, razem)
            if zrobione % 25 == 0:
                log.info("opisy: %d", zrobione)
                zapisz_jsonl(jednostki, sciezka)  # zapis częściowy – przerwanie nic nie traci
        except (BrakLLM, ValueError) as e:
            log.warning("Pominięto %s: %s", j.id, e)
            if isinstance(e, BrakLLM):
                break
    zapisz_jsonl(jednostki, sciezka)
    return zrobione
