"""Wspólne narzędzia testów ścieżki odpowiedzi: skryptowany model (odpowiedzi wg etykiety kroku), budowniczowie
werdyktów weryfikatora i indeksy zbudowane z tekstów testowych (bez embeddingów, bez prawdziwego modelu)."""
import dataclasses
import os

from lexrag import generate
from lexrag.index import Indeks
from lexrag.models import zapisz_jsonl
from lexrag.parse_ustawa import parsuj_ustawe

TU = os.path.dirname(__file__)
DANE = os.path.join(TU, "..", "data", "samples")
FUS = os.path.join(TU, "data", "fus_wyciag.txt")        # dosłowny wyciąg z tekstu jednolitego (art. 24, 27, 29, 31)
WERSJA_FUS = "tj. Dz.U. 2024 poz. 1631"


def tekst_fus() -> str:
    with open(FUS, encoding="utf-8") as f:
        return "\n".join(l for l in f.read().splitlines() if not l.startswith("#"))


def zbuduj_indeks_fus(katalog, obowiazuje_od=None, obowiazuje_do=None) -> Indeks:
    jednostki = parsuj_ustawe(tekst_fus(), "Ustawa o emeryturach i rentach z FUS (wyciąg do testów)", "fus",
                              obowiazuje_od, obowiazuje_do, wersja=WERSJA_FUS)
    plik = str(katalog / "fus.jsonl")
    zapisz_jsonl(jednostki, plik)
    return Indeks.zbuduj([plik], str(katalog / "index"), z_embeddingami=False)


def skrypt_llm(monkeypatch, z_naprawa: bool = False, **odpowiedzi) -> list:
    """Podmienia czat_json: odpowiedzi wg etykiety kroku („analiza”, „odpowiedz”, „weryfikacja”, „weryfikacja_zdania”,
    „naprawa”). Wartość to słownik (każde wywołanie dostaje to samo) albo lista (kolejne wywołania dostają kolejne
    elementy; ostatni się powtarza); wyjątek na liście jest rzucany. Zwraca listę wywołań [(etykieta, prompt, schemat)]."""
    kolejki = {k: (list(v) if isinstance(v, list) else [v]) for k, v in odpowiedzi.items()}
    wywolania: list = []

    def fake(wiadomosci, model=None, proby=2, **kw):
        et = kw.get("etykieta", "")
        wywolania.append((et, wiadomosci[-1]["content"], kw.get("schema")))
        kolejka = kolejki.get(et)
        if not kolejka:
            raise AssertionError(f"niespodziewane wywołanie modelu: {et!r}")
        w = kolejka.pop(0) if len(kolejka) > 1 else kolejka[0]
        if isinstance(w, BaseException):
            raise w
        return w

    monkeypatch.setattr(generate, "czat_json", fake)
    monkeypatch.setattr(generate, "llm_dostepny", lambda: True)
    # niezależnie od zmiennych środowiskowych LEXRAG_SKIP_ANALYSIS i LEXRAG_NAPRAWA
    monkeypatch.setattr(generate, "KONFIG", dataclasses.replace(generate.KONFIG, pomin_analize=False, naprawa=z_naprawa))
    return wywolania


def analiza(niezbedne=(), do_oceny=(), pytanie="", warianty=()):
    """Odpowiedź analizy; dane to pary (dana, dlaczego)."""
    dane = [{"dana": d, "zakres": "niezbedna", "dlaczego": p} for d, p in niezbedne]
    dane += [{"dana": d, "zakres": "do_pelnej_oceny", "dlaczego": p} for d, p in do_oceny]
    return {"warianty": list(warianty), "brakujace_dane": dane, "pytanie_doprecyzowujace": pytanie}


def odpowiedz_modelu(tekst, cytowane, brak_pokrycia=False, uwagi=""):
    return {"odpowiedz": tekst, "cytowane_id": list(cytowane), "brak_pokrycia": brak_pokrycia, "uwagi": uwagi}


def werdykt(nr, zrodlo, **kw):
    """Werdykt jednego zdania; domyślnie wszystkie kontrole przechodzą."""
    return {"nr": nr, "id": zrodlo, "pokryte": True, "struktura_warunkow_ok": True, "negacje_ok": True,
            "liczby_przypisane_ok": True, "zakres_podmiotowy_ok": True, "odeslania_ok": True,
            "potrzebne_do_pytania": True, "czesc_pytania": "główna wątpliwość", "odeslanie_w_zrodle": "", **kw}


def weryfikacja(zdania, **kw):
    """Odpowiedź weryfikatora; domyślnie całość w porządku."""
    wynik = {"zdania": zdania, "odpowiada_na_pytanie": True, "odpowiada_na_glowna_watpliwosc": True,
             "nie_laczy_odrebnych_podstaw": True, "odpowiedz_deklaruje_brak_danych": False,
             "dopytanie_jest_konieczne": False, "istotne_pominiecia": [], "bledy": []}
    wynik.update(kw)
    return wynik


def blad(zdanie, typ, opis, zrodlo=""):
    return {"zdanie": zdanie, "typ": typ, "opis": opis, "zrodlo": zrodlo}
