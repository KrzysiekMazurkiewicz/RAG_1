"""Parser procedur / instrukcji wewnętrznych z Markdown (DOCX → Markdown: patrz convert.py).

Jednostką jest sekcja pod nagłówkiem (#, ##, ###). Ścieżka = łańcuch nagłówków,
więc fragment "Terminy" z rozdziału "Upomnienie" nie myli się z "Terminami" z
rozdziału "Tytuł wykonawczy". Długie sekcje dzielone są na fragmenty po akapitach
(nigdy w środku tabeli), a cała sekcja zostaje jako rodzic do rozszerzania.
"""
from __future__ import annotations

import re
from typing import Optional

from .models import Jednostka

RE_NAGLOWEK = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
RE_NUMER_SEKCJI = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")


def _podziel_na_fragmenty(tekst: str, max_znakow: int) -> list[str]:
    """Dzieli po pustych liniach, trzymając tabele (linie z '|') w całości."""
    if len(tekst) <= max_znakow:
        return [tekst]
    bloki: list[str] = []
    biezacy: list[str] = []
    w_tabeli = False
    for l in tekst.split("\n"):
        jest_tabela = l.lstrip().startswith("|")
        if not l.strip() and not w_tabeli:
            if biezacy:
                bloki.append("\n".join(biezacy))
                biezacy = []
            continue
        if jest_tabela != w_tabeli and biezacy:
            bloki.append("\n".join(biezacy))
            biezacy = []
        w_tabeli = jest_tabela
        biezacy.append(l)
    if biezacy:
        bloki.append("\n".join(biezacy))

    fragmenty: list[str] = []
    akum = ""
    for b in bloki:
        if akum and len(akum) + len(b) + 2 > max_znakow:
            fragmenty.append(akum)
            akum = b
        else:
            akum = (akum + "\n\n" + b) if akum else b
    if akum:
        fragmenty.append(akum)
    return fragmenty


def parsuj_markdown(tekst: str, dokument: str, kod: str, max_znakow: int = 1800,
                    obowiazuje_od: Optional[str] = None, obowiazuje_do: Optional[str] = None,
                    wersja: str = "") -> list[Jednostka]:
    linie = tekst.replace(" ", " ").splitlines()
    stos: list[tuple[int, str]] = []   # (poziom, tytuł)
    sekcje: list[tuple[list[str], list[str]]] = []  # (ścieżka, linie)
    biezace: list[str] = []
    biezaca_sciezka: list[str] = []

    def zamknij():
        if biezace and "\n".join(biezace).strip():
            sekcje.append((list(biezaca_sciezka), list(biezace)))

    for l in linie:
        m = RE_NAGLOWEK.match(l)
        if m:
            zamknij()
            poziom = len(m.group(1))
            while stos and stos[-1][0] >= poziom:
                stos.pop()
            stos.append((poziom, m.group(2).strip()))
            biezaca_sciezka = [t for _, t in stos]
            biezace = []
            continue
        biezace.append(l)
    zamknij()

    out: list[Jednostka] = []
    for nr, (sciezka, tresc) in enumerate(sekcje, start=1):
        sekcja_tekst = "\n".join(tresc).strip()
        if sciezka and sciezka[0].lower().startswith(dokument.lower()[:20]):
            sciezka_txt = " > ".join(sciezka)          # H1 już jest nazwą dokumentu
        else:
            sciezka_txt = " > ".join([dokument] + sciezka) if sciezka else dokument
        id_sekcji = f"{kod}:s{nr}"
        fragmenty = _podziel_na_fragmenty(sekcja_tekst, max_znakow)
        opis_wersji = f"Wersja dokumentu: {wersja}." if wersja else ""
        if len(fragmenty) == 1:
            out.append(Jednostka(id=id_sekcji, akt=dokument, akt_kod=kod, poziom="sekcja",
                                 sciezka=sciezka_txt, tekst=sekcja_tekst, opis=opis_wersji,
                                 obowiazuje_od=obowiazuje_od, obowiazuje_do=obowiazuje_do, szukalny=True))
            continue
        out.append(Jednostka(id=id_sekcji, akt=dokument, akt_kod=kod, poziom="sekcja",
                             sciezka=sciezka_txt, tekst=sekcja_tekst,
                             obowiazuje_od=obowiazuje_od, obowiazuje_do=obowiazuje_do, szukalny=False))
        for i, fr in enumerate(fragmenty, start=1):
            out.append(Jednostka(id=f"{id_sekcji}:f{i}", akt=dokument, akt_kod=kod, poziom="fragment",
                                 sciezka=f"{sciezka_txt} > fragment {i}/{len(fragmenty)}", tekst=fr,
                                 opis=opis_wersji, rodzic=id_sekcji,
                                 obowiazuje_od=obowiazuje_od, obowiazuje_do=obowiazuje_do, szukalny=True))
    return out
