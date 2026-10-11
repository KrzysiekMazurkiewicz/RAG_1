"""Parser struktury polskiego aktu prawnego (ustawa / rozporządzenie) z tekstu
w układzie ISAP / Dziennika Ustaw:

    DZIAŁ II                    <- dział + tytuł w kolejnej linii
    Rozdział 1                  <- rozdział + tytuł w kolejnej linii
    Art. 24. 1. ...             <- artykuł; pierwszy ustęp może być w tej samej linii
    1a. (uchylony)
    2. ...                      <- ustęp
    1) ...                      <- punkt (numer może stać sam w linii, treść w następnej)
    a) ...                      <- litera
    – ...                       <- tiret
    § 1. ...                    <- paragraf w rozporządzeniu (traktowany jak artykuł)
    Art. 145–172. (pominięte)   <- zakres artykułów pominiętych w tekście jednolitym
    Załącznik ...               <- załączniki po ostatnim artykule (osobne jednostki)

Jednostką wyszukiwania jest ustęp (albo cały artykuł, gdy nie ma ustępów).
Punkty i litery zostają wewnątrz ustępu. Definicje ustawowe ("określenia
oznaczają") dostają dodatkowo osobne jednostki per punkt.

Śmieci z PDF Dziennika Ustaw (nagłówki stron, przypisy, znaczniki przypisów
przyklejone do numerów: "1a.15)", "Art. 95a.13)", "(utracił moc)5)") są usuwane
regexami. Dla PDF najlepsze efekty daje jednak fetch.pdf_do_tekstu, który odcina
przypisy po rozmiarze czcionki, zanim tekst trafi tutaj.

Parser jest celowo prosty i czytelny – w instytucji publicznej każdy musi
móc sprawdzić, dlaczego dany ustęp wygląda tak, a nie inaczej.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import Jednostka

# --- wzorce strukturalne (zakotwiczone na początku linii) ----------------
RE_DZIAL = re.compile(r"^DZIAŁ\s+([IVXLC]+[A-Za-z]?)\s*$")
RE_ROZDZIAL = re.compile(r"^Rozdział\s+(\d+[a-z]?)\s*$")
RE_ART = re.compile(r"^(?:Art\.|§)\s*(\d+[a-z]*)\.(?:\s+(.*))?$")
RE_ART_ZAKRES = re.compile(r"^(?:Art\.|§)\s*(\d+[a-z]*)\s*[–—-]\s*(\d+[a-z]*)\.(?:\s+(.*))?$")
RE_UST = re.compile(r"^(\d+[a-z]*)\.(?:\s+(.*))?$")
RE_PKT = re.compile(r"^(\d+[a-z]*)\)(?:\s+(.*))?$")
RE_LIT = re.compile(r"^([a-z]{1,2})\)(?:\s+(.*))?$")
RE_TIRET = re.compile(r"^[–—-]\s+(.*)$")
RE_UCHYLONY = re.compile(r"^\(?\s*(?:uchylon[yae]|utraci[łl]a?\s+moc|pomini[ęe]t[yea])\s*\)?\.?$", re.I)
RE_ZALACZNIK = re.compile(r"^Załącznik(?:i)?\b")
# „Załącznik do obwieszczenia Marszałka Sejmu” = początek tekstu jednolitego (nie załącznik ustawy);
# wszystko przed nim to preambuła obwieszczenia, w której bywają zacytowane artykuły ustaw zmieniających
RE_ZAL_OBWIESZCZENIA = re.compile(r"^Załącznik do obwieszczenia")
RE_DEFINICJE = re.compile(r"okre[śs]lenia\s+oznaczaj[ąa]", re.I)
RE_DEF_PKT = re.compile(r"^(\d+[a-z]*)\)\s+(.+?)\s+[–—-]\s+(.*)$")

# znaczniki przypisów przyklejone do numeracji (gdy tekst nie przeszedł przez pdf_do_tekstu)
RE_PRZYPIS_UST = re.compile(r"^(\d+[a-z]*)\.(\d{1,2})\)(\s|$)")                 # "1a.15) Dla" -> "1a. Dla"
RE_PRZYPIS_ART = re.compile(r"^((?:Art\.|§)\s*\d+[a-z]*\.)(\d{1,2})\)(\s|$)")    # "Art. 95a.13) 1." -> "Art. 95a. 1."
RE_PRZYPIS_KONIEC = re.compile(r"\)(\d{1,2})\)\s*$")
RE_PRZYPIS_ART_UST = re.compile(r"^((?:Art\.|§)\s*\d+[a-z]*\.\s+\d+[a-z]*\.)(\d{1,2})\)(\s|$)")  # "Art. 24. 1.2) Ubezp." -> "Art. 24. 1. Ubezp."                             # "(utracił moc)5)" -> "(utracił moc)"

# linie-śmieci z PDF Dziennika Ustaw / ISAP
RE_SZUM = [
    re.compile(r"^©?\s*Kancelaria Sejmu"),
    re.compile(r"^Dziennik Ustaw"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^[–-]\s*\d+\s*[–-]$"),
    re.compile(r"^Poz\.\s*\d+"),
    re.compile(r"^s\.\s*\d+/\d+$"),
    # typowe przypisy z dołu strony (gdy nie odcięto ich po rozmiarze czcionki)
    re.compile(r"^\d{1,3}\)\s+(Niniejsza ustawa|Zmiany tekstu jednolitego|Zmiany wymienionej ustawy|Dodany przez|"
               r"W brzmieniu ustalonym|Ze zmian[ąa] wprowadzon[ąa]|Ze zmianami wprowadzonymi|Na podstawie|"
               r"Utraci[łl]a? moc|Uchylony przez|Przez art\.|Obecnie|Wsz[eę]d[łl]a? w życie|Wprowadzenie do wyliczenia)"),
]

# odesłania: "art. 46, 47, 50, 50a, 50e i 184", "art. 29–31", "art. 27 ust. 1 pkt 2"
RE_REF = re.compile(
    r"\bart\.\s*(\d+[a-z]*)((?:\s*(?:,|–|—|-|\bi\b|\boraz\b|\blub\b|\balbo\b)\s*\d+[a-z]*)*)",
    re.I,
)
RE_REF_NUM = re.compile(r"\d+[a-z]*")
RE_REF_ZEWN = re.compile(
    r"^\s*(?:ust\.\s*\d+[a-z]*\s*)?(?:pkt\s*\d+[a-z]*\s*)?(?:lit\.\s*[a-z]\s*)?"
    r"(ustawy|rozporządzenia|dekretu|kodeksu|konstytucji)",
    re.I,
)


@dataclass
class _Ustep:
    nr: str
    linie: list[str] = field(default_factory=list)


@dataclass
class _Artykul:
    nr: str
    dzial: str
    rozdzial: str
    tresc: list[str] = field(default_factory=list)   # linie artykułu bez ustępów
    ustepy: list[_Ustep] = field(default_factory=list)
    zakres: bool = False                              # "Art. 145–172. (pominięte)"

    @property
    def ostatnie_linie(self) -> list[str]:
        return self.ustepy[-1].linie if self.ustepy else self.tresc


def _numer(n: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([a-z]*)", n)
    return (int(m.group(1)), m.group(2)) if m else (0, "")


def _czysc(linie: list[str]) -> list[str]:
    out = []
    for l in linie:
        l = l.replace(" ", " ").replace("﻿", "")
        l = re.sub(r"[ \t]+", " ", l).strip()
        if not l:
            continue
        if any(r.match(l) for r in RE_SZUM):
            continue
        l = RE_PRZYPIS_ART.sub(r"\1\3", l)
        l = RE_PRZYPIS_ART_UST.sub(r"\1\3", l)
        l = RE_PRZYPIS_UST.sub(r"\1.\3", l)
        l = RE_PRZYPIS_KONIEC.sub(")", l)
        out.append(l.strip())
    return out


def _strukturalna(l: str) -> bool:
    return any(r.match(l) for r in (RE_DZIAL, RE_ROZDZIAL, RE_ART, RE_ART_ZAKRES, RE_UST, RE_PKT, RE_LIT, RE_TIRET))


def _dopisz(linie: list[str], tekst: str, nowa_linia: bool) -> None:
    if nowa_linia or not linie:
        linie.append(tekst)
    else:
        # PDF łamie wiersze w środku zdania – sklejamy; łącznik na końcu usuwamy
        if linie[-1].endswith("-") and tekst[:1].islower():
            linie[-1] = linie[-1][:-1] + tekst
        else:
            linie[-1] = (linie[-1] + " " + tekst).strip()


def _reszta(m: "re.Match", grupa: int) -> str:
    return (m.group(grupa) or "").strip()


def _parsuj_strukture(tekst: str) -> tuple[str, list[_Artykul], list[list[str]]]:
    """Zwraca (tytuł aktu, lista artykułów, lista załączników jako listy linii)."""
    linie = _czysc(tekst.splitlines())
    artykuly: list[_Artykul] = []
    zalaczniki: list[list[str]] = []
    art: Optional[_Artykul] = None
    dzial = ""
    rozdzial = ""
    tytul_do: Optional[str] = None     # "dzial" | "rozdzial" – zbieramy tytuł z kolejnych linii
    tytul_linie = 0
    naglowek: list[str] = []           # linie przed pierwszym artykułem (tytuł aktu)

    for l in linie:
        if RE_ZAL_OBWIESZCZENIA.match(l) and not zalaczniki:
            # start właściwego tekstu jednolitego: odrzuć artykuły z preambuły obwieszczenia i zacznij od nowa
            artykuly, art, dzial, rozdzial, tytul_do, tytul_linie, naglowek = [], None, "", "", None, 0, []
            continue
        # po pierwszym "Załącznik" (za ostatnim artykułem) wszystko idzie do załączników
        if zalaczniki:
            if RE_ZALACZNIK.match(l):
                zalaczniki.append([l])
            else:
                zalaczniki[-1].append(l)
            continue
        if art is not None and RE_ZALACZNIK.match(l):
            zalaczniki.append([l])
            continue

        m = RE_DZIAL.match(l)
        if m:
            dzial = f"Dział {m.group(1)}"
            rozdzial = ""
            tytul_do, tytul_linie = "dzial", 0
            continue
        m = RE_ROZDZIAL.match(l)
        if m:
            rozdzial = f"Rozdział {m.group(1)}"
            tytul_do, tytul_linie = "rozdzial", 0
            continue

        m = RE_ART_ZAKRES.match(l)
        if m:
            tytul_do = None
            art = _Artykul(nr=f"{m.group(1)}-{m.group(2)}", dzial=dzial, rozdzial=rozdzial, zakres=True)
            artykuly.append(art)
            if _reszta(m, 3):
                art.tresc.append(_reszta(m, 3))
            continue

        m = RE_ART.match(l)
        if m:
            tytul_do = None
            art = _Artykul(nr=m.group(1), dzial=dzial, rozdzial=rozdzial)
            artykuly.append(art)
            reszta = _reszta(m, 2)
            if reszta:
                mu = RE_UST.match(reszta)
                if mu and _numer(mu.group(1))[0] == 1:
                    r = _reszta(mu, 2)
                    art.ustepy.append(_Ustep(nr=mu.group(1), linie=[r] if r else []))
                else:
                    art.tresc.append(reszta)
            continue

        # tytuł działu / rozdziału: 1–2 krótkie linie bez znaczników
        if tytul_do and not _strukturalna(l) and tytul_linie < 2 and len(l) < 120 and not l.endswith((";", ":")):
            if tytul_do == "dzial":
                dzial = f"{dzial} {l}"
            else:
                rozdzial = f"{rozdzial} {l}"
            tytul_linie += 1
            continue
        tytul_do = None

        if art is None:
            naglowek.append(l)
            continue

        mu = RE_UST.match(l)
        if mu:
            n, _ = _numer(mu.group(1))
            if art.ustepy:
                n_cur, _ = _numer(art.ustepy[-1].nr)
                sekwencyjny = n_cur <= n <= n_cur + 3
            else:
                # ustęp 1 może zaczynać się w nowej linii tylko, gdy artykuł nie ma jeszcze treści
                sekwencyjny = (n == 1 and not art.tresc)
            if sekwencyjny:
                r = _reszta(mu, 2)
                art.ustepy.append(_Ustep(nr=mu.group(1), linie=[r] if r else []))
                continue
            # w przeciwnym razie to zawinięta linia zaczynająca się liczbą – zwykły tekst

        if RE_PKT.match(l) or RE_LIT.match(l) or RE_TIRET.match(l):
            _dopisz(art.ostatnie_linie, l, nowa_linia=True)
            continue

        _dopisz(art.ostatnie_linie, l, nowa_linia=False)

    tytul = " ".join(naglowek[:4]).strip()
    return tytul, artykuly, zalaczniki


def _odwolania(tekst: str, kod: str, wlasny_art: str, znane_art: set[str]) -> tuple[list[str], list[str]]:
    wewn: list[str] = []
    zewn: list[str] = []
    for m in RE_REF.finditer(tekst):
        calosc = m.group(0)
        po = tekst[m.end(): m.end() + 80]
        if RE_REF_ZEWN.match(po):
            zewn.append((calosc + " " + RE_REF_ZEWN.match(po).group(0).strip()).strip())
            continue
        numery = RE_REF_NUM.findall(calosc[4:])  # bez "art."
        # zakresy "29–31"
        rozwin: list[str] = []
        czesci = re.split(r"\s*(?:,|\bi\b|\boraz\b|\blub\b|\balbo\b)\s*", calosc[4:].strip())
        for cz in czesci:
            r = re.match(r"\s*(\d+)([a-z]*)\s*[–—-]\s*(\d+)([a-z]*)\s*$", cz)
            if r and not r.group(2) and not r.group(4):
                a, b = int(r.group(1)), int(r.group(3))
                if 0 < b - a <= 30:
                    rozwin.extend(str(x) for x in range(a, b + 1))
                    continue
            rozwin.extend(RE_REF_NUM.findall(cz))
        for n in rozwin or numery:
            if n == wlasny_art:
                continue
            if n in znane_art:
                ident = f"{kod}:art{n}"
                if ident not in wewn:
                    wewn.append(ident)
            else:
                z = f"art. {n} (nierozwiązane)"
                if z not in zewn:
                    zewn.append(z)
    return wewn, zewn


def _podziel_linie(linie: list[str], max_znakow: int) -> list[str]:
    """Dzieli listę linii na fragmenty o długości ~max_znakow, nie łamiąc linii."""
    fragmenty: list[str] = []
    akum: list[str] = []
    dl = 0
    for l in linie:
        if akum and dl + len(l) + 1 > max_znakow:
            fragmenty.append("\n".join(akum))
            akum, dl = [], 0
        akum.append(l)
        dl += len(l) + 1
    if akum:
        fragmenty.append("\n".join(akum))
    return fragmenty


def parsuj_ustawe(
    tekst: str,
    akt: str,
    kod: str,
    obowiazuje_od: Optional[str] = None,
    obowiazuje_do: Optional[str] = None,
    max_znakow_zalacznika: int = 1800,
    wersja: str = "",
) -> list[Jednostka]:
    """Zamienia tekst aktu na listę jednostek (artykuły-rodzice, ustępy, definicje, załączniki)."""
    tytul, artykuly, zalaczniki = _parsuj_strukture(tekst)
    akt = akt or tytul or kod
    znane = {a.nr for a in artykuly if not a.zakres}
    out: list[Jednostka] = []
    wspolne = dict(akt=akt, akt_kod=kod, obowiazuje_od=obowiazuje_od, obowiazuje_do=obowiazuje_do, wersja=wersja)
    widziane: set[str] = set()

    def unikalny(ident: str) -> tuple[str, str]:
        """Tekst jednolity drukuje czasem dwa brzmienia tego samego ustępu (obecne i przyszłe);
        drugie dostaje sufiks w id i dopisek w ścieżce zamiast zdublowanego id."""
        if ident not in widziane:
            widziane.add(ident)
            return ident, ""
        for i, suf in enumerate("bcdefgh", start=2):
            kand = f"{ident}-{suf}"
            if kand not in widziane:
                widziane.add(kand)
                return kand, f" (brzmienie {i})"
        raise ValueError(f"Zbyt wiele powtórzeń jednostki {ident}")

    for a in artykuly:
        id_art, dopisek_art = unikalny(f"{kod}:art{a.nr}")
        etykieta = f"Art. {a.nr.replace('-', '–')}{dopisek_art}"
        sciezka_art = " > ".join(p for p in (a.dzial.strip(), a.rozdzial.strip(), etykieta) if p)
        tresc_art = list(a.tresc)
        for u in a.ustepy:
            tresc_art.append(f"{u.nr}. " + "\n".join(u.linie))
        tekst_art = "\n".join(tresc_art).strip()
        wewn, zewn = _odwolania(tekst_art, kod, a.nr, znane)
        ma_ustepy = bool(a.ustepy)

        out.append(Jednostka(
            id=id_art, poziom="art", sciezka=sciezka_art, tekst=tekst_art,
            rodzic=None, odwolania=wewn, odwolania_zewn=zewn,
            szukalny=(not ma_ustepy) and not a.zakres and not RE_UCHYLONY.match(tekst_art or ""),
            **wspolne,
        ))

        for u in a.ustepy:
            tekst_u = "\n".join(u.linie).strip()
            w_u, z_u = _odwolania(tekst_u, kod, a.nr, znane)
            id_u, dopisek_u = unikalny(f"{id_art}:ust{u.nr}")
            out.append(Jednostka(
                id=id_u, poziom="ust", sciezka=f"{sciezka_art} > ust. {u.nr}{dopisek_u}", tekst=tekst_u,
                rodzic=id_art, odwolania=w_u, odwolania_zewn=z_u,
                szukalny=bool(tekst_u) and not RE_UCHYLONY.match(tekst_u),
                **wspolne,
            ))

        # definicje ustawowe -> osobna jednostka na każdy punkt
        zrodla = [(id_art, "\n".join(a.tresc))] + [(j.id, j.tekst) for j in out if j.poziom == "ust" and j.rodzic == id_art]
        for rodzic_id, blok in zrodla:
            if not RE_DEFINICJE.search(blok):
                continue
            for linia in blok.split("\n"):
                md = RE_DEF_PKT.match(linia.strip())
                if not md:
                    continue
                termin = md.group(2).strip()
                if RE_UCHYLONY.match(md.group(3).strip()) or len(termin) > 80:
                    continue
                id_d, _dop = unikalny(f"{id_art}:pkt{md.group(1)}")
                out.append(Jednostka(
                    id=id_d, poziom="pkt-def",
                    sciezka=f"{sciezka_art} > pkt {md.group(1)} (definicja)",
                    tekst=linia.strip(), opis=f"Definicja ustawowa pojęcia: {termin}.",
                    rodzic=rodzic_id, terminy=[termin], szukalny=True, **wspolne,
                ))

    # załączniki: jedna jednostka (albo rodzic + fragmenty, gdy długi)
    for i, zal in enumerate(zalaczniki, start=1):
        tekst_zal = "\n".join(zal).strip()
        if len(tekst_zal) < 40:
            continue
        id_zal = f"{kod}:zal{i}"
        sciezka_zal = (f"Załącznik {i}" if len(zalaczniki) > 1 else "Załącznik") + f" ({zal[0][:80]})"
        fragmenty = _podziel_linie(zal, max_znakow_zalacznika)
        if len(fragmenty) == 1:
            out.append(Jednostka(id=id_zal, poziom="zalacznik", sciezka=sciezka_zal, tekst=tekst_zal,
                                 szukalny=True, **wspolne))
            continue
        out.append(Jednostka(id=id_zal, poziom="zalacznik", sciezka=sciezka_zal, tekst=tekst_zal,
                             szukalny=False, **wspolne))
        for k, fr in enumerate(fragmenty, start=1):
            out.append(Jednostka(id=f"{id_zal}:f{k}", poziom="fragment",
                                 sciezka=f"{sciezka_zal} > fragment {k}/{len(fragmenty)}", tekst=fr,
                                 rodzic=id_zal, szukalny=True, **wspolne))
    return out


def statystyki(jednostki: list[Jednostka]) -> dict:
    return {
        "artykuly": sum(1 for j in jednostki if j.poziom == "art"),
        "ustepy": sum(1 for j in jednostki if j.poziom == "ust"),
        "definicje": sum(1 for j in jednostki if j.poziom == "pkt-def"),
        "zalaczniki": sum(1 for j in jednostki if j.poziom == "zalacznik"),
        "uchylone": sum(1 for j in jednostki if j.poziom in ("art", "ust") and RE_UCHYLONY.match(j.tekst or "")),
        "szukalne": sum(1 for j in jednostki if j.szukalny),
        "odwolania": sum(len(j.odwolania) for j in jednostki),
        "odwolania_zewn": sum(len(j.odwolania_zewn) for j in jednostki),
    }
