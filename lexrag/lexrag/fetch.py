"""Pobieranie tekstu aktu z API ELI Sejmu RP (api.sejm.gov.pl) i konwersja do tekstu.

Identyfikator aktu to pozycja w Dzienniku Ustaw: "DU/2024/1631" (rok/pozycja).
Dla ustawy, która ma tekst jednolity, użyj identyfikatora OBWIESZCZENIA z tekstem
jednolitym (link "Tekst jednolity" na ISAP) – wtedy dostajesz aktualne brzmienie.

Jeśli masz już PDF/HTML z ISAP – pomiń fetch i podaj plik do `parse` / `wgraj`
(obsługuje .txt, .html, .pdf, .docx przez pandoc).

PDF Dziennika Ustaw czytamy przez PyMuPDF w trybie "dict" (spany z rozmiarem czcionki):
  * tekst zasadniczy ma jeden dominujący rozmiar (zwykle 10 pt),
  * przypisy na dole strony są mniejsze (9 pt) – wyrzucamy je,
  * znaczniki przypisów w tekście to indeks górny ("1a.15)") – wyrzucamy je,
  * spany z tej samej linii (np. "1)" i treść punktu) sklejamy w jeden wiersz.
Dzięki temu parser dostaje czysty tekst, bez przypisów wklejonych w ustępy.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections import Counter

log = logging.getLogger("lexrag")
BAZA = "https://api.sejm.gov.pl/eli/acts"
# nagłówek załącznika bywa mniejszą czcionką – zostaje mimo filtra rozmiaru
RE_ZALACZNIK_SPAN = re.compile(r"^\s*Załącznik(?:i)?\b")


def _pobierz(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "lexrag/0.2", "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def html_do_tekstu(h: str) -> str:
    """Prosta konwersja HTML → tekst z zachowaniem podziału na bloki/linie."""
    h = re.sub(r"(?is)<(script|style).*?</\1>", " ", h)
    h = re.sub(r"(?is)<sup[^>]*>.*?</sup>", " ", h)      # znaczniki przypisów
    h = re.sub(r"(?i)<br\s*/?>", "\n", h)
    h = re.sub(r"(?i)</(p|div|li|tr|h\d|section|article)>", "\n", h)
    h = re.sub(r"(?s)<[^>]+>", " ", h)
    t = html.unescape(h)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    return "\n".join(l.strip() for l in t.splitlines())


def _pymupdf():
    try:
        import pymupdf  # type: ignore
        return pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore  # starsze wersje
            return pymupdf
        except ImportError as e:
            raise RuntimeError("Do PDF potrzebny jest PyMuPDF: pip install pymupdf") from e


def pdf_do_tekstu(sciezka: str, tolerancja_pt: float = 0.5) -> str:
    """PDF → tekst liniami, bez przypisów (mniejsza czcionka) i bez indeksów górnych.

    Dominujący rozmiar czcionki liczony jest po liczbie znaków w całym dokumencie;
    zostają spany o rozmiarze >= dominujący - tolerancja_pt (tytuły są większe, więc
    zostają), odpadają mniejsze (przypisy) i spany z flagą indeksu górnego.
    """
    pymupdf = _pymupdf()
    doc = pymupdf.open(sciezka)
    hist: Counter = Counter()
    strony: list[list[tuple]] = []
    for page in doc:
        spany = []
        for b in page.get_text("dict")["blocks"]:
            if b.get("type", 0) != 0:
                continue
            for l in b.get("lines", []):
                for s in l["spans"]:
                    t = s["text"]
                    if not t.strip():
                        continue
                    hist[round(s["size"], 1)] += len(t)
                    spany.append((s["bbox"], s["size"], s["flags"], t))
        strony.append(spany)
    if not hist:
        return ""
    glowny = hist.most_common(1)[0][0]
    prog = glowny - tolerancja_pt
    log.debug("PDF: dominujący rozmiar czcionki %.1f pt, próg %.1f pt", glowny, prog)

    linie: list[str] = []
    for spany in strony:
        zostaw = [(bb, t) for bb, sz, fl, t in spany
                  if (sz >= prog and not (fl & 1)) or RE_ZALACZNIK_SPAN.match(t)]
        zostaw.sort(key=lambda x: (round((x[0][1] + x[0][3]) / 2), x[0][0]))
        wiersze: list[list] = []
        for bb, t in zostaw:
            yc = (bb[1] + bb[3]) / 2
            if wiersze and abs(wiersze[-1][0] - yc) <= 2.5:
                wiersze[-1][1].append((bb[0], t))
            else:
                wiersze.append([yc, [(bb[0], t)]])
        linie_strony: list[str] = []
        for _, czesci in wiersze:
            czesci.sort(key=lambda x: x[0])
            linia = " ".join(c.strip() for _, c in czesci if c.strip())
            if linia:
                linie_strony.append(linia)
        # nagłówek załącznika na początek strony: tabele obrócone o 90° mają inną kolejność y,
        # a parser musi zobaczyć "Załącznik" przed pierwszym wierszem tabeli
        zal = [l for l in linie_strony if RE_ZALACZNIK_SPAN.match(l)]
        if zal:
            linie_strony = zal + [l for l in linie_strony if not RE_ZALACZNIK_SPAN.match(l)]
        linie.extend(linie_strony)
        linie.append("")
    return "\n".join(linie)


def wczytaj_dowolny(sciezka: str) -> str:
    dolne = sciezka.lower()
    if dolne.endswith(".pdf"):
        return pdf_do_tekstu(sciezka)
    if dolne.endswith(".docx"):
        import shutil
        import subprocess
        if not shutil.which("pandoc"):
            raise RuntimeError("Do DOCX potrzebny jest pandoc (brew install pandoc) albo zapisz plik jako PDF/TXT.")
        return subprocess.run(["pandoc", sciezka, "-t", "plain", "--wrap=none"],
                              check=True, capture_output=True, text=True).stdout
    with open(sciezka, encoding="utf-8", errors="replace") as f:
        dane = f.read()
    if dolne.endswith((".html", ".htm")):
        return html_do_tekstu(dane)
    return dane


def pobierz_akt(ident: str, katalog: str) -> str:
    """Pobiera metadane + tekst (PDF, a gdy brak – HTML). Zwraca ścieżkę do pliku .txt."""
    ident = ident.strip("/")
    os.makedirs(katalog, exist_ok=True)
    nazwa = ident.replace("/", "_")
    try:
        meta = json.loads(_pobierz(f"{BAZA}/{ident}").decode("utf-8"))
        with open(os.path.join(katalog, nazwa + ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        log.info("Akt: %s", meta.get("title", "?"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API Sejmu: HTTP {e.code} dla {ident} – sprawdź identyfikator (np. DU/2024/1631).") from e

    tekst = ""
    # PDF jest źródłem pierwszego wyboru: ma rozmiary czcionek, więc da się odciąć przypisy
    try:
        pdf = os.path.join(katalog, nazwa + ".pdf")
        with open(pdf, "wb") as f:
            f.write(_pobierz(f"{BAZA}/{ident}/text.pdf"))
        tekst = pdf_do_tekstu(pdf)
    except urllib.error.HTTPError:
        pass
    if len(tekst) < 2000:
        try:
            h = _pobierz(f"{BAZA}/{ident}/text.html").decode("utf-8", "replace")
            with open(os.path.join(katalog, nazwa + ".html"), "w", encoding="utf-8") as f:
                f.write(h)
            tekst = html_do_tekstu(h)
        except urllib.error.HTTPError:
            pass
    if len(tekst) < 2000:
        raise RuntimeError(f"Nie udało się pobrać tekstu aktu {ident} (PDF ani HTML).")
    wyj = os.path.join(katalog, nazwa + ".txt")
    with open(wyj, "w", encoding="utf-8") as f:
        f.write(tekst)
    log.info("Zapisano %s (%d znaków)", wyj, len(tekst))
    return wyj
