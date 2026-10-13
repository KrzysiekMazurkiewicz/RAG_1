"""DOCX/PDF procedur → Markdown. Próbuje kolejno: pandoc (DOCX), docling (DOCX/PDF, zachowuje
tabele), PyMuPDF (PDF, sam tekst). Wynik to .md gotowy dla `parse-doc`.

Dokumenty wewnętrzne NIE opuszczają sieci firmowej – wszystko dzieje się lokalnie.
"""
from __future__ import annotations

import logging
import shutil
import subprocess

log = logging.getLogger("lexrag")


def do_markdown(wejscie: str, wyjscie: str) -> str:
    dolne = wejscie.lower()
    if dolne.endswith(".docx") and shutil.which("pandoc"):
        subprocess.run(["pandoc", wejscie, "-t", "gfm", "--wrap=none", "-o", wyjscie], check=True)
        log.info("pandoc → %s", wyjscie)
        return wyjscie
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        wynik = DocumentConverter().convert(wejscie)
        md = wynik.document.export_to_markdown()
        with open(wyjscie, "w", encoding="utf-8") as f:
            f.write(md)
        log.info("docling → %s", wyjscie)
        return wyjscie
    except ImportError:
        pass
    if dolne.endswith(".pdf"):
        from .fetch import pdf_do_tekstu
        with open(wyjscie, "w", encoding="utf-8") as f:
            f.write(pdf_do_tekstu(wejscie))
        log.warning("PDF przez PyMuPDF – bez nagłówków i tabel; lepiej docling.")
        return wyjscie
    raise RuntimeError("Zainstaluj pandoc (DOCX) albo `pip install docling` (DOCX/PDF z tabelami).")
