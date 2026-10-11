"""Porównanie modeli na podstawie logów z `eval-odpowiedzi` (JSONL).

    python scripts/porownaj_modele.py data/eval_odpowiedzi_fus_*.jsonl

Dla każdego logu: liczba pytań, pokrycie (odpowiedzi / pytania), precyzja (poprawne / odpowiedzi),
średni czas na pytanie. Nazwę modelu bierze z nazwy pliku (eval_odpowiedzi_<zbior>_<model>.jsonl).
"""
from __future__ import annotations

import json
import os
import sys


def wczytaj(sciezka: str) -> list[dict]:
    with open(sciezka, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(pliki: list[str]) -> None:
    print(f"{'log':<42} {'pytań':>5} {'odpow.':>6} {'poprawne':>8} {'precyzja':>9} {'pokrycie':>9} {'s/pyt':>6}")
    for p in pliki:
        rs = wczytaj(p)
        if not rs:
            continue
        n = len(rs)
        odp = [r for r in rs if r["tryb"] == "odpowiedz"]
        popr = sum(1 for r in odp if r.get("poprawna"))
        prec = popr / len(odp) if odp else 0.0
        czas = sum(r.get("czas_s", 0) for r in rs) / n
        print(f"{os.path.basename(p):<42} {n:>5} {len(odp):>6} {popr:>8} {prec:>8.0%} {len(odp) / n:>8.0%} {czas:>6.0f}")
    print("\nprecyzja = poprawne / udzielone odpowiedzi (to jest '99 %'); pokrycie = udzielone / wszystkie pytania")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(f"data/{x}" for x in os.listdir("data") if x.startswith("eval_odpowiedzi") and x.endswith(".jsonl")))
