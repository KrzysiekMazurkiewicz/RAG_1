"""Minimalny, czytelny BM25 (Okapi) na indeksie odwróconym.

~80 linii zamiast zależności: korpus ustawy to 1–5 tys. jednostek, więc
nic szybszego nie jest potrzebne, a każdy może przeczytać, jak liczony
jest wynik. Zapisywany do JSON.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_dok = 0
        self.dl: list[int] = []
        self.srednia_dl = 0.0
        self.postings: dict[str, dict[int, int]] = {}
        self.idf: dict[str, float] = {}

    # --- budowa ----------------------------------------------------------
    def dopasuj(self, dokumenty: list[list[str]]) -> "BM25":
        self.n_dok = len(dokumenty)
        self.dl = [len(d) for d in dokumenty]
        self.srednia_dl = (sum(self.dl) / self.n_dok) if self.n_dok else 0.0
        postings: dict[str, dict[int, int]] = defaultdict(dict)
        for i, d in enumerate(dokumenty):
            for t in d:
                postings[t][i] = postings[t].get(i, 0) + 1
        self.postings = dict(postings)
        self.idf = {
            t: math.log(1.0 + (self.n_dok - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }
        return self

    # --- wyszukiwanie ----------------------------------------------------
    def wyniki(self, zapytanie: list[str]) -> dict[int, float]:
        """Wynik dla każdego dokumentu, który ma choć jeden token zapytania."""
        akum: dict[int, float] = defaultdict(float)
        for t in zapytanie:
            p = self.postings.get(t)
            if not p:
                continue
            idf = self.idf[t]
            for i, tf in p.items():
                norm = 1.0 - self.b + self.b * (self.dl[i] / self.srednia_dl if self.srednia_dl else 1.0)
                akum[i] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return dict(akum)

    def top(self, zapytanie: list[str], k: int = 30) -> list[tuple[int, float]]:
        w = self.wyniki(zapytanie)
        return sorted(w.items(), key=lambda x: -x[1])[:k]

    # --- (de)serializacja ------------------------------------------------
    def zapisz(self, sciezka: str) -> None:
        dane = {
            "k1": self.k1, "b": self.b, "n_dok": self.n_dok, "dl": self.dl,
            "postings": {t: [[i, tf] for i, tf in p.items()] for t, p in self.postings.items()},
        }
        with open(sciezka, "w", encoding="utf-8") as f:
            json.dump(dane, f, ensure_ascii=False)

    @classmethod
    def wczytaj(cls, sciezka: str) -> "BM25":
        with open(sciezka, encoding="utf-8") as f:
            d = json.load(f)
        o = cls(d["k1"], d["b"])
        o.n_dok = d["n_dok"]
        o.dl = d["dl"]
        o.srednia_dl = (sum(o.dl) / o.n_dok) if o.n_dok else 0.0
        o.postings = {t: {int(i): tf for i, tf in p} for t, p in d["postings"].items()}
        o.idf = {
            t: math.log(1.0 + (o.n_dok - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in o.postings.items()
        }
        return o
