"""Pomiar jakości: (1) wyszukiwania i (2) odpowiedzi – na złotym zbiorze pytań.

Format pliku (JSONL), jedna linia = jedno pytanie:
    {"pytanie": "za jaki okres można żądać zwrotu", "oczekiwane": ["fus:art138:ust4"],
     "musi_zawierac": ["3 lata|trzech lat", "12 miesięcy"], "nie_moze_zawierac": ["5 lat"], "uwagi": "..."}

"oczekiwane" to id jednostek, które MUSZĄ znaleźć się w kontekście, żeby model mógł
odpowiedzieć poprawnie. "musi_zawierac" to frazy (alternatywy po "|"), które muszą paść w
odpowiedzi; "nie_moze_zawierac" – które nie mogą.

Metryki wyszukiwania:
    recall@k (dowolne)  – choć jedna oczekiwana jednostka w top-k
    recall@k (wszystkie)– wszystkie oczekiwane w top-k   <- ta liczy się dla pytań złożonych
    MRR                 – 1/pozycja pierwszej oczekiwanej
Bramka na start: recall@10 (wszystkie) >= 0,95. Dopiero potem warto włączać LLM.

Metryki odpowiedzi (eval-odpowiedzi, wymaga LLM):
    pokrycie   – odsetek pytań, na które system odpowiedział (reszta: dopytanie / odmowa / błąd)
    precyzja   – odsetek odpowiedzi, które cytują oczekiwany przepis (lub jego artykuł) i zawierają
                 wymagane frazy, a nie zawierają zakazanych   <- "99 %" dotyczy tej liczby
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .index import Indeks
from .retrieve import rerank, rozszerz, szukaj


@dataclass
class WynikPytania:
    pytanie: str
    oczekiwane: list[str]
    ranking: list[str]
    w_kontekscie: list[str] = field(default_factory=list)

    def pozycje(self) -> list[Optional[int]]:
        """Pozycja każdego oczekiwanego id w rankingu; wpis „a|b” to alternatywa – liczy się lepsza pozycja."""
        out: list[Optional[int]] = []
        for o in self.oczekiwane:
            poz = [self.ranking.index(a) + 1 for a in o.split("|") if a in self.ranking]
            out.append(min(poz) if poz else None)
        return out


def wczytaj_zloty(sciezka: str) -> list[dict]:
    out = []
    with open(sciezka, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l and not l.startswith("#"):
                out.append(json.loads(l))
    return out


# --- wyszukiwanie ------------------------------------------------------------------------
def ewaluuj(indeks: Indeks, zloty: list[dict], uzyj_emb: bool = True, uzyj_reranker: bool = True,
            k_lista: tuple[int, ...] = (1, 3, 5, 10, 20), na_dzien: Optional[str] = None,
            warianty_fn=None) -> tuple[dict, list[WynikPytania]]:
    wyniki: list[WynikPytania] = []
    for w in zloty:
        pytanie = w["pytanie"]
        zapytania = [pytanie]
        if warianty_fn:
            try:
                zapytania += [z for z in warianty_fn(pytanie) if z]
            except Exception:
                pass
        kandydaci = szukaj(indeks, zapytania, na_dzien=na_dzien, uzyj_emb=uzyj_emb)
        if uzyj_reranker:
            top = rerank(indeks, pytanie, kandydaci, top_n=max(k_lista))
            # reranker zwraca top_n; ogon zostaje z RRF, żeby recall@20 miał sens
            ids = [t.id for t in top] + [t.id for t in kandydaci if t.id not in {x.id for x in top}]
            do_kontekstu = top[:8]
        else:
            ids = [t.id for t in kandydaci]
            do_kontekstu = kandydaci[:8]
        kontekst = rozszerz(indeks, do_kontekstu, na_dzien=na_dzien, pytanie=pytanie)
        wyniki.append(WynikPytania(pytanie, list(w["oczekiwane"]), ids[: max(k_lista) * 2],
                                   [j.id for j in kontekst]))

    n = len(wyniki) or 1
    podsumowanie: dict = {"liczba_pytan": len(wyniki)}
    for k in k_lista:
        dowolne = sum(1 for r in wyniki if any(p is not None and p <= k for p in r.pozycje()))
        wszystkie = sum(1 for r in wyniki if all(p is not None and p <= k for p in r.pozycje()))
        podsumowanie[f"recall@{k}_dowolne"] = dowolne / n
        podsumowanie[f"recall@{k}_wszystkie"] = wszystkie / n
    mrr = 0.0
    for r in wyniki:
        poz = [p for p in r.pozycje() if p is not None]
        mrr += (1.0 / min(poz)) if poz else 0.0
    podsumowanie["mrr"] = mrr / n
    podsumowanie["w_kontekscie_wszystkie"] = sum(
        1 for r in wyniki if all(any(a in r.w_kontekscie for a in o.split("|")) for o in r.oczekiwane)) / n
    return podsumowanie, wyniki


def raport(podsumowanie: dict, wyniki: list[WynikPytania], pokaz_bledy: bool = True) -> str:
    linie = [f"Pytań: {podsumowanie['liczba_pytan']}"]
    for k in (1, 3, 5, 10, 20):
        a = podsumowanie.get(f"recall@{k}_dowolne")
        b = podsumowanie.get(f"recall@{k}_wszystkie")
        if a is not None:
            linie.append(f"  recall@{k:<3} dowolne: {a:6.1%}   wszystkie: {b:6.1%}")
    linie.append(f"  MRR: {podsumowanie['mrr']:.3f}")
    linie.append(f"  komplet oczekiwanych w kontekście LLM (po rozszerzeniu): "
                 f"{podsumowanie['w_kontekscie_wszystkie']:.1%}")
    if pokaz_bledy:
        zle = [r for r in wyniki if not all(p is not None and p <= 10 for p in r.pozycje())]
        if zle:
            linie.append("\nPoza top-10:")
            for r in zle:
                poz = ", ".join(f"{o}→{p if p else '–'}" for o, p in zip(r.oczekiwane, r.pozycje()))
                linie.append(f"  • {r.pytanie}\n      oczekiwane: {poz}\n      top-3: {', '.join(r.ranking[:3])}")
    return "\n".join(linie)


# --- odpowiedzi (pełny przebieg z LLM) ---------------------------------------------------
def _fraza_w(tekst: str, fraza: str) -> bool:
    t = " ".join(tekst.lower().split())
    return any(" ".join(alt.lower().split()) in t for alt in fraza.split("|") if alt.strip())


def ewaluuj_odpowiedzi(indeks: Indeks, zloty: list[dict], na_dzien: Optional[str] = None,
                       uzyj_emb: bool = True, uzyj_reranker: bool = True, dopytuj: bool = False,
                       weryfikacja: bool = True, limit: Optional[int] = None,
                       postep: Optional[Callable[[int, int, dict], None]] = None,
                       analiza: Optional[bool] = None) -> list[dict]:
    from .generate import odpowiedz
    pytania = zloty[:limit] if limit else zloty
    wyniki: list[dict] = []
    for i, w in enumerate(pytania, 1):
        t0 = time.time()
        o = odpowiedz(indeks, w["pytanie"], na_dzien=na_dzien, uzyj_emb=uzyj_emb, uzyj_reranker=uzyj_reranker,
                      dopytuj=dopytuj, weryfikacja=weryfikacja, analiza=analiza)
        oczek = list(w.get("oczekiwane", []))
        cyt = [c.id for c in o.cytaty]
        kont = [k.id for k in o.kontekst]

        def cytowano(ident: str) -> bool:
            j = indeks.jednostki.get(ident)
            if ident in cyt:
                return True
            if j and j.rodzic and j.rodzic in cyt:       # cytowany cały artykuł
                return True
            return any(c.startswith(ident + ":") for c in cyt)  # cytowany ustęp oczekiwanego artykułu

        musi = [f for f in w.get("musi_zawierac", []) if f]
        nie = [f for f in w.get("nie_moze_zawierac", []) if f]
        cytat_ok = all(any(cytowano(a) for a in x.split("|")) for x in oczek) if oczek else None
        kontekst_ok = all(any(a in kont for a in x.split("|")) for x in oczek) if oczek else None
        brak_fraz = [f for f in musi if not _fraza_w(o.tekst, f)]
        zakazane = [f for f in nie if _fraza_w(o.tekst, f)]
        frazy_ok = (not brak_fraz and not zakazane) if (musi or nie) else None
        odpowiedziano = o.tryb == "odpowiedz"
        r = {
            "pytanie": w["pytanie"], "tryb": o.tryb, "odpowiedz": o.tekst, "uwagi": o.uwagi,
            "cytowane": cyt, "oczekiwane": oczek, "cytat_ok": cytat_ok, "kontekst_ok": kontekst_ok,
            "brak_fraz": brak_fraz, "zakazane_frazy": zakazane, "frazy_ok": frazy_ok,
            "zdania_usuniete": o.zdania_usuniete, "warianty": o.warianty,
            "status_weryfikacji": o.status_weryfikacji, "naprawa": o.naprawa, "przeglad": o.przeglad,
            "ostrzezenia": [x["kod"] for x in o.ostrzezenia],
            "poprawna": odpowiedziano and cytat_ok is not False and frazy_ok is not False,
            "czas_s": round(time.time() - t0, 1), "czasy": o.czasy,
        }
        wyniki.append(r)
        if postep:
            postep(i, len(pytania), r)
    return wyniki


def raport_odpowiedzi(wyniki: list[dict]) -> str:
    n = len(wyniki) or 1
    tryby: dict[str, int] = {}
    for r in wyniki:
        tryby[r["tryb"]] = tryby.get(r["tryb"], 0) + 1
    odp = [r for r in wyniki if r["tryb"] == "odpowiedz"]
    linie = [f"Pytań: {len(wyniki)}   tryby: " + ", ".join(f"{k}: {v}" for k, v in sorted(tryby.items()))]
    linie.append(f"  pokrycie (udzielono odpowiedzi): {len(odp) / n:.1%}")
    if odp:
        m = len(odp)
        c_ok = sum(1 for r in odp if r["cytat_ok"] is not False)
        f_ok = sum(1 for r in odp if r["frazy_ok"] is not False)
        p_ok = sum(1 for r in odp if r["poprawna"])
        linie.append(f"  wśród odpowiedzi: cytat trafny {c_ok / m:.1%}   frazy OK {f_ok / m:.1%}   "
                     f"POPRAWNE (cytat i frazy) {p_ok / m:.1%}")
        linie.append(f"  kontekst zawierał komplet oczekiwanych: "
                     f"{sum(1 for r in wyniki if r['kontekst_ok'] is not False) / n:.1%}")
        linie.append(f"  średni czas: {sum(r['czas_s'] for r in wyniki) / n:.1f} s/pytanie; "
                     f"zdań usuniętych łącznie: {sum(len(r['zdania_usuniete']) for r in wyniki)}")
        statusy: dict[str, int] = {}
        for r in wyniki:
            s = r.get("status_weryfikacji") or "-"
            statusy[s] = statusy.get(s, 0) + 1
        linie.append("  statusy weryfikacji: " + ", ".join(f"{k}: {v}" for k, v in sorted(statusy.items()))
                     + f"; odpowiedzi po cyklu naprawczym: {sum(1 for r in wyniki if r.get('naprawa'))}")
        zw = [r for r in odp if r.get('status_weryfikacji') == 'verified']
        linie.append(f"  zweryfikowane (verified) wśród odpowiedzi: {len(zw)}/{m}; "
                     f"poprawne i zweryfikowane: {sum(1 for r in zw if r['poprawna'])}")
    zle = [r for r in wyniki if r["tryb"] == "odpowiedz" and not r["poprawna"]]
    if zle:
        linie.append("\nOdpowiedzi NIEPOPRAWNE (do przejrzenia):")
        for r in zle:
            powod = []
            if r["cytat_ok"] is False:
                powod.append(f"cytowano {r['cytowane']} zamiast {r['oczekiwane']}")
            if r["brak_fraz"]:
                powod.append(f"brak fraz {r['brak_fraz']}")
            if r["zakazane_frazy"]:
                powod.append(f"zakazane frazy {r['zakazane_frazy']}")
            linie.append(f"  • {r['pytanie']}\n      {'; '.join(powod)}\n      odpowiedź: {r['odpowiedz'][:300]}")
    inne = [r for r in wyniki if r["tryb"] != "odpowiedz"]
    if inne:
        linie.append("\nBez odpowiedzi:")
        for r in inne:
            linie.append(f"  • [{r['tryb']}] {r['pytanie']}\n      {r['odpowiedz'][:200]}")
    return "\n".join(linie)
