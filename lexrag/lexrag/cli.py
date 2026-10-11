"""Interfejs linii poleceń: python -m lexrag <polecenie> ...

    wgraj          ustawa (.pdf/.txt/.html/.docx) → jednostki JSONL → indeks; jedno polecenie na start
    chat           rozmowa z indeksem (pytania jedno po drugim, modele ładowane raz)
    ask            jedno pytanie (retrieval-only bez LLM; z LLM: odpowiedź z cytatami)
    llm-test       sprawdź połączenie z modelem on-premise (tryb API, czas, JSON)
    eval           recall@k wyszukiwania na złotym zbiorze
    eval-odpowiedzi pełny przebieg z LLM na złotym zbiorze: pokrycie i precyzja odpowiedzi
    fetch          pobierz akt z API Sejmu (DU/rok/pozycja) → data/raw/*.txt
    parse          ustawa → jednostki JSONL (bez indeksu)
    parse-doc      procedura (.md/.docx/.pdf) → jednostki JSONL
    contextualize  dopisz opisy kontekstowe (wymaga LLM)
    index          zbuduj indeks z plików JSONL (BM25 + embeddingi)
    show           pokaż jednostkę po id
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import KONFIG
from .models import zapisz_jsonl


def _log(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    for halas in ("sentence_transformers", "transformers", "httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(halas).setLevel(logging.WARNING)


def _skrot(t: str, n: int = 140) -> str:
    t = " ".join(t.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _parsuj_do_jsonl(plik: str, kod: str, akt: str, out: str, obowiazuje_od, obowiazuje_do, wersja: str = "") -> tuple[list, dict]:
    from .fetch import wczytaj_dowolny
    from .parse_ustawa import parsuj_ustawe, statystyki
    tekst = wczytaj_dowolny(plik)
    jednostki = parsuj_ustawe(tekst, akt or "", kod, obowiazuje_od, obowiazuje_do, wersja=wersja)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n = zapisz_jsonl(jednostki, out)
    st = statystyki(jednostki)
    print(f"{out}: {n} jednostek, {st}")
    arts = [j for j in jednostki if j.poziom == "art"]
    if arts:
        print("  artykuły: " + ", ".join(j.id.split(':')[1] for j in arts[:6]) +
              " … " + ", ".join(j.id.split(':')[1] for j in arts[-3:]))
    if n < 5:
        print("UWAGA: bardzo mało jednostek – sprawdź, czy plik ma 'Art. N.' na początku linii.", file=sys.stderr)
    return jednostki, st


def cmd_fetch(a):
    from .fetch import pobierz_akt
    print(pobierz_akt(a.ident, a.out))


def cmd_parse(a):
    out = a.out or f"data/units/{a.kod}.jsonl"
    _parsuj_do_jsonl(a.plik, a.kod, a.akt, out, a.obowiazuje_od, a.obowiazuje_do, wersja=a.wersja)


def cmd_wgraj(a):
    from .index import Indeks
    out = a.out or f"data/units/{a.kod}.jsonl"
    _parsuj_do_jsonl(a.plik, a.kod, a.akt, out, a.obowiazuje_od, a.obowiazuje_do, wersja=a.wersja)
    pliki = [out] + list(a.dolacz or [])
    idx = Indeks.zbuduj(pliki, a.index, z_embeddingami=not a.bez_emb)
    print(f"Indeks: {len(idx)} jednostek szukalnych z {len(idx.jednostki)} ogółem → {a.index} "
          f"(embeddingi: {'tak' if idx.emb is not None else 'nie'}, lematyzacja: {idx.meta['lematyzacja']})")
    print(f"Sprawdź wyrywkowo: python -m lexrag show {a.kod}:art1 --index {a.index}")
    print(f"Pytaj:            python -m lexrag chat --index {a.index}")


def cmd_parse_doc(a):
    from .parse_doc import parsuj_markdown
    plik = a.plik
    if not plik.lower().endswith((".md", ".txt", ".markdown")):
        from .convert import do_markdown
        plik = do_markdown(plik, os.path.splitext(plik)[0] + ".md")
    with open(plik, encoding="utf-8") as f:
        tekst = f.read()
    jednostki = parsuj_markdown(tekst, a.dokument or os.path.basename(plik), a.kod,
                                max_znakow=a.max_znakow, obowiazuje_od=a.obowiazuje_od,
                                obowiazuje_do=a.obowiazuje_do, wersja=a.wersja or "")
    out = a.out or f"data/units/{a.kod}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n = zapisz_jsonl(jednostki, out)
    print(f"{out}: {n} jednostek ({sum(1 for j in jednostki if j.szukalny)} szukalnych)")


def cmd_contextualize(a):
    from .contextualize import kontekstualizuj_plik
    from .llm import llm_dostepny
    if not llm_dostepny():
        raise SystemExit("Ustaw LEXRAG_LLM_MODEL (i LEXRAG_LLM_BASE_URL) – opisy generuje LLM.")
    n = kontekstualizuj_plik(a.plik, limit=a.limit, nadpisz=a.nadpisz)
    print(f"Opisano {n} jednostek w {a.plik}. Teraz przebuduj indeks: python -m lexrag index ...")


def cmd_index(a):
    from .index import Indeks
    idx = Indeks.zbuduj(a.pliki, a.out, z_embeddingami=not a.bez_emb)
    print(f"Indeks: {len(idx)} jednostek szukalnych z {len(idx.jednostki)} ogółem → {a.out} "
          f"(embeddingi: {'tak' if idx.emb is not None else 'nie'}, lematyzacja: {idx.meta['lematyzacja']})")


def wypisz_odpowiedz(w, idx, pokaz_kontekst: bool = False) -> None:
    if w.tryb == "retrieval-only":
        print(f"[retrieval-only – brak LLM] top-{len(w.trafienia)}:")
        for i, t in enumerate(w.trafienia, 1):
            j = idx.jednostki[t.id]
            print(f"{i:2}. {t.id:<28} {t.wynik:7.3f}  {j.sciezka}\n     {_skrot(j.tekst)}")
        if pokaz_kontekst:
            print(f"\nKontekst po rozszerzeniu ({len(w.kontekst)} jednostek, "
                  f"{sum(len(j.tekst) for j in w.kontekst)} znaków):")
            for j in w.kontekst:
                print(f"  - {j.id}  ({j.sciezka})")
        return

    czasy = ", ".join(f"{k} {v}s" for k, v in w.czasy.items())
    print(f"[{w.tryb}]" + (f"  ({czasy})" if czasy else ""))
    if w.warianty and pokaz_kontekst:
        print("warianty zapytania:", "; ".join(w.warianty))
    if w.tryb == "dopytanie":
        print("Brakuje:", ", ".join(w.brakujace_parametry))
        print("Pytanie do użytkownika:", w.tekst)
        return
    print("\n" + w.tekst)
    if getattr(w, "status_weryfikacji", ""):
        print(f"status weryfikacji: {w.status_weryfikacji}" + (" (odpowiedź po cyklu naprawczym)" if getattr(w, "naprawa", False) else ""))
    if getattr(w, "przeglad", False):
        print("(odpowiedź przeglądowa – pytanie ogólne albo bez danych o sytuacji)")
    if getattr(w, "dopytanie", ""):
        print("Dopytanie:", w.dopytanie)
    if w.uwagi:
        print("uwagi:", w.uwagi)
    if w.cytaty:
        print("\nPodstawa (tekst z indeksu, nie z modelu):")
        for j in w.cytaty:
            print(f"  [{j.id}] {j.sciezka}\n    {j.tekst}")
    if w.zdania_usuniete:
        print(f"\nUsunięto {len(w.zdania_usuniete)} zdań bez pokrycia:")
        for z in w.zdania_usuniete:
            print("  ✗", z)
    if pokaz_kontekst:
        print("\nTrafienia:", ", ".join(f"{t.id}({t.wynik:.2f})" for t in w.trafienia))
        print("Kontekst:", ", ".join(j.id for j in w.kontekst))


def _opcje_odpowiedzi(a) -> dict:
    return dict(na_dzien=a.na_dzien, uzyj_emb=not a.bez_emb, uzyj_reranker=not a.bez_reranker,
                dopytuj=not a.bez_dopytania, weryfikacja=not a.bez_weryfikacji, top_n=a.top,
                uzyj_llm=not a.bez_llm, analiza=not a.bez_analizy)


def cmd_ask(a):
    from .generate import odpowiedz
    from .index import Indeks
    idx = Indeks.wczytaj(a.index)
    w = odpowiedz(idx, a.pytanie, **_opcje_odpowiedzi(a))
    if getattr(a, "slad", None):
        with open(a.slad, "w", encoding="utf-8") as f:
            json.dump(w.slad, f, ensure_ascii=False, indent=1, default=str)
        print(f"ślad diagnostyczny: {a.slad}", file=sys.stderr)
    if a.json:
        print(json.dumps({"tryb": w.tryb, "tekst": w.tekst, "status_weryfikacji": w.status_weryfikacji, "naprawa": w.naprawa,
                          "cytaty": [{"id": j.id, "sciezka": j.sciezka, "tekst": j.tekst} for j in w.cytaty],
                          "zdania_usuniete": w.zdania_usuniete, "uwagi": w.uwagi, "czasy": w.czasy,
                          "trafienia": [t.id for t in w.trafienia]}, ensure_ascii=False, indent=1))
        return
    wypisz_odpowiedz(w, idx, a.pokaz_kontekst)


def cmd_chat(a):
    from .generate import odpowiedz
    from .index import Indeks
    from .llm import llm_dostepny, tryb_api
    try:
        import readline  # noqa: F401  (edycja linii i historia w terminalu)
    except ImportError:
        pass
    idx = Indeks.wczytaj(a.index)
    llm = f"{KONFIG.llm_model} ({tryb_api()}, {KONFIG.llm_base_url})" if llm_dostepny() and not a.bez_llm else "brak – tryb retrieval-only"
    print(f"Indeks: {len(idx)} jednostek szukalnych ({len(idx.jednostki)} ogółem), embeddingi: "
          f"{'tak' if idx.emb is not None and not a.bez_emb else 'nie'}. LLM: {llm}")
    print("Pytaj. Polecenia: /q – koniec, /pokaz <id> – treść jednostki, /kontekst – więcej szczegółów.")
    pokaz = a.pokaz_kontekst
    while True:
        try:
            p = input("\npytanie> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not p:
            continue
        if p in ("/q", "/quit", "/wyjdz", "/exit"):
            break
        if p == "/kontekst":
            pokaz = not pokaz
            print("szczegóły:", "włączone" if pokaz else "wyłączone")
            continue
        if p.startswith("/pokaz"):
            ident = p.split(maxsplit=1)[1].strip() if len(p.split()) > 1 else ""
            j = idx.jednostki.get(ident)
            print(f"{j.id}  {j.sciezka}\n{j.tekst}" if j else f"Brak jednostki {ident!r}")
            continue
        w = odpowiedz(idx, p, **_opcje_odpowiedzi(a))
        wypisz_odpowiedz(w, idx, pokaz)


def cmd_llm_test(a):
    from .llm import BrakLLM, llm_dostepny, ping
    if not llm_dostepny():
        raise SystemExit("Ustaw LEXRAG_LLM_MODEL (np. export LEXRAG_LLM_MODEL=hf.co/ConwayResearch/Underdog-Saluki-27B-1.0).")
    try:
        w = ping(a.model)
    except (BrakLLM, ValueError) as e:
        raise SystemExit(f"LLM nie działa: {e}")
    print(json.dumps(w, ensure_ascii=False))
    print("OK – model odpowiada poprawnym JSON.")


def cmd_eval(a):
    from .evaluate import ewaluuj, raport, wczytaj_zloty
    from .index import Indeks
    idx = Indeks.wczytaj(a.index)
    zloty = wczytaj_zloty(a.plik)
    warianty_fn = None
    if a.warianty_llm:
        from .generate import analizuj_pytanie
        warianty_fn = lambda p: analizuj_pytanie(p).get("warianty", [])
    pods, wyniki = ewaluuj(idx, zloty, uzyj_emb=not a.bez_emb, uzyj_reranker=not a.bez_reranker,
                           na_dzien=a.na_dzien, warianty_fn=warianty_fn)
    tryb = f"BM25{'' if a.bez_emb else '+emb'}{'' if a.bez_reranker else '+rerank'}{'+wariantyLLM' if a.warianty_llm else ''}"
    print(f"Konfiguracja: {tryb}")
    print(raport(pods, wyniki, pokaz_bledy=not a.cicho))


def cmd_eval_odp(a):
    from .evaluate import ewaluuj_odpowiedzi, raport_odpowiedzi, wczytaj_zloty
    from .index import Indeks
    from .llm import llm_dostepny
    if not llm_dostepny():
        raise SystemExit("eval-odpowiedzi wymaga LLM – ustaw LEXRAG_LLM_MODEL.")
    idx = Indeks.wczytaj(a.index)
    zloty = wczytaj_zloty(a.plik)
    log_f = open(a.log, "w", encoding="utf-8") if a.log else None

    def postep(i, n, r):
        znak = "✓" if r["poprawna"] else ("–" if r["tryb"] != "odpowiedz" else "✗")
        print(f"{i:3}/{n} {znak} [{r['tryb']}] {r['czas_s']:6.1f}s  {_skrot(r['pytanie'], 70)}", flush=True)
        if log_f:
            log_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log_f.flush()

    wyniki = ewaluuj_odpowiedzi(idx, zloty, na_dzien=a.na_dzien, uzyj_emb=not a.bez_emb,
                                uzyj_reranker=not a.bez_reranker, dopytuj=a.dopytuj,
                                weryfikacja=not a.bez_weryfikacji, limit=a.limit, postep=postep,
                                analiza=not a.bez_analizy)
    if log_f:
        log_f.close()
    print()
    print(raport_odpowiedzi(wyniki))


def cmd_show(a):
    from .index import Indeks
    idx = Indeks.wczytaj(a.index)
    j = idx.jednostki.get(a.id)
    if not j:
        raise SystemExit(f"Brak jednostki {a.id}")
    print(f"{j.id}  [{j.poziom}]  szukalny={j.szukalny}\n{j.sciezka}\nopis: {j.opis or '(brak)'}\n"
          f"rodzic: {j.rodzic}\nodwołania: {j.odwolania}\nzewn.: {j.odwolania_zewn}\n\n{j.tekst}")


def _opcje_pytania(s):
    s.add_argument("--index", default="data/index")
    s.add_argument("--na-dzien", help="stan prawny na dzień YYYY-MM-DD")
    s.add_argument("--top", type=int); s.add_argument("--bez-emb", action="store_true")
    s.add_argument("--bez-reranker", action="store_true"); s.add_argument("--bez-llm", action="store_true")
    s.add_argument("--bez-dopytania", action="store_true"); s.add_argument("--bez-weryfikacji", action="store_true")
    s.add_argument("--bez-analizy", action="store_true", help="pomiń krok analizy pytania (szybciej na CPU)")
    s.add_argument("--pokaz-kontekst", action="store_true")
    s.add_argument("--slad", help="zapisz pełny ślad diagnostyczny (JSON: analiza, ranking, prompty, surowe odpowiedzi, werdykty)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="lexrag", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("wgraj", help="ustawa → JSONL → indeks (jedno polecenie)");  s.set_defaults(fn=cmd_wgraj)
    s.add_argument("plik", help="plik .pdf/.txt/.html/.docx z ISAP")
    s.add_argument("--kod", required=True, help="krótki prefiks id, np. fus")
    s.add_argument("--akt", help="pełna nazwa aktu"); s.add_argument("--out", help="plik JSONL jednostek")
    s.add_argument("--index", default="data/index", help="katalog indeksu")
    s.add_argument("--dolacz", nargs="*", help="dodatkowe pliki JSONL do tego samego indeksu")
    s.add_argument("--bez-emb", action="store_true", help="tylko BM25 (bez embeddingów)")
    s.add_argument("--obowiazuje-od"); s.add_argument("--obowiazuje-do")
    s.add_argument("--wersja", default="", help="etykieta wersji aktu, np. 'tj. Dz.U. 2024 poz. 1631'")

    s = sub.add_parser("chat", help="rozmowa z indeksem");  s.set_defaults(fn=cmd_chat)
    _opcje_pytania(s)

    s = sub.add_parser("ask", help="jedno pytanie");  s.set_defaults(fn=cmd_ask)
    s.add_argument("pytanie"); _opcje_pytania(s)
    s.add_argument("--json", action="store_true", help="wynik jako JSON (dla integracji)")

    s = sub.add_parser("llm-test", help="sprawdź połączenie z LLM");  s.set_defaults(fn=cmd_llm_test)
    s.add_argument("--model")

    s = sub.add_parser("eval", help="recall@k na złotym zbiorze");  s.set_defaults(fn=cmd_eval)
    s.add_argument("plik"); s.add_argument("--index", default="data/index")
    s.add_argument("--bez-emb", action="store_true"); s.add_argument("--bez-reranker", action="store_true")
    s.add_argument("--warianty-llm", action="store_true", help="dodaj warianty zapytań z LLM")
    s.add_argument("--na-dzien"); s.add_argument("--cicho", action="store_true")

    s = sub.add_parser("eval-odpowiedzi", help="pokrycie i precyzja odpowiedzi (z LLM)");  s.set_defaults(fn=cmd_eval_odp)
    s.add_argument("plik"); s.add_argument("--index", default="data/index")
    s.add_argument("--bez-emb", action="store_true"); s.add_argument("--bez-reranker", action="store_true")
    s.add_argument("--bez-weryfikacji", action="store_true"); s.add_argument("--bez-analizy", action="store_true")
    s.add_argument("--dopytuj", action="store_true", help="pozwól na tryb dopytania (domyślnie wyłączony w ewaluacji)")
    s.add_argument("--limit", type=int); s.add_argument("--na-dzien")
    s.add_argument("--log", help="zapisz wszystkie odpowiedzi do JSONL")

    s = sub.add_parser("fetch", help="pobierz akt z API Sejmu");  s.set_defaults(fn=cmd_fetch)
    s.add_argument("ident", help="np. DU/2024/1631 (rok/pozycja Dz.U.)")
    s.add_argument("--out", default="data/raw")

    s = sub.add_parser("parse", help="ustawa → JSONL");  s.set_defaults(fn=cmd_parse)
    s.add_argument("plik"); s.add_argument("--kod", required=True, help="krótki prefiks id, np. fus")
    s.add_argument("--akt", help="pełna nazwa aktu"); s.add_argument("--out")
    s.add_argument("--obowiazuje-od"); s.add_argument("--obowiazuje-do")
    s.add_argument("--wersja", default="", help="etykieta wersji aktu, np. 'tj. Dz.U. 2024 poz. 1631'")

    s = sub.add_parser("parse-doc", help="procedura (.md/.docx/.pdf) → JSONL");  s.set_defaults(fn=cmd_parse_doc)
    s.add_argument("plik"); s.add_argument("--kod", required=True); s.add_argument("--dokument")
    s.add_argument("--wersja"); s.add_argument("--max-znakow", type=int, default=1800); s.add_argument("--out")
    s.add_argument("--obowiazuje-od"); s.add_argument("--obowiazuje-do")

    s = sub.add_parser("contextualize", help="opisy kontekstowe przez LLM");  s.set_defaults(fn=cmd_contextualize)
    s.add_argument("plik"); s.add_argument("--limit", type=int); s.add_argument("--nadpisz", action="store_true")

    s = sub.add_parser("index", help="zbuduj indeks z JSONL");  s.set_defaults(fn=cmd_index)
    s.add_argument("pliki", nargs="+"); s.add_argument("--out", default="data/index")
    s.add_argument("--bez-emb", action="store_true", help="tylko BM25")

    s = sub.add_parser("show", help="pokaż jednostkę");  s.set_defaults(fn=cmd_show)
    s.add_argument("id"); s.add_argument("--index", default="data/index")

    a = p.parse_args(argv)
    _log(a.verbose)
    a.fn(a)


if __name__ == "__main__":
    main()
