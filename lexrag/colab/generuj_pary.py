"""[Colab / A100]  Syntetyczne pary (pytanie potoczne ↔ jednostka) do dostrojenia embeddingów.

Wejście:  JSONL jednostek z `python -m lexrag parse` (TYLKO akty publiczne – ustawy,
          rozporządzenia; procedur wewnętrznych tu nie wolno wysyłać).
Wyjście:  pary.jsonl  {"zapytanie": "...", "id": "fus:art24:ust1", "tekst": "..."}

LLM: dowolny serwer zgodny z API OpenAI. Na Colabie najprościej vLLM:
    pip install vllm
    python -m vllm.entrypoints.openai.api_server --model CYFRAGOVPL/Llama-PLLuM-8B-instruct --dtype bfloat16 &
    export LEXRAG_LLM_BASE_URL=http://localhost:8000/v1  LEXRAG_LLM_MODEL=CYFRAGOVPL/Llama-PLLuM-8B-instruct
(nazwę modelu sprawdź na Hugging Face – rodzina PLLuM ma kilkanaście wariantów; im większy
model tu, tym lepsze pary; 12B w bf16 i 8x7B w 4-bit mieszczą się na A100 40 GB).

Uruchomienie:  python colab/generuj_pary.py data/units/fus.jsonl pary.jsonl --na-jednostke 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lexrag.llm import BrakLLM, czat_json  # noqa: E402
from lexrag.models import wczytaj_jsonl  # noqa: E402

STYLE = [
    "pracownik działu świadczeń, krótko, żargonem urzędowym",
    "osoba starsza, potocznie, bez znajomości przepisów",
    "pełnomocnik, precyzyjnie, z użyciem terminów ustawowych",
    "pytanie z literówką lub skrótem myślowym",
    "pytanie 'czy mogę / czy muszę' opisujące konkretną sytuację życiową",
]

PROMPT = """Fragment aktu prawnego:
[{sciezka}]
{tekst}

Wygeneruj {n} RÓŻNYCH pytań, na które ten fragment jest właściwą odpowiedzią.
Każde pytanie w innym stylu, po jednym z listy: {style}.
Pytania mają być takie, jakie naprawdę ktoś zadaje wyszukiwarce: bez numeru artykułu,
bez cytowania fragmentu, różnym słownictwem niż w przepisie.

Zwróć JSON: {{"pytania": ["...", "..."]}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jednostki")
    ap.add_argument("wyjscie")
    ap.add_argument("--na-jednostke", type=int, default=5)
    ap.add_argument("--limit", type=int, help="ile jednostek (do prób)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)

    jednostki = [j for j in wczytaj_jsonl(a.jednostki) if j.szukalny and len(j.tekst) > 40]
    if a.limit:
        jednostki = jednostki[: a.limit]
    zrobione = set()
    if os.path.exists(a.wyjscie):
        with open(a.wyjscie, encoding="utf-8") as f:
            zrobione = {json.loads(l)["id"] for l in f if l.strip()}
    print(f"jednostek: {len(jednostki)}, już zrobionych: {len(zrobione)}")

    with open(a.wyjscie, "a", encoding="utf-8") as out:
        for i, j in enumerate(jednostki, 1):
            if j.id in zrobione:
                continue
            style = random.sample(STYLE, k=min(a.na_jednostke, len(STYLE)))
            try:
                w = czat_json([{"role": "user", "content": PROMPT.format(
                    sciezka=j.sciezka, tekst=j.tekst[:2500], n=a.na_jednostke, style="; ".join(style))}],
                    max_tokens=600, temperatura=0.8)
            except (BrakLLM, ValueError) as e:
                print(f"pominięto {j.id}: {e}", file=sys.stderr)
                if isinstance(e, BrakLLM):
                    break
                continue
            for q in w.get("pytania", []):
                if isinstance(q, str) and 8 <= len(q) <= 300:
                    out.write(json.dumps({"zapytanie": q.strip(), "id": j.id, "tekst": j.tekst_do_indeksu()},
                                         ensure_ascii=False) + "\n")
            if i % 20 == 0:
                out.flush()
                print(f"{i}/{len(jednostki)}")
    print("gotowe:", a.wyjscie)


if __name__ == "__main__":
    main()
