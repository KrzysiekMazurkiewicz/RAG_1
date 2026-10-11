"""[Colab / A100]  Dostrojenie modelu embeddingów na parach (pytanie potoczne ↔ jednostka).

To łata rozjazd słownictwa: "kiedy mogę przejść na emeryturę" ↔ "przysługuje emerytura po
osiągnięciu wieku emerytalnego". Strata MultipleNegativesRanking: w batchu każda para jest
pozytywna dla siebie, a pozostałe jednostki są negatywami – dlatego batch powinien być
duży (A100: 64–128) i bez duplikatów id.

    pip install -U sentence-transformers datasets accelerate
    python colab/trenuj_embeddingi.py pary.jsonl --baza sdadas/mmlw-retrieval-roberta-large \
        --wyjscie modele/mmlw-fus --epoki 2 --golden data/golden/fus.jsonl --jednostki data/units/fus.jsonl

Wynik (katalog modelu) kopiujesz na serwer on-prem i ustawiasz LEXRAG_EMB_MODEL=/opt/modele/mmlw-fus.
Prefiks zapytania musi być TEN SAM przy treningu i w produkcji (LEXRAG_EMB_QUERY_PREFIX).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def wczytaj_pary(sciezka: str) -> list[dict]:
    with open(sciezka, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pary")
    ap.add_argument("--baza", default="sdadas/mmlw-retrieval-roberta-large")
    ap.add_argument("--prefiks-zapytania", default="zapytanie: ")
    ap.add_argument("--wyjscie", default="modele/emb-dostrojony")
    ap.add_argument("--epoki", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--walidacja", type=float, default=0.1, help="udział id odłożonych do walidacji")
    ap.add_argument("--golden", help="złoty zbiór JSONL (pytanie, oczekiwane) do oceny po treningu")
    ap.add_argument("--jednostki", help="JSONL jednostek – korpus do oceny na złotym zbiorze")
    a = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers

    pary = wczytaj_pary(a.pary)
    random.seed(7)
    ids = sorted({p["id"] for p in pary})
    random.shuffle(ids)
    wal = set(ids[: int(len(ids) * a.walidacja)])   # podział po jednostce, nie po pytaniu – bez przecieku
    tren = [p for p in pary if p["id"] not in wal]
    walid = [p for p in pary if p["id"] in wal]
    print(f"par: {len(pary)}  trening: {len(tren)}  walidacja: {len(walid)}  jednostek: {len(ids)}")

    model = SentenceTransformer(a.baza)
    ds = Dataset.from_dict({
        "anchor": [a.prefiks_zapytania + p["zapytanie"] for p in tren],
        "positive": [p["tekst"] for p in tren],
    })

    # ewaluator IR na odłożonych jednostkach: zapytania → właściwa jednostka
    korpus = {p["id"]: p["tekst"] for p in pary}
    zapytania = {f"q{i}": a.prefiks_zapytania + p["zapytanie"] for i, p in enumerate(walid)}
    relewantne = {f"q{i}": {p["id"]} for i, p in enumerate(walid)}
    ewal = InformationRetrievalEvaluator(zapytania, korpus, relewantne, name="walidacja",
                                         show_progress_bar=False) if walid else None

    args = SentenceTransformerTrainingArguments(
        output_dir=a.wyjscie + "-ckpt", num_train_epochs=a.epoki,
        per_device_train_batch_size=a.batch, learning_rate=a.lr, warmup_ratio=0.1,
        fp16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch" if ewal else "no", save_strategy="no", logging_steps=20,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                                         loss=losses.MultipleNegativesRankingLoss(model), evaluator=ewal)
    if ewal:
        print("przed treningiem:", ewal(model))
    trainer.train()
    if ewal:
        print("po treningu:", ewal(model))
    model.save(a.wyjscie)
    print("zapisano", a.wyjscie)

    if a.golden and a.jednostki:
        from lexrag.models import wczytaj_jsonl
        jedn = {j.id: j for j in wczytaj_jsonl(a.jednostki) if j.szukalny}
        korpus2 = {i: j.tekst_do_indeksu() for i, j in jedn.items()}
        zl = [json.loads(l) for l in open(a.golden, encoding="utf-8") if l.strip() and not l.startswith("#")]
        q2 = {f"g{i}": a.prefiks_zapytania + z["pytanie"] for i, z in enumerate(zl)}
        r2 = {f"g{i}": set(z["oczekiwane"]) for i, z in enumerate(zl)}
        print("złoty zbiór:", InformationRetrievalEvaluator(q2, korpus2, r2, name="golden")(model))


if __name__ == "__main__":
    main()
