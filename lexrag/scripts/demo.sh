#!/usr/bin/env bash
# Demo na ustawie testowej: parse -> index -> eval -> ask. Bez LLM i bez modeli (sam BM25).
set -e
cd "$(dirname "$0")/.."
python -m lexrag parse data/samples/ustawa_testowa.txt --kod test --akt "Ustawa testowa o świadczeniach testowych"
python -m lexrag parse-doc data/samples/procedura_testowa.md --kod pnps --dokument "Procedura P-NPS"
python -m lexrag index data/units/test.jsonl data/units/pnps.jsonl --bez-emb
python -m lexrag eval data/golden/ustawa_testowa.jsonl --bez-emb --bez-reranker
python -m lexrag ask "za jaki okres można żądać zwrotu nienależnie pobranych świadczeń" --bez-emb --bez-reranker --top 5
