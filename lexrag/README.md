# lexrag — pytania do ustawy z cytowaniem, model on-premise

Wkładasz ustawę (PDF z ISAP), pytasz po ludzku, dostajesz odpowiedź opartą na konkretnym
ustępie — z treścią przepisu wyświetloną **z indeksu, nie z modelu**. Wszystko działa lokalnie:
parser, wyszukiwanie (BM25 + embeddingi + reranker) i model językowy przez Ollamę / llama.cpp / vLLM.
Żadne dane nie wychodzą poza Twój serwer.

Co to znaczy „99 % poprawnych odpowiedzi” w tym systemie: **precyzja wśród odpowiedzi udzielonych**.
Model nie ma jak zmyślić przepisu (cytaty renderuje system po id), każda liczba w odpowiedzi musi
występować w cytowanych przepisach (kontrola deterministyczna), każde zdanie jest sprawdzane
osobnym przebiegiem modelu, a gdy pokrycia nie ma — system odmawia i oddaje sprawę referentowi.
Odmowa jest tańsza niż błąd. Pokrycie (ile pytań dostaje odpowiedź) mierzysz osobno i podnosisz
lepszym retrievalem i lepszym modelem. Liczbę mierzy polecenie `eval-odpowiedzi` na Twoim złotym
zbiorze — bez pomiaru to tylko deklaracja.

## Szybki start

```bash
cd lexrag
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-ml.txt pymupdf
source .venv/bin/activate

# 1. model on-premise (Ollama >= 0.40). Karta >= 12 GB VRAM: Gemma 4 12B – 100 % precyzji na obu zmierzonych ustawach (patrz „Zmierzone”)
ollama pull gemma4:12b
export LEXRAG_LLM_MODEL=gemma4:12b
# karta 8 GB: qwen3:8b (97,5 % na FUS, ale 82 % na ustawie zasiłkowej); laptop 16 GB bez karty albo karta 6 GB: gemma4:e4b
# karta >= 12 GB VRAM i priorytet na precyzję: Saluki 27B (2-bit, 7,9 GB) – 100 % precyzji, ale 3x wolniej:
#   curl -L -o saluki.gguf https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0/resolve/main/Underdog-Saluki-27B-1.0-IQ2-mix.gguf
#   printf 'FROM ./saluki.gguf\n' > Modelfile && ollama create saluki -f Modelfile && export LEXRAG_LLM_MODEL=saluki
python -m lexrag llm-test                                      # tryb API, czas, czy JSON działa

# 2. ustawa: tekst jednolity z ISAP (link „Tekst jednolity”, plik PDF) → jednostki → indeks
python -m lexrag wgraj ustawa.pdf --kod fus --akt "Ustawa o emeryturach i rentach z FUS" --obowiazuje-od 2024-11-07
python -m lexrag show fus:art1          # sprawdź wyrywkowo, czy ustępy wyglądają jak w ustawie

# 3. pytaj
python -m lexrag chat
python -m lexrag ask "za jaki okres można żądać zwrotu nienależnie pobranych świadczeń" --pokaz-kontekst
```

Zamiast `wgraj` z pliku można pobrać akt z API Sejmu: `python -m lexrag fetch DU/2024/1631`
(rok/pozycja Dz.U. **obwieszczenia z tekstem jednolitym**), a potem `wgraj data/raw/DU_2024_1631.txt ...`.

Bez ustawionego `LEXRAG_LLM_MODEL` wszystko działa w trybie retrieval-only: dostajesz listę
trafionych ustępów zamiast odpowiedzi. To dobry moment na pierwszą kontrolę jakości parsera i wyszukiwania.

## Jak to działa

```
ustawa (.pdf/.txt/.html/.docx) ─► wgraj: parse ─► jednostki JSONL (ustęp = jednostka) ─► indeks BM25 + embeddingi
                                                                                                  │
pytanie ─► [1] analiza + warianty (LLM) ─► [2] BM25 + wektory (RRF) ─► [3] reranker ─► [4] rozszerzenie: cały artykuł,
           dopytanie, gdy brak danych        retrieve.szukaj            retrieve.rerank   odesłania (1 skok), definicje
                                                                                                  │
   odpowiedź + cytaty ◄── [6b] weryfikacja zdań (LLM) ◄── [6a] kontrola liczb ◄── [5] odpowiedź + id cytatów (LLM) ◄─┘
```

| krok | co robi | moduł |
|---|---|---|
| ingest | PDF Dz.U.: tekst zasadniczy po dominującym rozmiarze czcionki, **przypisy (mniejsza czcionka) i znaczniki przypisów (indeks górny) odpadają**, spany sklejane w wiersze | `fetch.pdf_do_tekstu` |
| ingest | hierarchia dział/rozdział/art./ust./pkt; ustęp = jednostka; `Art. 145–172. (pominięte)`, dwa brzmienia tego samego ustępu (`ust1`, `ust1-b`), `(uchylony)`/`(utracił moc)`, załączniki; odesłania „art. 46, 47 i 184” do grafu; definicje z „określenia oznaczają” osobno | `parse_ustawa.py` |
| 2 | BM25 na lematach (simplemma) + wektory (`sdadas/mmlw-retrieval-roberta-large`), scalone RRF, filtr na datę obowiązywania | `bm25.py`, `tokenize.py`, `retrieve.py` |
| 3 | cross-encoder `BAAI/bge-reranker-v2-m3` przestawia top-50 na top-8; opcjonalny próg odmowy | `retrieve.rerank` |
| 4 | do promptu: trafienia → cały artykuł-rodzic → odesłania → definicje; twardy limit znaków (`LEXRAG_MAX_CTX_CHARS`) | `retrieve.rozszerz` |
| 1, 5, 6b | trzy wywołania LLM ze **schematem JSON** (dekodowanie ograniczone gramatyką — Ollama `format`, llama.cpp/vLLM `response_format`), tryb myślenia wyłączony | `generate.py`, `llm.py` |
| 6a | każda liczba w odpowiedzi (lata, kwoty, procenty, terminy, numery artykułów) musi wystąpić w kontekście — także słownie („trzech lat”); zdania z liczbami znikąd wypadają bez pytania modelu | `generate.liczby_bez_pokrycia` |
| eval | recall@k wyszukiwania; pokrycie i precyzja odpowiedzi (cytat trafny + frazy kluczowe) | `evaluate.py` |

## Model on-premise — co wybrać

Pipeline jest niezależny od modelu: zmiana to `LEXRAG_LLM_MODEL` i ewentualnie `LEXRAG_LLM_BASE_URL`.

| opcja | RAM | uwagi |
|---|---|---|
| **Gemma 4 E4B** (`gemma4:e4b`, 8,1 mld parametrów) | ~6 GB | **rekomendacja na laptop 16 GB i na kartę 6 GB**: na złotym zbiorze FUS 97,4 % precyzji przy 95 % pokryciu, ~90 s/pytanie na M2 bez GPU, 13 s na T4; gdy nie jest pewna – odmawia. Na Apple Silicon działa na runnerze MLX bez structured output, klient sam przechodzi na JSON z promptu. |
| Qwen3 8B (`qwen3:8b`, Q4_K_M 5,2 GB) | ~7 GB VRAM | na FUS 97,5 % precyzji przy 100 % pokrycia, ale na ustawie zasiłkowej **82 %** (cytuje ustępy szczególne zamiast reguły, odpowiedzi niepełne) – kompromis na kartę 8 GB, nie rekomendacja; 10 s/pytanie na L4, 21 s na T4 |
| Qwen3 4B (`qwen3:4b`) | 3–4 GB | 92,5 % precyzji przy 100 % pokryciu, 13 s na T4, ~140 s na M2 – odpowiada zawsze, także gdy nie powinien (pytania tak/nie, reguła vs wyjątek) |
| **Saluki 27B** (`hf.co/ConwayResearch/Underdog-Saluki-27B-1.0`) | ~9–11 GB | Qwen3.5-27B skompresowany do 2 bitów (7,9 GB). Jedyny 27B, który mieści się w laptopie z 16 GB. Na T4: **100 % precyzji** przy 95 % pokrycia (2 odmowy, 0 błędów), 62 s/pytanie – najdokładniejszy, ale 3x wolniejszy od Qwen3 8B i potrzebuje ~10–12 GB VRAM. Cena 2 bitów: słabsze rozumowanie i matematyka niż pełny model. Wymaga Ollamy ≥ 0.40 (architektura `qwen35`); `ollama pull hf.co/...` nie działa – trzeba pobrać GGUF i `ollama create` (patrz Szybki start). |
| Qwen3.5-27B / Qwen3.x-32B w Q4_K_M lub Q8_0 | 18–30 GB | ten sam model bez utraty precyzji; na serwerze z 32–64 GB to lepszy wybór niż Saluki |
| **Gemma 4 12B** (`gemma4:12b` ~7,6 GB; także `gemma4:26b-a4b` MoE ~18 GB, `gemma4:31b` ~20 GB) | ~10 GB VRAM | **rekomendacja produkcyjna**: jedyny model bez ani jednego błędu na obu ustawach (100 % precyzji przy 100 % pokrycia na FUS i na ustawie zasiłkowej, L4), 12–13 s/pytanie na L4; Apache 2.0, bardzo dobry polski, 256k kontekstu; na M2 bez GPU ~240 s/pytanie |
| Gemma 3 27B | 18 GB (Q4) | poprzednia generacja, nadal mocna w polskim (PLCC lepsza od Bielika i PLLuM) |
| Bielik 11B v3, PLLuM 12B / 8x7B / 70B | 7–45 GB | polskie modele; PLLuM powstał dla administracji publicznej; w rankingach ogólnych niżej niż Gemma/Qwen — sprawdź na swoich pytaniach |

Praktyka, która decyduje o jakości bardziej niż wybór modelu:

* **Okno kontekstu**: Ollama domyślnie ucina prompt do kilku tysięcy tokenów **bez ostrzeżenia** —
  dlatego klient wysyła `num_ctx` (`LEXRAG_LLM_NUM_CTX`, domyślnie 12288) i ostrzega, gdy prompt zbliża się do limitu.
* **Tryb myślenia** (Qwen3/Saluki/DeepSeek): wyłączony (`LEXRAG_LLM_THINK=0`), inaczej model zużywa
  limit tokenów na rozważania, a JSON przychodzi ucięty.
* **Schemat JSON**: Ollama `format`, llama.cpp/vLLM `response_format: json_schema` → model nie może
  oddać niepoprawnej struktury; przy odrzuceniu pola klient degraduje do `json_object` i do zwykłego promptu.
* **Czas na CPU/laptopie**: wąskim gardłem jest czytanie promptu (prefill). Zmierzono na MacBooku Air M2 16 GB
  z Saluki: prefill ~16 tok/s, generowanie ~2 tok/s — pełny przebieg (3 wywołania) trwał 12 min.
  Na wolnym sprzęcie użyj `--bez-analizy` (jedno wywołanie mniej), `LEXRAG_MAX_CTX_CHARS=7000`,
  `LEXRAG_TOP_N=6`; na serwerze z kartą te same kroki trwają sekundy.

Inne serwery:

```bash
# llama.cpp (najszybciej dostaje nowe architektury): brew install llama.cpp
llama-server -m model.gguf --jinja -c 12288 --port 8080
export LEXRAG_LLM_BASE_URL=http://localhost:8080/v1 LEXRAG_LLM_MODEL=model
# vLLM na karcie: python -m vllm.entrypoints.openai.api_server --model ... → LEXRAG_LLM_BASE_URL=http://host:8000/v1
# tryb API wybiera się sam (ollama, gdy adres ma :11434), ręcznie: LEXRAG_LLM_API=ollama|openai
```

## Konfiguracja (zmienne środowiskowe)

| zmienna | domyślnie | znaczenie |
|---|---|---|
| `LEXRAG_LLM_BASE_URL` | `http://localhost:11434/v1` | adres serwera modelu |
| `LEXRAG_LLM_API` | `auto` | `ollama` (natywne /api/chat) albo `openai` (/v1/chat/completions) |
| `LEXRAG_DEVICE` | auto | urządzenie embeddingów i rerankera: auto (karta NVIDIA, gdy jest), cpu, cuda |
| `LEXRAG_LLM_MODEL` | puste | nazwa modelu; puste = retrieval-only |
| `LEXRAG_LLM_MODEL_SMALL` | puste | mniejszy model do kroków 1 i 6b |
| `LEXRAG_LLM_NUM_CTX` | `12288` | okno kontekstu wysyłane do Ollamy |
| `LEXRAG_LLM_THINK` | `0` | `0` wyłącza tryb myślenia, `1` włącza, puste – nie wysyłaj |
| `LEXRAG_LLM_TIMEOUT` | `900` | sekundy na jedno wywołanie |
| `LEXRAG_LLM_EXTRA_JSON` | puste | dodatkowe pola body, np. `{"chat_template_kwargs": {"enable_thinking": false}}` dla llama.cpp |
| `LEXRAG_SKIP_ANALYSIS` | `0` | `1` = pomiń krok analizy pytania (to samo co `--bez-analizy`) |
| `LEXRAG_NUMERIC_CHECK` | `1` | kontrola liczb w odpowiedzi |
| `LEXRAG_RERANK_THRESHOLD` | `0` | np. `0.05`: odmowa, gdy najlepszy kandydat ma niską trafność wg rerankera |
| `LEXRAG_MAX_CTX_CHARS` | `12000` | budżet znaków kontekstu dla LLM (~4 tys. tokenów) |
| `LEXRAG_TOP_N` / `LEXRAG_K_EACH` | `8` / `30` | ile jednostek po reranku / ile kandydatów z każdej metody |
| `LEXRAG_EMB_MODEL`, `LEXRAG_RERANKER_MODEL` | mmlw-roberta-large, bge-reranker-v2-m3 | ścieżka lokalna też działa; `LEXRAG_DISABLE_EMB=1`, `LEXRAG_DISABLE_RERANKER=1` wyłączają |

## Jakość: złoty zbiór i bramki

`data/golden/*.jsonl` — jedna linia = jedno pytanie:

```json
{"pytanie": "za jaki okres wstecz można żądać zwrotu nienależnie pobranych świadczeń",
 "oczekiwane": ["fus:art138:ust4"], "musi_zawierac": ["3 lata|trzech lat", "12 miesięcy"]}
```

```bash
python -m lexrag eval data/golden/fus.jsonl --index data/index_fus                      # recall@k, pełna hybryda
python -m lexrag eval data/golden/fus.jsonl --index data/index_fus --bez-reranker       # ablacje
python -m lexrag eval data/golden/fus.jsonl --index data/index_fus --bez-emb --bez-reranker
python -m lexrag eval-odpowiedzi data/golden/fus.jsonl --index data/index_fus --log odpowiedzi.jsonl
```

Kolejność budowy i bramki (nie przeskakuj):

1. **Parser**: `show` kilku ustępów, liczba artykułów bez luk w numeracji, zero przypisów w treści.
2. **Wyszukiwanie bez LLM**: `recall@10 wszystkie ≥ 95 %`. Jeśli właściwy ustęp nie wchodzi do dziesiątki,
   żaden model tego nie uratuje.
3. **Odpowiedzi**: `eval-odpowiedzi` — `POPRAWNE ≥ 99 %` wśród udzielonych; pokrycie raportowane osobno.
   Każda zła odpowiedź to nowa linia złotego zbioru i poprawka promptu albo retrievalu, nie wiara w model.
4. Gdy recall kuleje na potocznych pytaniach: `contextualize` (opisy jednostek przez LLM) i `colab/` (dostrojenie embeddingów).

Wyniki na ustawie o emeryturach i rentach z FUS (Dz.U. 2024 poz. 1631, 216 artykułów, 589 ustępów,
złoty zbiór 40 pytań, `data/golden/fus.jsonl`): patrz sekcja „Zmierzone” niżej.

**Drugi akt i wspólny indeks.** Żeby sprawdzić, czy system nie gubi się przy wielu ustawach, jest drugi złoty
zbiór: `data/golden/zas.jsonl` – 40 pytań do ustawy o świadczeniach pieniężnych z ubezpieczenia społecznego
w razie choroby i macierzyństwa (tekst jednolity Dz.U. 2026 poz. 854, `python -m lexrag fetch DU/2026/854`,
kod `zas`: 100 artykułów, 322 ustępy). Oba akty w jednym indeksie:

```bash
python -m lexrag wgraj data/raw/DU_2026_854.txt --kod zas --akt "Ustawa o świadczeniach pieniężnych z ubezpieczenia społecznego w razie choroby i macierzyństwa" --obowiazuje-od 2026-06-19 --out data/units/zas.jsonl --bez-emb
python -m lexrag index data/units/fus.jsonl data/units/zas.jsonl --out data/index_wspolny
python -m lexrag eval data/golden/zas.jsonl --index data/index_wspolny
python -m lexrag eval data/golden/fus.jsonl --index data/index_wspolny    # czy FUS nie straciło po dołożeniu drugiego aktu
```

Każde pytanie w zbiorze ma frazy `musi_zawierac` sprawdzone skryptem wprost w treści oczekiwanego ustępu, więc
błąd w zbiorze nie udaje błędu modelu.

**Pytania hasłowe i odpowiedź przeglądowa.** Ludzie piszą „wcześniejsza emerytura”, nie „czy kobieta urodzona
w 1948 r. …”. Pytanie hasłowe (do 6 słów, bez zaimka pytajnego i znaku zapytania) albo pytanie, w którym analiza
wykryła brak danych o sytuacji, dostaje **odpowiedź przeglądową**: 3–8 zdań ze wszystkimi sytuacjami, warunkami
i wysokościami z odnalezionych przepisów, od reguły ogólnej po wyjątki, każda z cytatem, a na końcu dopytanie
o dane, które pozwolą odpowiedzieć konkretnie (pola `przeglad` i `dopytanie` w wyniku). Dopytanie bez odpowiedzi
zostaje tylko wtedy, gdy odpowiedź przeglądowa nie ma pokrycia w przepisach. Mierzą to osobne zbiory
`data/golden/fus_haslowe.jsonl` i `data/golden/zas_haslowe.jsonl` (po 12 pytań); wpis `oczekiwane` w formie
`a|b` oznacza alternatywę (wystarczy zacytować jeden z przepisów).

Zmierzone na wspólnym indeksie (1254 jednostki, 990 szukalnych; BM25 + embeddingi + reranker, laptop M2):

| złoty zbiór | recall@1 | recall@3 | recall@5 | recall@10 | MRR | komplet w kontekście LLM |
|---|---|---|---|---|---|---|
| zasiłkowa (40 pytań), wspólny indeks | 75,0 % | **100 %** | 100 % | 100 % | 0,867 | 100 % |
| FUS (40 pytań), wspólny indeks | 77,5 % | 87,5 % | 97,5 % | **100 %** | 0,848 | 100 % |
| FUS (40 pytań), indeks samej ustawy o FUS | 77,5 % | 90,0 % | 97,5 % | 100 % | 0,854 | 100 % |

Dołożenie drugiej ustawy prawie nie kosztuje: dla FUS jedno pytanie spadło z top-3 do top-5, recall@10 i komplet
w kontekście zostały na 100 %. Prefiks kodu aktu w id (`fus:`, `zas:`) i nazwa aktu w nagłówku jednostki
pozwalają modelowi odróżnić, z której ustawy cytuje.

## Wgrywanie własnej ustawy — na co uważać

* Bierz **tekst jednolity** (obwieszczenie Marszałka Sejmu) w **PDF** z ISAP: parser czyta rozmiary
  czcionek i wycina przypisy. Z `.txt` / `.html` przypisy wycina tylko regexami (gorzej).
* Podaj `--obowiazuje-od` (data ogłoszenia tekstu jednolitego); kolejne wersje ustawy wgrywaj z innym
  `--kod` i datami, a `ask --na-dzien 2019-03-01` ograniczy odpowiedzi do stanu prawnego z tego dnia.
* Po `wgraj` zawsze: `show kod:art1`, `show` ustępu z liczbami, które znasz, i kilka pytań w `chat --bez-llm`.
* Tabele w załącznikach trafiają do indeksu jako płaski tekst — pytania „z tabeli” wymagają innego podejścia.
* Rozporządzenia (`§ 1.`) parsują się tak samo jak ustawy.

## Procedury wewnętrzne

```bash
pandoc procedura.docx -t gfm --wrap=none -o procedura.md
python -m lexrag parse-doc procedura.md --kod pnps --dokument "Procedura NPS" --wersja 3.2
python -m lexrag index data/units/fus.jsonl data/units/pnps.jsonl
```

Nagłówki stają się ścieżką jednostki, długie sekcje są dzielone z zachowaniem tabel. Nic nie opuszcza sieci.

## Testy

```bash
python -m pytest -q      # 34 testy: parser (w tym przypadki z prawdziwego PDF Dz.U.), kontrola liczb, orkiestracja LLM (zamockowana)
```

## Interfejs WWW i API

```bash
pip install -r requirements-web.txt
uvicorn lexrag.web:app --host 0.0.0.0 --port 8090      # http://localhost:8090
```

Jedna strona: pytanie (z datą „na dzień” i opcjonalną analizą), odpowiedź z podstawą prawną renderowaną
z indeksu, zdania usunięte przez bramki, czasy kroków; formularz wgrywania aktu (PDF/TXT/HTML/DOCX) lub
dokumentu wewnętrznego (nagłówki Markdown/DOCX) z kodem, nazwą i okresem obowiązywania. Wgranie buduje
w tle jeden wspólny indeks ze wszystkich plików `data/units/*.jsonl`. API: `GET /api/stan`,
`POST /api/pytanie {"pytanie", "na_dzien", "analiza"}`, `POST /api/wgraj` (multipart), `GET /api/jednostka/{id}`.
Odpowiedź niesie też `przeglad` (true dla pytań hasłowych) i `dopytanie` (pytanie doprecyzowujące).
Pytanie bez `na_dzien` liczy się na dziś: gdy w indeksie są dwie wersje aktu (różne kody, okresy obowiązywania
`--obowiazuje-od/--obowiazuje-do`), odpowiada bieżąca, a z datą ta, która obowiązywała w tym dniu.

## NAS z kartą NVIDIA (Docker)

Sprawdzone założenia: QNAP TS-x73A, Container Station z Dockerem ≥ 27 i runtime `nvidia`, RTX 3050 6 GB.

```bash
scp -r lexrag krzymaz@nas:/share/CACHEDEV1_DATA/lexrag      # bez .venv i data/index*
ssh krzymaz@nas
cd /share/CACHEDEV1_DATA/lexrag && cp .env.example.docker .env
docker compose up -d --build                                 # obraz z torch+CUDA ≈ 6 GB, pierwsza budowa 10–20 min
docker exec lexrag-ollama ollama pull qwen3:4b               # model z .env
```

Dwa kontenery: `lexrag-ollama` (model na GPU, port 11434) i `lexrag-app` (WWW i API na porcie 8090; reranker
i embeddingi też na GPU, `LEXRAG_DEVICE=auto`). Dane trwałe w `./dane`: indeksy i wgrane pliki, modele Ollamy,
cache modeli embeddingów. Na 6 GB VRAM mieści się w całości Qwen3 4B; Gemma 4 E4B częściowo przechodzi na CPU.
Bez karty: usuń `runtime: nvidia` z obu usług, ustaw `LEXRAG_DEVICE=cpu` i zbuduj obraz z
`--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu`.

## Pomiar na GPU w Google Colab (tylko akty publiczne)

`colab/lexrag_pomiar_colab.ipynb` uruchamia Ollamę na karcie Colaba, buduje indeks ustawy o FUS pobranej z API
Sejmu, mierzy sześć modeli (Gemma 4 E4B, Qwen3 4B/8B/14B, Gemma 4 12B, Saluki 27B) na złotym zbiorze, dokłada
drugą ustawę do wspólnego indeksu i powtarza pomiar na jej pytaniach – razem około 2,5 godziny na L4. Do notatnika
wgrywa się `lexrag-colab.zip` (repozytorium bez `.venv`, indeksów i surowych plików; zip budowany z katalogu
repozytorium: `zip -r ../lexrag-colab.zip . -x ".venv/*" "data/index*" "data/raw/*" "dane/*" "*/__pycache__/*"
"data/eval_odpowiedzi_*"`), **po połączeniu z maszyną** – plik wgrany wcześniej przepada.

Karta: **L4** (24 GB VRAM, 53 GB RAM, ~2 jednostki/h). T4 (16 GB, 12,7 GB RAM, ~1 jednostka/h) wystarcza dla modeli
do 8B, ale przy 12B/14B/27B jej maszyna wyczerpuje RAM: runner Ollamy ginie („model runner has unexpectedly
stopped”, pytanie liczy się jako błąd), a w końcu zawiesza się serwer sesji i notatnik traci łączność. To ogranicza
pamięć systemowa, nie karta – A100 nie jest potrzebna. **Instrukcje wewnętrzne i dokumenty
instytucji nie są wgrywane do Colaba** – te testuje się wyłącznie na własnym sprzęcie. Obok leżą skrypty
`colab/generuj_pary.py` i `colab/trenuj_embeddingi.py` do dostrojenia embeddingów na A100.

## Zmierzone (10–11 października 2026, MacBook Air M2 16 GB i Tesla T4 w Colabie)

**Parser** na tekście jednolitym ustawy o FUS (PDF z api.sejm.gov.pl, 72 strony): 216 artykułów (w tym
`Art. 145–172. (pominięte)`), 589 ustępów, 14 definicji, 30 jednostek uchylonych / „utracił moc”, 1 załącznik,
818 odesłań wewnętrznych; zero luk w numeracji, zero przypisów w treści ustępów, jedno podwójne brzmienie
(art. 95 ust. 1 → `fus:art95:ust1` i `fus:art95:ust1-b`).

**Wyszukiwanie** (`eval`, 40 potocznych pytań, `data/golden/fus.jsonl`):

| konfiguracja | recall@1 | recall@3 | recall@5 | recall@10 | MRR | komplet w kontekście LLM |
|---|---|---|---|---|---|---|
| BM25 (lematy) | 32,5 % | 57,5 % | 67,5 % | 72,5 % | 0,478 | 72,5 % |
| BM25 + embeddingi (mmlw-roberta-large) | 42,5 % | 72,5 % | 82,5 % | 97,5 % | 0,587 | 87,5 % |
| BM25 + embeddingi + reranker (bge-reranker-v2-m3) | 77,5 % | 90,0 % | 97,5 % | **100 %** | 0,854 | **100 %** |

Wniosek: sam BM25 nie wystarcza na potoczne pytania; embeddingi i reranker są obowiązkowe. Reranker na
CPU kosztuje ~20 s na pytanie (50 kandydatów) — na serwerze z kartą ułamek sekundy.

**Odpowiedzi** (`eval-odpowiedzi`, Ollama 0.40.2, `--bez-analizy`; wiersze „10 pytań”: `LEXRAG_TOP_N=6`,
`LEXRAG_MAX_CTX_CHARS=7000`, pierwsze 10 pytań złotego zbioru; wiersze „budżet domyślny”: `LEXRAG_TOP_N=8`,
`LEXRAG_MAX_CTX_CHARS=12000`; `scripts/porownaj_modele.py` zestawia logi):

| model | pytań | odpowiedzi | poprawne | precyzja | pokrycie | s / pytanie |
|---|---|---|---|---|---|---|
| Saluki 27B (Qwen3.5, 2-bit) | 7 (przerwane) | 6 | 6 | **100 %** | 86 % (1 odmowa) | 304 |
| Gemma 4 12B (nvfp4, runner MLX) | 1 (przerwane) | 1 | 1 | 100 % | 100 % | 240 |
| Gemma 4 E4B (runner MLX) | 10 | 10 | 8 | 80 % | 100 % | 89 |
| Qwen3 4B (Q4_K_M) | 10 | 10 | 8 | 80 % | 100 % | 79 |
| Qwen3 4B (Q4_K_M), budżet domyślny, przed poprawką weryfikacji | **40** | 34 | 29 | 85 % | 85 % (6 odmów) | 112 |
| Qwen3 8B (Q4_K_M), budżet domyślny | 10 | 10 | 9 | 90 % | 100 % | 201 |
| Qwen3 4B, budżet domyślny, **po poprawce weryfikacji** | **40** | 40 | 37 | 92,5 % | 100 % | 143 |
| **Gemma 4 E4B, budżet domyślny, po poprawce weryfikacji** | **40** | 38 | 37 | **97,4 %** | 95 % (2 odmowy) | 92 |

Odczyt: duży model raczej **odmawia**, gdy nie jest pewny (Saluki: 1 odmowa, 0 błędów); małe modele odpowiadają
zawsze i na 10 pytaniach oba pomyliły się na tych samych dwu – raz cytując przepisy o ponownym ustaleniu emerytury
zamiast art. 25 ust. 1 (treść odpowiedzi była poprawna; to pytanie myli też Qwen3 8B, a Saluki na nim odmówił),
raz odpowiadając o innym przepisie (art. 13 ust. 3a zamiast art. 58 ust. 2). Dlatego weryfikacja sprawdza także,
**czy odpowiedź dotyczy pytania** (`odpowiada_na_pytanie` → odmowa), a budżet kontekstu dla małych modeli zostaje
domyślny (dla 4B prefill jest tani).

Pełny przebieg Qwen3 4B na 40 pytaniach pokazał, że z 5 błędnych odpowiedzi 4 wynikały z **orkiestracji, nie z
modelu** – i to zostało naprawione w kodzie (testy `test_weryfikacja_dopytuje_o_brakujace_werdykty`,
`test_cytaty_odzyskane_z_tresci_gdy_model_nie_podal_id`, `test_odmowa_gdy_zostaje_tylko_powtorzenie_pytania`):

* weryfikator (ten sam 4B) zwracał werdykt **tylko dla pierwszego zdania**, a brak werdyktu był traktowany jako
  „usuń”: w 16 z 34 odpowiedzi zostało dokładnie jedno zdanie – czasem samo przepisane pytanie („Z ilu lat bierze
  się zarobki…?”), podczas gdy poprawne zdanie z art. 15 ust. 1 wylatywało. Teraz schemat weryfikacji wymusza
  `minItems = maxItems = liczba zdań` (sprawdzone na żywo: 4B zwraca wtedy komplet), brakujące werdykty są
  dopytywane zdanie po zdaniu, a zdania będące pytaniem lub przepisaniem pytania odpadają deterministycznie;
* w 6 odmowach model nie wypełnił `cytowane_id`, choć w treści podał „art. 59 ust. 1” – teraz adresy
  „art. X ust. Y” / „§ X” z treści są mapowane na jednostki **z kontekstu** (ustęp spoza kontekstu → tylko
  jednostka artykułowa), gdy model nie poda id.

Jedyny błąd czysto modelowy: na pytanie „czy trzeba rozwiązać umowę o pracę, żeby ZUS wypłacał emeryturę” 4B
odpowiedział „Nie” wbrew art. 103a, a jego własny weryfikator to przepuścił – pytania tak/nie to słaby punkt małych
modeli i argument, by odpowiedź zawsze czytać razem z wyświetlonym cytatem (albo weryfikować większym modelem:
`LEXRAG_LLM_MODEL_SMALL` wskazuje model do analizy i weryfikacji).

**Powtórka 40 pytań po poprawkach**: Qwen3 4B 92,5 % precyzji przy 100 % pokrycia (143 s/pytanie – dopytywanie
zdanie po zdaniu kosztuje), **Gemma 4 E4B 97,4 % przy 95 % pokrycia (92 s/pytanie)**. E4B pomyliła się tylko na
dwuznacznym pytaniu o podstawę obliczenia emerytury (cytuje przepisy o ponownym ustaleniu zamiast art. 25 ust. 1 –
jak wszystkie małe modele), a na dwu najtrudniejszych pytaniach **odmówiła**, zamiast się pomylić: raz sama
(„brak informacji, czy konieczne jest rozwiązanie umowy o pracę”), raz zatrzymała ją kontrola trafności
(odpowiedź o art. 13 ust. 3a zamiast art. 58 ust. 2). Qwen3 4B na tych samych pytaniach odpowiada błędnie,
a do tego dobiera wyjątek zamiast reguły (art. 116 ust. 1a zamiast ust. 1, art. 173 ust. 1a zamiast ust. 1).

Wniosek: **na laptopie 16 GB rekomendowany jest Gemma 4 E4B** – 97 % precyzji to najwięcej, co dają małe modele,
a jego odmowy trafiają właśnie tam, gdzie pytanie jest podchwytliwe. 99 % precyzji wymaga modelu klasy 27B
(Saluki: 100 % na 6 odpowiedziach i 1 odmowa) – na laptopie to 5 min/pytanie, na serwerze z kartą 24 GB VRAM
sekundy. Pozostała luka to głównie pytania tak/nie i pytania dwuznaczne (reguła vs wyjątek); tam pomaga
dopytanie (krok 1) i czytanie odpowiedzi razem z wyświetlonym cytatem.

**Pomiar na karcie (Google Colab, Tesla T4 16 GB, Ollama 0.40.3, 11 października 2026)** – ten sam kod, te same
40 pytań i ten sam budżet domyślny co w ostatnich wierszach powyżej; wszystkie modele w tej samej sesji,
`colab/lexrag_pomiar_colab.ipynb`:

| model | odpowiedzi | poprawne | precyzja | pokrycie | s / pytanie |
|---|---|---|---|---|---|
| **Saluki 27B** (Qwen3.5-27B, 2-bit IQ2, 7,9 GB) | 38 | 38 | **100 %** | 95 % (2 odmowy) | 62 |
| **Qwen3 8B** (Q4_K_M) | 40 | 39 | 97,5 % | **100 %** | 21 |
| Gemma 4 E4B | 38 | 37 | 97,4 % | 95 % (2 odmowy) | 13 |
| Gemma 4 12B | 34 | 33 | 97,1 % | 85 % (6 błędów runnera) | 33 |
| Qwen3 4B (Q4_K_M) | 40 | 37 | 92,5 % | 100 % | 13 |

Co z tego wynika:

* **Saluki 27B daje 100 % precyzji** – ani jednej błędnej odpowiedzi; na dwu najtrudniejszych pytaniach (podstawa
  obliczenia nowej emerytury, rozwiązanie umowy o pracę a wypłata emerytury) odmówił. To jedyny model, który
  spełnia cel „99 %”, ale kosztuje 62 s/pytanie na T4 i ~10–12 GB VRAM (nie mieści się na karcie 6 GB).
* **Qwen3 8B na FUS wygląda na dobry kompromis** (97,5 % przy 100 % pokrycia i 21 s/pytanie), ale na drugiej
  ustawie spadł do 82 % (tabela L4 niżej); jedyny błąd na FUS to to samo
  dwuznaczne pytanie o art. 25 ust. 1, na którym mylą się wszystkie modele poniżej 27B (cytują przepisy
  o ponownym ustaleniu emerytury – treść odpowiedzi jest poprawna, cytat nie). Na pytaniu tak/nie o umowę
  o pracę 8B odpowiada poprawnie, czego 4B nie potrafi.
* **Wynik Gemma 4 12B na T4 był zafałszowany**: 6 pytań padło na błędzie „model runner has unexpectedly
  stopped”, którego przyczyną okazała się pamięć systemowa maszyny T4 (12,7 GB RAM), nie karta. Na L4 (53 GB RAM)
  ten sam model ma 100 % na obu ustawach (tabela niżej). Embeddingi i reranker ładują się od tej pory w fp16
  (`LEXRAG_DEVICE=auto`), co zwalnia ~2 GB VRAM.
* **Sprzęt**: 4B/E4B potrzebują 4–5,5 GB VRAM (mieszczą się na RTX 3050 6 GB), Qwen3 8B ~7 GB (na 6 GB
  częściowo na CPU – wolniej, ale działa), 12B i 27B 10–12 GB. Aby na NAS-ie uruchomić 8B lub Saluki w pełni
  na karcie, potrzebna jest karta 12–16 GB (np. RTX 4060 Ti 16 GB).
* Błędy w logach Colaba („model runner has unexpectedly stopped”) to ograniczenie pamięci karty, nie modelu:
  po zwolnieniu VRAM te same modele odpowiadają normalnie. Serwera nie trzeba „podnosić” do A100 – T4 wystarcza
  do pomiaru wszystkich pięciu modeli.

**Pomiar na L4 (Google Colab, 11 października 2026, Ollama 0.40.3, ten sam kod; sześć modeli, dwie ustawy po 40 pytań,
wspólny indeks 1 254 jednostek, `colab/lexrag_pomiar_colab.ipynb`):**

| model | FUS: precyzja / pokrycie | zasiłkowa: precyzja / pokrycie | s / pytanie (L4) |
|---|---|---|---|
| **Gemma 4 12B** | **100 % / 100 %** | **100 % / 100 %** | 12–13 |
| Saluki 27B (2-bit) | 100 % / 97,5 % | 95 % / 97,5 % | 24–25 |
| Qwen3 14B | 95 % / 95 % | 100 % / 97,5 % | 16–17 |
| Gemma 4 E4B | 97 % / 92,5 % | 97 % / 97,5 % | 5 |
| Qwen3 8B | 97,5 % / 100 % | 82 % / 100 % | 9–10 |
| Qwen3 4B | 92,5 % / 100 % | 85 % / 100 % | 7–8 |

Kontrola krzyżowa: Qwen3 8B i Saluki pytane o FUS ze wspólnego indeksu obu ustaw dały dokładnie te same liczby co
z indeksu samej ustawy o FUS (97,5 %/100 % i 100 %/97,5 %). Druga ustawa w indeksie nic nie psuje.

Wnioski z dwóch ustaw:

* **Rekomendacja produkcyjna: Gemma 4 12B** – jedyny model bez ani jednego błędu na 80 pytaniach; ~10 GB VRAM,
  12 s/pytanie na L4. Na kartę 6 GB zostaje E4B (97 %), na 8 GB Qwen3 8B z zastrzeżeniem niżej.
* **Qwen3 8B jest niestabilny między ustawami**: na zasiłkowej 7 błędów na 40 – 3 razy zacytował ustęp szczególny
  (1a, 1e, 5–9) zamiast reguły ogólnej, 3 razy odpowiedział niepełnie (bez wymaganej liczby lub terminu), raz podał
  złą podstawę (art. 13 ust. 2 zamiast art. 7). Zbiór jest w porządku: 12B i 14B rozwiązały go w 100 %.
* **Saluki 27B** na zasiłkowej 95 % (2 błędy) – 2-bitowa kompresja kosztuje; 12B w Q4 jest lepszy i dwa razy szybszy.
* Wyszukiwanie przy dwóch ustawach: recall@10 100 % dla obu zbiorów; recall@3 97,5 % (zasiłkowa) i 87,5 % (FUS).
