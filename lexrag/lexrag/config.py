"""Konfiguracja przez zmienne środowiskowe (README ma komplet z opisem)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass


def _env(nazwa: str, domyslna: str) -> str:
    return os.environ.get(nazwa, domyslna)


@dataclass(frozen=True)
class Konfig:
    # --- embeddingi (sentence-transformers; opcjonalne) ---
    # mmlw-retrieval wymaga prefiksu zapytania "zapytanie: "; bge-m3 – bez prefiksów;
    # multilingual-e5 – "query: " / "passage: ". Ścieżka lokalna też działa.
    emb_model: str = _env("LEXRAG_EMB_MODEL", "sdadas/mmlw-retrieval-roberta-large")
    emb_prefix_zapytania: str = _env("LEXRAG_EMB_QUERY_PREFIX", "zapytanie: ")
    emb_prefix_dokumentu: str = _env("LEXRAG_EMB_DOC_PREFIX", "")
    emb_wylacz: bool = _env("LEXRAG_DISABLE_EMB", "0") == "1"
    # urządzenie dla embeddingów i rerankera: "auto" = cuda, gdy jest karta NVIDIA, inaczej cpu;
    # "cpu" wymusza procesor (np. gdy GPU ma zostać w całości dla Ollamy); "cuda" wymusza kartę
    urzadzenie: str = _env("LEXRAG_DEVICE", "auto")

    # --- reranker (cross-encoder; opcjonalny) ---
    reranker_model: str = _env("LEXRAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_wylacz: bool = _env("LEXRAG_DISABLE_RERANKER", "0") == "1"
    # 0 = wyłączone; np. 0.05 – gdy najlepszy kandydat ma prawdopodobieństwo trafności (sigmoida
    # z wyniku rerankera) poniżej progu, system odmawia zamiast puszczać do LLM nietrafiony kontekst
    prog_rerank: float = float(_env("LEXRAG_RERANK_THRESHOLD", "0"))

    # --- LLM on-premise ---
    # Ollama: http://localhost:11434/v1 (tryb "ollama" – natywne /api/chat, schemat JSON, num_ctx)
    # llama.cpp / vLLM / LM Studio: http://host:8080/v1 (tryb "openai" – /v1/chat/completions)
    llm_base_url: str = _env("LEXRAG_LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api: str = _env("LEXRAG_LLM_API", "auto")          # auto | ollama | openai
    llm_model: str = _env("LEXRAG_LLM_MODEL", "")            # puste = brak LLM, tryb retrieval-only
    llm_model_maly: str = _env("LEXRAG_LLM_MODEL_SMALL", "") # do kroków 1 i 6; puste = ten sam
    llm_api_key: str = _env("LEXRAG_LLM_API_KEY", "nie-wymagany")
    llm_timeout_s: int = int(_env("LEXRAG_LLM_TIMEOUT", "900"))
    # okno kontekstu modelu: Ollama bez tego ucina prompt po cichu (domyślnie 4096 tokenów)
    llm_num_ctx: int = int(_env("LEXRAG_LLM_NUM_CTX", "12288"))
    # "0" wyłącza tryb "myślenia" (Qwen3, DeepSeek-R1, Saluki…) – szybciej i JSON nie ucieka w <think>;
    # "1" włącza; "" – nie wysyłaj pola (modele bez obsługi think)
    llm_think: str = _env("LEXRAG_LLM_THINK", "0")
    llm_seed: int = int(_env("LEXRAG_LLM_SEED", "7"))
    # dodatkowe pola body jako JSON, np. dla llama.cpp: {"chat_template_kwargs": {"enable_thinking": false}}
    llm_extra_json: str = _env("LEXRAG_LLM_EXTRA_JSON", "")

    # --- retrieval ---
    k_kazde: int = int(_env("LEXRAG_K_EACH", "30"))      # top-k z BM25 i z embeddingów na wariant
    rrf_k: int = int(_env("LEXRAG_RRF_K", "60"))
    top_po_reranku: int = int(_env("LEXRAG_TOP_N", "8"))
    max_znakow_kontekstu: int = int(_env("LEXRAG_MAX_CTX_CHARS", "12000"))  # ~4 tys. tokenów

    # --- kroki LLM ---
    # 1 = pomiń krok analizy pytania (warianty zapytań, dopytanie) – jedno wywołanie LLM mniej;
    # sensowne na wolnym sprzęcie, gdy embeddingi + reranker i tak dają dobry kontekst
    pomin_analize: bool = _env("LEXRAG_SKIP_ANALYSIS", "0") == "1"

    # --- kontrola odpowiedzi ---
    # każda liczba w odpowiedzi (kwoty, terminy, lata, procenty) musi występować w kontekście;
    # zdania z liczbami "znikąd" są usuwane bez pytania modelu
    kontrola_liczb: bool = _env("LEXRAG_NUMERIC_CHECK", "1") == "1"
    # jeden cykl naprawczy: gdy weryfikacja wskaże błędy (odpowiedź nie na temat, bez rozstrzygnięcia pytania
    # zamkniętego, zdania usunięte przez weryfikator), model pisze odpowiedź raz jeszcze z listą błędów i kontrola
    # biegnie ponownie; po drugim niepowodzeniu zostaje to, co przeszło kontrolę. "0" wyłącza.
    naprawa: bool = _env("LEXRAG_NAPRAWA", "1") == "1"

    # --- ślad diagnostyczny ---
    # Pełny ślad (pytanie, wszystkie wiadomości do modelu, surowe odpowiedzi, werdykty) zawiera dane z pytań użytkowników,
    # więc domyślnie NIE jest zapisywany. Włączenie: LEXRAG_TRACE=1 (każda odpowiedź) albo pole "slad" w żądaniu
    # (jedna odpowiedź). Odczyt tylko z adresu lokalnego albo z tokenem LEXRAG_TRACE_TOKEN; pliki starsze niż
    # LEXRAG_TRACE_DAYS dni są usuwane przy kolejnym zapisie.
    slad_wlaczony: bool = _env("LEXRAG_TRACE", "0") == "1"
    slad_dni: int = int(_env("LEXRAG_TRACE_DAYS", "7"))
    slad_token: str = _env("LEXRAG_TRACE_TOKEN", "")


KONFIG = Konfig()


def konfiguracja_do_sladu() -> dict:
    """Konfiguracja potrzebna do odtworzenia przebiegu, bez sekretów: klucza API nie zapisujemy, a z adresu serwera
    usuwamy dane logowania (user:hasło@)."""
    return {
        "llm_model": KONFIG.llm_model, "llm_model_maly": KONFIG.llm_model_maly, "llm_api": KONFIG.llm_api,
        "llm_adres": re.sub(r"//[^/@]*@", "//", KONFIG.llm_base_url), "llm_num_ctx": KONFIG.llm_num_ctx,
        "llm_seed": KONFIG.llm_seed, "llm_think": KONFIG.llm_think, "temperatura": 0.0,
        "emb_model": KONFIG.emb_model, "reranker_model": KONFIG.reranker_model, "urzadzenie": KONFIG.urzadzenie,
        "k_kazde": KONFIG.k_kazde, "rrf_k": KONFIG.rrf_k, "top_po_reranku": KONFIG.top_po_reranku,
        "max_znakow_kontekstu": KONFIG.max_znakow_kontekstu, "prog_rerank": KONFIG.prog_rerank,
        "pomin_analize": KONFIG.pomin_analize, "kontrola_liczb": KONFIG.kontrola_liczb, "naprawa": KONFIG.naprawa,
    }


def ustaw(**zmiany) -> None:
    """Zmiana wybranych pól konfiguracji w działającym procesie, np. nowy adres serwera modelu z panelu
    administratora (tunel z Colaba ma inny adres w każdej sesji). Konfig jest zamrożony, więc pola podmieniamy
    przez object.__setattr__; wszystkie moduły trzymają ten sam obiekt KONFIG, więc widzą zmianę od razu."""
    for nazwa, wartosc in zmiany.items():
        if not hasattr(KONFIG, nazwa):
            raise KeyError(nazwa)
        object.__setattr__(KONFIG, nazwa, wartosc)
