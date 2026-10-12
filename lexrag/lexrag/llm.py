"""Klient LLM dla modeli on-premise. Bez zewnętrznych zależności.

Dwa tryby (LEXRAG_LLM_API, domyślnie "auto": ollama gdy adres ma :11434, inaczej openai):

  * "ollama" – natywne API Ollamy (POST /api/chat):
        - `format` = schemat JSON (dekodowanie ograniczone gramatyką → zawsze poprawny JSON),
        - `options.num_ctx` – bez tego Ollama po cichu ucina prompt do domyślnego okna,
        - `think` – wyłączenie trybu myślenia (Qwen3/Saluki/DeepSeek) – inaczej model zużywa
          limit tokenów na rozważania, a JSON przychodzi ucięty,
        - `seed` – powtarzalność.
  * "openai" – dowolny serwer zgodny z /v1/chat/completions (llama.cpp server, vLLM, LM Studio):
        `response_format` json_schema → json_object → bez (degradacja, gdy serwer nie zna pola).

Zmiana sprzętu (CPU → karta, inny serwer) to tylko zmiana LEXRAG_LLM_BASE_URL.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .config import KONFIG

log = logging.getLogger("lexrag")

# --- ślad diagnostyczny: pełne żądania i odpowiedzi modelu w bieżącym wątku ----------------
_SLAD = threading.local()


def rozpocznij_slad() -> list:
    """Od tej chwili każde wywołanie modelu w tym wątku zapisuje wiadomości, parametry, surową odpowiedź i błędy."""
    _SLAD.wpisy = []
    return _SLAD.wpisy


def zakoncz_slad() -> list:
    wpisy = getattr(_SLAD, "wpisy", None)
    _SLAD.wpisy = None
    return wpisy or []


def _do_sladu(wpis: dict) -> None:
    wpisy = getattr(_SLAD, "wpisy", None)
    if wpisy is not None:
        wpisy.append(wpis)


class BrakLLM(RuntimeError):
    pass


class _BladHTTP(Exception):
    def __init__(self, kod: int, tresc: str):
        super().__init__(f"HTTP {kod}: {tresc}")
        self.kod = kod
        self.tresc = tresc


def llm_dostepny() -> bool:
    return bool(KONFIG.llm_model)


def resetuj_obsluge() -> None:
    """Po zmianie serwera modelu zapomnij, czego poprzedni serwer nie obsługiwał (schemat JSON, think)."""
    _OBSLUGA.update(schema=True, json=True, think=True)


def tryb_api() -> str:
    if KONFIG.llm_api in ("openai", "ollama"):
        return KONFIG.llm_api
    return "ollama" if ":11434" in KONFIG.llm_base_url else "openai"


def _post(url: str, cialo: dict, naglowki: Optional[dict] = None) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(cialo).encode("utf-8"),
        headers={"Content-Type": "application/json", **(naglowki or {})}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=KONFIG.llm_timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise _BladHTTP(e.code, e.read().decode("utf-8", "replace")[:600]) from e
    except urllib.error.URLError as e:
        raise BrakLLM(f"LLM nieosiągalny pod {url}: {e.reason}") from e
    except (TimeoutError, OSError) as e:
        raise BrakLLM(f"LLM nie odpowiedział w {KONFIG.llm_timeout_s} s ({url}): {e}") from e


def _extra() -> dict:
    if not KONFIG.llm_extra_json:
        return {}
    try:
        return json.loads(KONFIG.llm_extra_json)
    except json.JSONDecodeError:
        log.warning("LEXRAG_LLM_EXTRA_JSON nie jest poprawnym JSON – ignoruję.")
        return {}


# --- Ollama (natywne) --------------------------------------------------------
# co serwer/runner obsługuje – ustalane przy pierwszym odrzuceniu, żeby nie powtarzać błędnych żądań
# (runner MLX w Ollamie na Apple Silicon zwraca 501 "structured output is unavailable" dla schematu)
_OBSLUGA: dict[str, bool] = {"schema": True, "json": True, "think": True}


def _post_ollama(baza: str, cialo: dict, model: str) -> dict:
    """POST /api/chat z degradacją: schemat JSON → format=json → sam prompt; bez 'think', gdy nieznane."""
    if not _OBSLUGA["think"]:
        cialo.pop("think", None)
    if isinstance(cialo.get("format"), dict) and not _OBSLUGA["schema"]:
        cialo["format"] = "json"
    if cialo.get("format") == "json" and not _OBSLUGA["json"]:
        cialo.pop("format", None)
    for _ in range(4):
        try:
            return _post(f"{baza}/api/chat", cialo)
        except _BladHTTP as e:
            t = e.tresc.lower()
            if "think" in cialo and "think" in t:
                _OBSLUGA["think"] = False
                cialo.pop("think")
                log.debug("Model bez trybu myślenia – powtarzam bez 'think'.")
                continue
            if isinstance(cialo.get("format"), dict) and (e.kod == 501 or "structured" in t or "format" in t or "schema" in t):
                _OBSLUGA["schema"] = False
                cialo["format"] = "json"
                log.warning("Serwer nie obsługuje schematu JSON (%s) – używam format=json.", e.tresc[:100])
                continue
            if cialo.get("format") == "json" and (e.kod == 501 or "format" in t or "json" in t or "structured" in t):
                _OBSLUGA["json"] = False
                cialo.pop("format")
                log.warning("Serwer nie obsługuje format=json (%s) – zostaje sam prompt.", e.tresc[:100])
                continue
            if e.kod == 404:
                raise BrakLLM(f"Ollama nie zna modelu '{model}' (sprawdź `ollama list`): {e.tresc}") from e
            raise BrakLLM(f"LLM {e}") from e
    raise BrakLLM("LLM odrzucił wszystkie warianty żądania.")


def _czat_ollama(wiadomosci, schema, json_mode, temperatura, model, max_tokens) -> str:
    baza = re.sub(r"/v1/?$", "", KONFIG.llm_base_url.rstrip("/"))
    cialo: dict[str, Any] = {
        "model": model, "messages": wiadomosci, "stream": False, "keep_alive": "30m",
        "options": {"temperature": temperatura, "num_ctx": KONFIG.llm_num_ctx,
                    "num_predict": max_tokens, "seed": KONFIG.llm_seed},
    }
    if schema:
        cialo["format"] = schema
    elif json_mode:
        cialo["format"] = "json"
    if KONFIG.llm_think in ("0", "1"):
        cialo["think"] = KONFIG.llm_think == "1"
    cialo.update(_extra())
    dane = _post_ollama(baza, cialo, model)
    _SLAD.meta = ({k: dane.get(k) for k in ("prompt_eval_count", "eval_count", "done_reason")} if isinstance(dane, dict) else {})
    _SLAD.meta["format"] = cialo.get("format") if isinstance(cialo.get("format"), str) else ("schemat" if cialo.get("format") else None)
    msg = dane.get("message") if isinstance(dane, dict) else None
    if not isinstance(msg, dict):
        raise BrakLLM(f"Niespodziewana odpowiedź Ollamy: {str(dane)[:300]}")
    tresc = msg.get("content") or ""
    if not tresc and msg.get("thinking"):
        log.warning("Model zwrócił tylko 'thinking' – ustaw LEXRAG_LLM_THINK=0 albo zwiększ max_tokens.")
    if dane.get("done_reason") == "length":
        log.warning("Odpowiedź ucięta limitem tokenów (%d) – zwiększ max_tokens.", max_tokens)
    pe = dane.get("prompt_eval_count") or 0
    if pe and pe >= KONFIG.llm_num_ctx - max_tokens:
        log.warning("Prompt (%d tokenów) zbliża się do okna num_ctx=%d – kontekst mógł zostać ucięty. "
                    "Zwiększ LEXRAG_LLM_NUM_CTX albo zmniejsz LEXRAG_MAX_CTX_CHARS.", pe, KONFIG.llm_num_ctx)
    return tresc


# --- OpenAI-compatible (llama.cpp, vLLM, LM Studio) -----------------------------
def _czat_openai(wiadomosci, schema, json_mode, temperatura, model, max_tokens) -> str:
    url = KONFIG.llm_base_url.rstrip("/") + "/chat/completions"
    naglowki = {"Authorization": f"Bearer {KONFIG.llm_api_key}"}
    baza: dict[str, Any] = {"model": model, "messages": wiadomosci, "temperature": temperatura,
                            "max_tokens": max_tokens, "stream": False, "seed": KONFIG.llm_seed}
    baza.update(_extra())
    warianty: list[dict] = []
    if schema:
        warianty.append({"response_format": {"type": "json_schema",
                                             "json_schema": {"name": "odpowiedz", "schema": schema, "strict": True}}})
    if schema or json_mode:
        warianty.append({"response_format": {"type": "json_object"}})
    warianty.append({})
    dane = None
    for w in warianty:
        try:
            dane = _post(url, {**baza, **w}, naglowki)
            break
        except _BladHTTP as e:
            if e.kod == 400 and w:
                log.debug("Serwer odrzucił %s – próbuję prostszego wariantu (%s).", list(w)[0], e.tresc[:120])
                continue
            raise BrakLLM(f"LLM {e}") from e
    if dane is None:
        raise BrakLLM("LLM odrzucił wszystkie warianty żądania.")
    try:
        wybor = dane["choices"][0]
        tresc = wybor["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise BrakLLM(f"Niespodziewana odpowiedź LLM: {str(dane)[:300]}") from e
    _SLAD.meta = {"finish_reason": wybor.get("finish_reason"), "usage": dane.get("usage")}
    if wybor.get("finish_reason") == "length":
        log.warning("Odpowiedź ucięta limitem tokenów (%d) – zwiększ max_tokens.", max_tokens)
    return tresc


def czat(wiadomosci: list[dict], json_mode: bool = False, temperatura: float = 0.0,
         model: Optional[str] = None, max_tokens: int = 1500, schema: Optional[dict] = None,
         etykieta: str = "") -> str:
    """Jedno wywołanie czatu; zwraca treść odpowiedzi (string).

    `schema` – JSON Schema oczekiwanej odpowiedzi (dekodowanie ograniczone, gdy serwer umie);
    `json_mode` – tylko "odpowiedz JSON-em", bez schematu; `etykieta` – nazwa kroku w śladzie diagnostycznym.
    """
    model = model or KONFIG.llm_model
    if not model:
        raise BrakLLM("Nie ustawiono LEXRAG_LLM_MODEL – tryb retrieval-only.")
    _SLAD.meta = {}
    t0 = time.time()
    wpis = {"etykieta": etykieta, "model": model, "tryb": tryb_api(), "messages": wiadomosci,
            "max_tokens": max_tokens, "temperatura": temperatura, "seed": KONFIG.llm_seed,
            "num_ctx": KONFIG.llm_num_ctx, "schemat": schema is not None, "json_mode": json_mode}
    try:
        if tryb_api() == "ollama":
            tresc = _czat_ollama(wiadomosci, schema, json_mode, temperatura, model, max_tokens)
        else:
            tresc = _czat_openai(wiadomosci, schema, json_mode, temperatura, model, max_tokens)
    except Exception as e:
        wpis.update(blad=f"{type(e).__name__}: {e}", czas_s=round(time.time() - t0, 1))
        _do_sladu(wpis)
        raise
    wpis.update(odpowiedz_surowa=tresc, czas_s=round(time.time() - t0, 1), meta=getattr(_SLAD, "meta", {}))
    _do_sladu(wpis)
    return tresc


_RE_JSON = re.compile(r"\{.*\}", re.S)
_RE_THINK = re.compile(r"<think>.*?</think>", re.S)


def wyciagnij_json(tekst: str) -> dict:
    """Wyciąga pierwszy obiekt JSON z odpowiedzi modelu (toleruje ```json ... ``` i <think>)."""
    tekst = _RE_THINK.sub("", tekst).strip()
    tekst = re.sub(r"^```(?:json)?\s*|\s*```$", "", tekst, flags=re.I).strip()
    try:
        w = json.loads(tekst)
        if isinstance(w, dict):
            return w
    except json.JSONDecodeError:
        pass
    m = _RE_JSON.search(tekst)
    if m:
        try:
            w = json.loads(m.group(0))
            if isinstance(w, dict):
                return w
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Model nie zwrócił poprawnego JSON: {tekst[:200]!r}")


def czat_json(wiadomosci: list[dict], model: Optional[str] = None, proby: int = 2,
              schema: Optional[dict] = None, **kw) -> dict:
    """Czat z wymuszonym JSON. Przy błędzie parsowania powtarza z dopiskiem (temperatura 0 bez
    zmiany promptu dałaby tę samą odpowiedź)."""
    ostatni: Optional[Exception] = None
    wiad = list(wiadomosci)
    for proba in range(proby):
        try:
            return wyciagnij_json(czat(wiad, json_mode=True, model=model, schema=schema, **kw))
        except ValueError as e:
            ostatni = e
            _do_sladu({"etykieta": kw.get("etykieta", ""), "proba": proba + 1, "blad_json": str(e)[:300]})
            wiad = list(wiadomosci) + [{"role": "user", "content":
                                        "Poprzednia odpowiedź nie była poprawnym JSON. Zwróć WYŁĄCZNIE jeden "
                                        "obiekt JSON zgodny z podanym schematem, bez komentarzy."}]
    raise ValueError(str(ostatni))


def ping(model: Optional[str] = None) -> dict:
    """Szybki test połączenia: tryb API, model, czas, czy JSON działa."""
    import time
    t0 = time.time()
    w = czat_json([{"role": "user", "content": 'Odpowiedz JSON-em: {"ok": true, "model_mowi": "<jedno słowo po polsku>"}'}],
                  model=model, max_tokens=60,
                  schema={"type": "object", "properties": {"ok": {"type": "boolean"}, "model_mowi": {"type": "string"}},
                          "required": ["ok", "model_mowi"]})
    return {"tryb": tryb_api(), "model": model or KONFIG.llm_model, "czas_s": round(time.time() - t0, 1), "odpowiedz": w}
