"""Warstwa odpowiedzi: analiza pytania → wyszukiwanie → odpowiedź z cytatami → kontrola.

Zasady, które dają wysoką precyzję na tym, co system w ogóle odpowiada:
  * model odpowiada WYŁĄCZNIE z dostarczonych jednostek i zwraca ich id,
  * tekst cytatów renderuje system z indeksu po id – model nie ma jak przekręcić przepisu,
  * każda liczba w odpowiedzi (termin, kwota, procent, rok) musi występować w kontekście –
    sprawdzane deterministycznie, bez pytania modelu (kontrola_liczb),
  * każde zdanie odpowiedzi jest sprawdzane osobnym przebiegiem LLM; niepokryte wylatują
    (schemat wymusza werdykt dla KAŻDEGO zdania; brakujące werdykty dopytujemy zdanie po zdaniu –
    małe modele oceniają chętnie tylko pierwsze zdanie),
  * zdania będące pytaniem lub przepisaniem pytania użytkownika odpadają deterministycznie,
  * gdy model nie poda id cytatów, odczytujemy adresy „art. 58 ust. 2” z treści odpowiedzi
    i mapujemy je na jednostki z kontekstu (tylko z kontekstu),
  * gdy brakuje parametrów – dopytanie; gdy brakuje pokrycia – odmowa i sprawa do referenta.
"""
from __future__ import annotations

import copy
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import KONFIG
from .index import Indeks
from .llm import BrakLLM, czat_json, llm_dostepny
from .models import Jednostka
from .retrieve import Trafienie, rozszerz, szukaj_pelnie

log = logging.getLogger("lexrag")

SYSTEM = (
    "Jesteś asystentem pracownika instytucji publicznej. Odpowiadasz po polsku, "
    "rzeczowo, wyłącznie na podstawie dostarczonych fragmentów przepisów lub procedur. "
    "Nigdy nie zgadujesz i nie dopowiadasz treści, której nie ma w fragmentach. "
    "Odpowiadasz zawsze jednym poprawnym obiektem JSON zgodnym z podanym schematem."
)

# --- schematy JSON (dekodowanie ograniczone gramatyką, gdy serwer umie) -------------
SCHEMA_ANALIZA = {
    "type": "object",
    "properties": {
        "warianty": {"type": "array", "items": {"type": "string"}},
        "brakujace_parametry": {"type": "array", "items": {"type": "string"}},
        "pytanie_doprecyzowujace": {"type": "string"},
    },
    "required": ["warianty", "brakujace_parametry", "pytanie_doprecyzowujace"],
}
SCHEMA_ODPOWIEDZ = {
    "type": "object",
    "properties": {
        "odpowiedz": {"type": "string"},
        "cytowane_id": {"type": "array", "items": {"type": "string"}},
        "brak_pokrycia": {"type": "boolean"},
        "uwagi": {"type": "string"},
    },
    "required": ["odpowiedz", "cytowane_id", "brak_pokrycia", "uwagi"],
}
SCHEMA_WERYFIKACJA = {
    "type": "object",
    "properties": {
        "zdania": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"nr": {"type": "integer"}, "pokryte": {"type": "boolean"}, "id": {"type": "string"}},
                "required": ["nr", "pokryte", "id"],
            },
        },
        "odpowiada_na_pytanie": {"type": "boolean"},
    },
    "required": ["zdania", "odpowiada_na_pytanie"],
}
SCHEMA_WERYFIKACJA_ZDANIE = {
    "type": "object",
    "properties": {"pokryte": {"type": "boolean"}, "id": {"type": "string"}},
    "required": ["pokryte", "id"],
}

PROMPT_ANALIZA = """Pytanie użytkownika:
{pytanie}

Zadanie:
1. Zaproponuj 3–5 wariantów zapytania do wyszukiwarki przepisów, w języku ustawy i urzędu
   (terminologia prawnicza, synonimy, numery artykułów, jeśli się ich domyślasz).
2. Oceń, czy do wskazania właściwego przepisu brakuje danych o sytuacji osoby
   (np. data urodzenia, płeć, staż, rodzaj pracy, okres, kwota). Dopytuj TYLKO wtedy, gdy bez
   tej informacji nie da się wskazać przepisu ani odpowiedzieć warunkowo ("jeśli X, to …").
   W przeciwnym razie "brakujace_parametry" = [] i "pytanie_doprecyzowujace" = "".

Zwróć JSON: {{"warianty": ["..."], "brakujace_parametry": ["..."], "pytanie_doprecyzowujace": ""}}
"""

PROMPT_ODPOWIEDZ = """Pytanie:
{pytanie}

Fragmenty przepisów (każdy ma id, ścieżkę i treść). Używaj tylko ich:
{fragmenty}

Zasady odpowiedzi:
1. Odpowiadaj WYŁĄCZNIE na podstawie fragmentów. Każde zdanie musi wynikać wprost z któregoś fragmentu.
2. Liczby, terminy, kwoty, procenty i warunki przepisuj dokładnie tak, jak stoją we fragmentach.
3. Jeśli przepis ma warunki lub wyjątki, wymień je; nie rozstrzygaj za użytkownika, gdy brakuje danych.
3a. Odpowiadaj z przepisu OGÓLNEGO (reguły), chyba że pytanie wprost dotyczy sytuacji szczególnej: ponownego
   ustalenia lub przeliczenia świadczenia, konkretnej grupy osób, wyjątku. Przepisy o „ponownym ustaleniu”,
   ustępy z literą (np. ust. 1a) i przepisy przejściowe cytuj jako podstawę tylko wtedy, gdy pytanie o nie pyta;
   w innym razie co najwyżej wspomnij o nich osobno, po odpowiedzi z reguły.
4. Odpowiedź zwięzła: 1–6 zdań, po polsku, bez numerowania zdań; podaj numer artykułu i ustępu w treści.
5. "cytowane_id": id fragmentów, na których opierasz odpowiedź (tylko id z listy powyżej).
6. Jeśli fragmenty nie pozwalają odpowiedzieć, ustaw "brak_pokrycia": true i w "uwagi" napisz, czego brakuje.

Zwróć JSON: {{"odpowiedz": "...", "cytowane_id": ["..."], "brak_pokrycia": false, "uwagi": ""}}
"""

PROMPT_PRZEGLAD = """
Uwaga: pytanie jest ogólne (hasłowe) albo brakuje w nim danych o sytuacji osoby. Zamiast zasady 4 odpowiedz
PRZEGLĄDOWO, w 3–8 zdaniach: wymień wszystkie sytuacje, warunki, terminy i wysokości, które regulują dostarczone
fragmenty, każdą z numerem artykułu i ustępu, zaczynając od reguły ogólnej, potem wyjątki i grupy szczególne.
Nie zakładaj danych, których pytanie nie podaje (wiek, staż, daty, rodzaj umowy) – pokaż, od czego zależy wynik.
Ostatnie zdanie: jakie dane o sytuacji osoby pozwolą odpowiedzieć konkretnie.
"""

_PYTAJNIKI = {"czy", "ile", "kiedy", "kto", "komu", "kogo", "jak", "jaki", "jaka", "jakie", "jakim", "jakiej",
              "jakiego", "co", "czego", "czemu", "gdzie", "dlaczego", "który", "która", "które", "którego",
              "skąd", "dokąd", "odkąd", "dokiedy"}


def pytanie_haslowe(pytanie: str) -> bool:
    """Pytanie hasłowe: kilka słów bez zaimka pytajnego i bez znaku zapytania, np. „wcześniejsza emerytura”.
    Takie pytanie dostaje odpowiedź przeglądową (wszystko, co regulują znalezione przepisy) zamiast strzału w jeden ustęp."""
    if pytanie.strip().endswith("?"):
        return False
    slowa = re.findall(r"\w+", pytanie.lower())
    return 0 < len(slowa) <= 6 and not any(s in _PYTAJNIKI for s in slowa)


PROMPT_WERYFIKACJA = """Sprawdzasz odpowiedź asystenta na pytanie użytkownika.

Pytanie użytkownika:
{pytanie}

Fragmenty przepisów:
{fragmenty}

Zdania odpowiedzi (numerowane):
{zdania}

Zadanie 1 – dla każdego zdania: "pokryte" = true, gdy treść zdania wynika wprost z fragmentów
(parafraza jest w porządku). "pokryte" = false, gdy zdanie dodaje fakty, liczby, terminy, warunki lub
wyjątki, których we fragmentach nie ma, albo przeinacza ich sens. Zdania czysto porządkujące
("Poniżej warunki:") traktuj jako pokryte. "id" = id fragmentu, który pokrywa zdanie (albo "").
Zadanie 2 – "odpowiada_na_pytanie": true, gdy odpowiedź jako całość dotyczy tego, o co pytano;
false, gdy mówi o innej kwestii, innym świadczeniu, innym warunku lub innym terminie niż w pytaniu
(nawet jeśli każde zdanie ma pokrycie w przepisach).

Zwróć JSON: {{"zdania": [{{"nr": 1, "pokryte": true, "id": "..."}}], "odpowiada_na_pytanie": true}}
"""

PROMPT_WERYFIKACJA_ZDANIE = """Sprawdzasz jedno zdanie odpowiedzi asystenta.

Fragmenty przepisów:
{fragmenty}

Zdanie:
{zdanie}

"pokryte" = true, gdy treść zdania wynika wprost z fragmentów (parafraza jest w porządku).
"pokryte" = false, gdy zdanie dodaje fakty, liczby, terminy, warunki lub wyjątki, których we fragmentach
nie ma, albo przeinacza ich sens. "id" = id fragmentu, który pokrywa zdanie (albo "").

Zwróć JSON: {{"pokryte": true, "id": "..."}}
"""


@dataclass
class Odpowiedz:
    tryb: str                                   # "dopytanie" | "odpowiedz" | "odmowa" | "retrieval-only" | "blad"
    tekst: str = ""
    cytaty: list[Jednostka] = field(default_factory=list)
    zdania_usuniete: list[str] = field(default_factory=list)
    brakujace_parametry: list[str] = field(default_factory=list)
    warianty: list[str] = field(default_factory=list)
    trafienia: list[Trafienie] = field(default_factory=list)
    kontekst: list[Jednostka] = field(default_factory=list)
    uwagi: str = ""
    czasy: dict = field(default_factory=dict)   # sekundy na krok
    przeglad: bool = False                      # odpowiedź przeglądowa (pytanie hasłowe albo brak danych o sytuacji)
    dopytanie: str = ""                         # pytanie doprecyzowujące dołączone do odpowiedzi przeglądowej


def _fragmenty_do_promptu(jednostki: list[Jednostka]) -> str:
    return "\n\n".join(f"[id: {j.id}] ({j.sciezka})\n{j.tekst}" for j in jednostki)


# --- podział na zdania (bez łamania po skrótach: art., ust., poz., Dz. U., r., np.) ---------
_SKROTY = set(
    "art ust pkt lit poz dz u r nr np tj tzn in ok zł gr proc par al ww tzw godz min str itd itp "
    "ds wł ewent zob por cd m t j ś p s w z o e d n b c a k l f g h i".split()
)
_RE_KANDYDAT = re.compile(r"(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ0-9„\"(])")


def _zdania(tekst: str) -> list[str]:
    tekst = " ".join(tekst.split())
    if not tekst:
        return []
    zdania: list[str] = []
    start = 0
    for m in _RE_KANDYDAT.finditer(tekst):
        przed = tekst[start:m.start()]
        slowo = re.split(r"[.\s(„\"/-]", przed.rstrip(".!?"))[-1].lower()
        if slowo in _SKROTY:
            continue
        if przed.strip():
            zdania.append(przed.strip())
        start = m.end()
    reszta = tekst[start:].strip()
    if reszta:
        zdania.append(reszta)
    return zdania


_RE_SLOWO = re.compile(r"[0-9a-ząćęłńóśźż]+")


def powtarza_pytanie(zdanie: str, pytanie: str) -> bool:
    """Zdanie, które jest pytaniem albo przepisaniem pytania użytkownika (małe modele tak zaczynają
    odpowiedź: „Z ilu lat bierze się zarobki…?”). Takie zdanie nie niesie treści i nie może być odpowiedzią."""
    z = zdanie.strip()
    if z.endswith("?"):
        return True
    a = set(_RE_SLOWO.findall(z.lower()))
    b = set(_RE_SLOWO.findall(pytanie.lower()))
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.75


# --- odzysk cytatów: id podane wprost, id w tekście albo adres „art. 58 ust. 2” / „§ 5” ----------
_RE_ID_W_TEKSCIE = re.compile(r"\b[a-z0-9_]+:art\d+[a-z]*(?::ust\d+[a-z]*)?(?:-[b-z])?\b")
_RE_ADRES = re.compile(r"(?:art\.|§)\s*(\d+[a-z]*)(?:\s+ust\.\s*(\d+[a-z]*))?", re.I)


def odzyskaj_cytaty(tekst: str, surowe: list, kontekst: list[Jednostka]) -> list[str]:
    """Id jednostek KONTEKSTU, na które wskazuje model. Kolejność prób: poprawne id w „cytowane_id”,
    id lub adresy artykułów wpisane w „cytowane_id”, a dopiero gdy nic z tego – adresy w treści odpowiedzi.
    Adres bez ustępu → jednostka artykułowa (albo jej ustępy z kontekstu); adres z ustępem, którego nie ma
    w kontekście → sama jednostka artykułowa (nigdy inny ustęp)."""
    dozwolone = {j.id for j in kontekst}
    artykuly: dict[str, list[str]] = {}
    ustepy: dict[str, list[str]] = {}
    ustepy_art: dict[str, list[str]] = {}
    for j in kontekst:
        cz = j.id.split(":")
        if len(cz) < 2 or not cz[1].startswith("art"):
            continue
        art = re.sub(r"-[b-z]$", "", cz[1])
        if len(cz) == 2:
            artykuly.setdefault(art, []).append(j.id)
        elif cz[2].startswith("ust"):
            ustepy.setdefault(f"{art}:{re.sub(r'-[b-z]$', '', cz[2])}", []).append(j.id)
            ustepy_art.setdefault(art, []).append(j.id)

    wynik: list[str] = []

    def dodaj(ids: list[str]) -> None:
        for i in ids:
            if i in dozwolone and i not in wynik:
                wynik.append(i)

    def z_adresow(t: str) -> None:
        dodaj(_RE_ID_W_TEKSCIE.findall(t))
        for m in _RE_ADRES.finditer(t):
            art = f"art{m.group(1).lower()}"
            if m.group(2):
                dodaj(ustepy.get(f"{art}:ust{m.group(2).lower()}") or artykuly.get(art, []))
            else:
                dodaj(artykuly.get(art) or ustepy_art.get(art, []))

    for c in surowe:
        if isinstance(c, str) and c.strip() in dozwolone:
            dodaj([c.strip()])
    if wynik:
        return wynik
    for c in surowe:
        if isinstance(c, str):
            z_adresow(c)
    if wynik:
        return wynik
    z_adresow(tekst)
    return wynik


# --- kontrola liczb: każda liczba w zdaniu musi występować w kontekście --------------------
_RE_LICZBA = re.compile(r"\d+(?:[.,]\d+)*")
_SLOWNE = {
    "1": ["jeden", "jedn", "jedno", "jednego", "jednym", "jednej", "jedna", "jeden"],
    "2": ["dwa", "dwu", "dwóch", "dwie", "dwoma", "dwiema"], "3": ["trzy", "trzech", "trzema"],
    "4": ["cztery", "czterech", "czterema"], "5": ["pięć", "pięciu", "pięcioma"],
    "6": ["sześć", "sześciu"], "7": ["siedem", "siedmiu"], "8": ["osiem", "ośmiu"],
    "9": ["dziewięć", "dziewięciu"], "10": ["dziesięć", "dziesięciu"], "11": ["jedenaście", "jedenastu"],
    "12": ["dwanaście", "dwunastu"], "14": ["czternaście", "czternastu"], "15": ["piętnaście", "piętnastu"],
    "18": ["osiemnaście", "osiemnastu"], "20": ["dwadzieścia", "dwudziestu"],
    "24": ["dwadzieścia cztery", "dwudziestu czterech"], "25": ["dwadzieścia pięć", "dwudziestu pięciu"],
    "30": ["trzydzieści", "trzydziestu"], "36": ["trzydzieści sześć", "trzydziestu sześciu"],
    "40": ["czterdzieści", "czterdziestu"], "50": ["pięćdziesiąt", "pięćdziesięciu"],
    "60": ["sześćdziesiąt", "sześćdziesięciu"], "65": ["sześćdziesiąt pięć", "sześćdziesięciu pięciu"],
    "90": ["dziewięćdziesiąt", "dziewięćdziesięciu"], "100": ["sto", "stu"],
}


def _normalizuj_liczby(t: str) -> str:
    t = t.replace(" ", " ")
    t = re.sub(r"(?<=\d) (?=\d{3}\b)", "", t)   # "4 000" -> "4000"
    return t.lower()


def korpus_liczb(jednostki: list[Jednostka]) -> str:
    return _normalizuj_liczby("\n".join(f"{j.sciezka}\n{j.tekst}" for j in jednostki))


def liczby_bez_pokrycia(zdanie: str, korpus_norm: str) -> list[str]:
    """Liczby ze zdania, których nie ma w kontekście (ani cyfrą, ani słownie)."""
    brak: list[str] = []
    for m in _RE_LICZBA.finditer(_normalizuj_liczby(zdanie)):
        n = m.group(0)
        warianty = {n, n.replace(",", "."), n.replace(".", ",")}
        if n.isdigit():
            warianty.update(_SLOWNE.get(n, []))
        if not any(re.search(r"(?<!\d)" + re.escape(w) + r"(?!\d)", korpus_norm) for w in warianty):
            if n not in brak:
                brak.append(n)
    return brak


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --- kroki LLM ----------------------------------------------------------------------------
def analizuj_pytanie(pytanie: str) -> dict:
    model = KONFIG.llm_model_maly or None
    return czat_json([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": PROMPT_ANALIZA.format(pytanie=pytanie)}],
                     model=model, max_tokens=500, schema=SCHEMA_ANALIZA)


def generuj(pytanie: str, kontekst: list[Jednostka], przeglad: bool = False) -> dict:
    tresc = PROMPT_ODPOWIEDZ.format(pytanie=pytanie, fragmenty=_fragmenty_do_promptu(kontekst))
    if przeglad:
        tresc += PROMPT_PRZEGLAD
    return czat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": tresc}],
                     max_tokens=1600 if przeglad else 1200, schema=SCHEMA_ODPOWIEDZ)


def _schema_weryfikacji(n: int) -> dict:
    """Schemat z wymuszoną liczbą werdyktów – bez tego małe modele oceniają tylko pierwsze zdanie."""
    sch = copy.deepcopy(SCHEMA_WERYFIKACJA)
    sch["properties"]["zdania"]["minItems"] = n
    sch["properties"]["zdania"]["maxItems"] = n
    return sch


def weryfikuj(zdania: list[str], kontekst: list[Jednostka], pytanie: str = "") -> tuple[list[str], list[str], bool]:
    """Zwraca (zdania pokryte, zdania usunięte, czy odpowiedź dotyczy pytania) wg osobnego przebiegu LLM.
    Każde zdanie dostaje werdykt: z przebiegu zbiorczego, a gdy go tam brakuje – z osobnego pytania o to zdanie."""
    if not zdania or not kontekst:
        return [], list(zdania), True
    model = KONFIG.llm_model_maly or None
    fragmenty = _fragmenty_do_promptu(kontekst)
    numerowane = "\n".join(f"{i + 1}. {z}" for i, z in enumerate(zdania))
    wynik = czat_json([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": PROMPT_WERYFIKACJA.format(
                           pytanie=pytanie or "(brak)", fragmenty=fragmenty, zdania=numerowane)}],
                      model=model, max_tokens=800, schema=_schema_weryfikacji(len(zdania)))
    wpisy = wynik.get("zdania", [])
    wpisy = [w for w in wpisy if isinstance(w, dict)] if isinstance(wpisy, list) else []
    pokryte: dict[int, bool] = {}
    if len(wpisy) == len(zdania):                       # komplet – ufamy kolejności, nie numeracji modelu
        pokryte = {i + 1: bool(w.get("pokryte")) for i, w in enumerate(wpisy)}
    else:
        for w in wpisy:
            try:
                nr = int(w["nr"])
            except (KeyError, ValueError, TypeError):
                continue
            if 1 <= nr <= len(zdania) and nr not in pokryte:
                pokryte[nr] = bool(w.get("pokryte"))
    for i, z in enumerate(zdania, start=1):              # brakujące werdykty: zdanie po zdaniu
        if i in pokryte:
            continue
        log.info("Weryfikacja: brak werdyktu dla zdania %d – pytam osobno.", i)
        w1 = czat_json([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": PROMPT_WERYFIKACJA_ZDANIE.format(fragmenty=fragmenty, zdanie=z)}],
                       model=model, max_tokens=200, schema=SCHEMA_WERYFIKACJA_ZDANIE)
        pokryte[i] = bool(w1.get("pokryte"))
    zostaja = [z for i, z in enumerate(zdania) if pokryte.get(i + 1, False)]
    usuniete = [z for i, z in enumerate(zdania) if not pokryte.get(i + 1, False)]
    na_temat = wynik.get("odpowiada_na_pytanie", True)
    return zostaja, usuniete, (True if na_temat is None else bool(na_temat))


def _z_rodzicami(indeks: Indeks, jednostki: list[Jednostka]) -> list[Jednostka]:
    out: list[Jednostka] = []
    widziane: set[str] = set()
    for j in jednostki:
        for ident in (j.id, j.rodzic):
            if ident and ident not in widziane and ident in indeks.jednostki:
                widziane.add(ident)
                out.append(indeks.jednostki[ident])
    return out


# --- pełny przebieg -----------------------------------------------------------------------
def odpowiedz(indeks: Indeks, pytanie: str, na_dzien: Optional[str] = None,
              uzyj_emb: bool = True, uzyj_reranker: bool = True,
              dopytuj: bool = True, weryfikacja: bool = True,
              top_n: Optional[int] = None, uzyj_llm: bool = True,
              analiza: Optional[bool] = None) -> Odpowiedz:
    """Pełny przebieg. Bez skonfigurowanego LLM zwraca same trafienia (retrieval-only)."""
    czasy: dict[str, float] = {}
    t0 = time.time()
    if not uzyj_llm or not llm_dostepny():
        traf = szukaj_pelnie(indeks, pytanie, na_dzien=na_dzien, uzyj_emb=uzyj_emb,
                             uzyj_reranker=uzyj_reranker, top_n=top_n)
        czasy["szukanie"] = round(time.time() - t0, 1)
        return Odpowiedz(tryb="retrieval-only", trafienia=traf, kontekst=rozszerz(indeks, traf), czasy=czasy)

    # krok 1 – analiza pytania i warianty (pomijalny: --bez-analizy / LEXRAG_SKIP_ANALYSIS=1)
    warianty: list[str] = []
    brakujace: list[str] = []
    pyt_dop = ""
    przeglad = pytanie_haslowe(pytanie)
    if analiza is None:
        analiza = not KONFIG.pomin_analize
    if analiza:
        try:
            a = analizuj_pytanie(pytanie)
            warianty = [w.strip() for w in a.get("warianty", []) if isinstance(w, str) and w.strip()][:5]
            brakujace = [b.strip() for b in a.get("brakujace_parametry", []) if isinstance(b, str) and b.strip()]
            pyt_dop = (a.get("pytanie_doprecyzowujace") or "").strip()
            czasy["analiza"] = round(time.time() - t0, 1)
            if brakujace and pyt_dop:
                przeglad = True   # zamiast samego dopytania: odpowiedź przeglądowa z dopytaniem na końcu
        except (BrakLLM, ValueError) as e:
            log.warning("Analiza pytania nieudana (%s) – szukam samym pytaniem.", e)
            czasy["analiza"] = round(time.time() - t0, 1)

    # kroki 2–4 – wyszukiwanie i kontekst
    t1 = time.time()
    traf = szukaj_pelnie(indeks, pytanie, warianty, na_dzien=na_dzien,
                         uzyj_emb=uzyj_emb, uzyj_reranker=uzyj_reranker, top_n=top_n)
    kontekst = rozszerz(indeks, traf)
    czasy["szukanie"] = round(time.time() - t1, 1)
    if not kontekst:
        return Odpowiedz(tryb="odmowa", tekst="Nie znaleziono przepisów pasujących do pytania.",
                         warianty=warianty, trafienia=traf, czasy=czasy)
    if KONFIG.prog_rerank > 0 and traf and traf[0].zrodlo == "rerank":
        p = _sigmoid(traf[0].wynik)
        if p < KONFIG.prog_rerank:
            return Odpowiedz(tryb="odmowa", warianty=warianty, trafienia=traf, kontekst=kontekst, czasy=czasy,
                             tekst=f"Nie znaleziono przepisu pasującego do pytania (najlepszy kandydat: "
                                   f"{p:.0%} trafności, próg {KONFIG.prog_rerank:.0%}).")

    # krok 5 – odpowiedź z id cytatów
    t2 = time.time()
    try:
        g = generuj(pytanie, kontekst, przeglad=przeglad)
    except (BrakLLM, ValueError) as e:
        return Odpowiedz(tryb="blad", tekst=f"Błąd LLM: {e}", trafienia=traf, kontekst=kontekst, czasy=czasy)
    czasy["odpowiedz"] = round(time.time() - t2, 1)

    tekst = (g.get("odpowiedz") or "").strip()
    surowe = g.get("cytowane_id") or []
    cyt_id = odzyskaj_cytaty(tekst, surowe if isinstance(surowe, list) else [surowe], kontekst)
    cytaty = [indeks.jednostki[c] for c in cyt_id]
    uwagi = (g.get("uwagi") or "").strip()
    wspolne = dict(warianty=warianty, trafienia=traf, kontekst=kontekst, cytaty=cytaty, czasy=czasy,
                   przeglad=przeglad, brakujace_parametry=brakujace, dopytanie=(pyt_dop if dopytuj else ""))

    def _odmowa(**kw) -> Odpowiedz:
        """Odmowa; gdy analiza przygotowała dopytanie, oddajemy je zamiast samej odmowy (powód trafia do uwag)."""
        if dopytuj and pyt_dop:
            kw["uwagi"] = " ".join(x for x in (kw.get("tekst", ""), kw.get("uwagi", "")) if x).strip()
            kw["tekst"] = pyt_dop
            return Odpowiedz(**kw, tryb="dopytanie")
        return Odpowiedz(**kw, tryb="odmowa")
    if g.get("brak_pokrycia") or not tekst:
        return _odmowa( uwagi=uwagi, **wspolne,
                         tekst="Dostarczone przepisy nie pozwalają odpowiedzieć jednoznacznie. " + uwagi)
    if not cytaty:
        return _odmowa( uwagi=uwagi, **wspolne,
                         tekst="Model nie wskazał podstawy prawnej spośród dostarczonych fragmentów – "
                              "sprawa wymaga referenta.")

    # krok 6a – kontrole deterministyczne: powtórzenie pytania, liczby spoza przepisów
    usuniete: list[str] = []
    zdania: list[str] = []
    for z in _zdania(tekst):
        if powtarza_pytanie(z, pytanie):
            usuniete.append(f"{z}  [powtórzenie pytania]")
        else:
            zdania.append(z)
    if KONFIG.kontrola_liczb:
        korpus = korpus_liczb(kontekst)
        zostaja: list[str] = []
        for z in zdania:
            brak = liczby_bez_pokrycia(z, korpus)
            if brak:
                usuniete.append(f"{z}  [liczby spoza przepisów: {', '.join(brak)}]")
            else:
                zostaja.append(z)
        zdania = zostaja

    # krok 6b – weryfikacja zdanie po zdaniu (LLM)
    if weryfikacja and zdania:
        t3 = time.time()
        try:
            pyt_wer = pytanie + (" [pytanie hasłowe – odpowiedź przeglądowa omawiająca to świadczenie lub"
                                 " instytucję z kilku stron dotyczy pytania]" if przeglad else "")
            zdania, usun_llm, na_temat = weryfikuj(zdania, _z_rodzicami(indeks, cytaty), pyt_wer)
            usuniete += [f"{z}  [bez pokrycia wg weryfikacji]" for z in usun_llm]
            if not na_temat:
                czasy["weryfikacja"] = round(time.time() - t3, 1)
                return _odmowa( uwagi=uwagi, **wspolne,
                                 zdania_usuniete=usuniete + [f"{z}  [nie na temat pytania]" for z in zdania],
                                 tekst="Odpowiedź modelu dotyczyła innej kwestii niż pytanie – sprawa wymaga referenta.")
        except (BrakLLM, ValueError) as e:
            log.warning("Weryfikacja nieudana (%s) – odpowiedź oznaczona jako niezweryfikowana.", e)
            uwagi = (uwagi + " NIEZWERYFIKOWANA: " + str(e)).strip()
        czasy["weryfikacja"] = round(time.time() - t3, 1)

    tekst = " ".join(zdania).strip()
    if not tekst:
        return _odmowa( zdania_usuniete=usuniete, uwagi=uwagi, **wspolne,
                         tekst="Żadne zdanie odpowiedzi nie znalazło pokrycia w przepisach – sprawa wymaga referenta.")
    return Odpowiedz(tryb="odpowiedz", tekst=tekst, zdania_usuniete=usuniete, uwagi=uwagi, **wspolne)
