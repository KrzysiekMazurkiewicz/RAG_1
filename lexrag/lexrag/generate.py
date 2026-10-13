"""Warstwa odpowiedzi: analiza pytania → wyszukiwanie → odpowiedź z cytatami → kontrola → jedna próba naprawy.

Zasady, które dają wysoką precyzję na tym, co system w ogóle odpowiada:
  * model odpowiada WYŁĄCZNIE z dostarczonych jednostek i zwraca ich id,
  * tekst cytatów renderuje system z indeksu po id – model nie ma jak przekręcić przepisu,
  * każda liczba w odpowiedzi musi występować w kontekście (kontrola_liczb) – to warunek konieczny, nie dowód
    poprawności: przypisanie liczby do grupy i warunku sprawdza weryfikator,
  * weryfikator (osobny przebieg LLM) dostaje oryginalne pytanie i ocenia każde zdanie (pokrycie z poprawnym id źródła,
    struktura warunków, negacje, przypisanie liczb, zakres podmiotowy) oraz całość (czy rozstrzyga główną wątpliwość,
    czy nie łączy odrębnych podstaw, czy dopytanie jest konieczne, pominięcia); werdykty mapowane po numerach zdań,
  * po usunięciu zdań to, co zostało, jest weryfikowane jeszcze raz,
  * po wykryciu błędów jedna próba naprawy (pytanie, kontekst, poprzednia odpowiedź, lista błędów) i ponowna kontrola,
  * status_weryfikacji: verified | unverified | insufficient_sources; awaria weryfikatora nigdy nie daje verified,
  * brak danych o sytuacji NIE włącza trybu przeglądowego: odpowiedź warunkowa + dopytanie o dane niezbędne obok niej,
  * data „na dzień” bez wersji historycznej w indeksie daje jawne ostrzeżenie, a nie domysł.
"""
from __future__ import annotations

import copy
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .config import KONFIG, konfiguracja_do_sladu
from .index import Indeks
from .llm import BrakLLM, czat_json, llm_dostepny, rozpocznij_slad, zakoncz_slad
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
        "brakujace_dane": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"dana": {"type": "string"},
                               "zakres": {"type": "string", "enum": ["niezbedna", "do_pelnej_oceny"]},
                               "dlaczego": {"type": "string"}},
                "required": ["dana", "zakres", "dlaczego"],
            },
        },
        "pytanie_doprecyzowujace": {"type": "string"},
    },
    "required": ["warianty", "brakujace_dane", "pytanie_doprecyzowujace"],
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

TYPY_BLEDOW = ["pokrycie", "struktura_warunkow", "negacja", "liczby", "zakres_podmiotowy", "odrebna_podstawa",
               "odeslanie", "zbedne_zdanie", "pominiecie", "niepotrzebne_dopytanie", "nie_na_temat", "inne"]
# błędy, które wskazują konkretne zdanie do usunięcia; pominięcia i „nie na temat” naprawia się odpowiedzią jako całością
TYPY_BLEDOW_ZDANIA = {"pokrycie", "struktura_warunkow", "negacja", "liczby", "zakres_podmiotowy", "odrebna_podstawa",
                      "odeslanie", "zbedne_zdanie", "niepotrzebne_dopytanie"}

_POLA_ZDANIA = {
    "id": {"type": "string"},
    "czesc_pytania": {"type": "string"},
    "odeslanie_w_zrodle": {"type": "string"},
    "pokryte": {"type": "boolean"},
    "struktura_warunkow_ok": {"type": "boolean"},
    "negacje_ok": {"type": "boolean"},
    "liczby_przypisane_ok": {"type": "boolean"},
    "zakres_podmiotowy_ok": {"type": "boolean"},
    "odeslania_ok": {"type": "boolean"},
    "potrzebne_do_pytania": {"type": "boolean"},
}
SCHEMA_WERYFIKACJA = {
    "type": "object",
    "properties": {
        "zdania": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"nr": {"type": "integer"}, **_POLA_ZDANIA},
                      "required": ["nr", *_POLA_ZDANIA]},
        },
        "odpowiada_na_pytanie": {"type": "boolean"},
        "odpowiada_na_glowna_watpliwosc": {"type": "boolean"},
        "nie_laczy_odrebnych_podstaw": {"type": "boolean"},
        "odpowiedz_deklaruje_brak_danych": {"type": "boolean"},
        "dopytanie_jest_konieczne": {"type": "boolean"},
        "istotne_pominiecia": {"type": "array", "items": {"type": "string"}},
        "bledy": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"zdanie": {"type": "integer"}, "typ": {"type": "string", "enum": TYPY_BLEDOW},
                                     "opis": {"type": "string"}, "zrodlo": {"type": "string"}},
                      "required": ["zdanie", "typ", "opis", "zrodlo"]},
        },
    },
    "required": ["zdania", "odpowiada_na_pytanie", "odpowiada_na_glowna_watpliwosc", "nie_laczy_odrebnych_podstaw",
                 "odpowiedz_deklaruje_brak_danych", "dopytanie_jest_konieczne", "istotne_pominiecia",
                 "bledy"],
}
SCHEMA_WERYFIKACJA_ZDANIE = {
    "type": "object",
    "properties": dict(_POLA_ZDANIA),
    "required": list(_POLA_ZDANIA),
}

PROMPT_ANALIZA = """Pytanie użytkownika:
{pytanie}

Zadanie:
1. Zaproponuj 3–5 wariantów zapytania do wyszukiwarki przepisów, w języku ustawy i urzędu: terminologia prawnicza,
   synonimy, nazwa instytucji prawnej i zagadnienia. Nie zgaduj numerów artykułów. Każdy wariant dotyczy jednej z kwestii
   zadanych w pytaniu i zachowuje jej sens; nie dodawaj zagadnień, o które nie zapytano (np. pytanie o wypłatę
   świadczenia to nie pytanie o jego obliczenie, a pytanie o to, czy zarobki mają znaczenie dla wypłaty, to nie
   pytanie o wpływ zarobków na wysokość świadczenia).
2. Oceń, jakich danych o sytuacji osoby brakuje, i rozdziel je na dwie grupy ("zakres"):
   - "niezbedna": bez tej informacji nie da się odpowiedzieć na GŁÓWNE pytanie, bo od niej zależy, który przepis
     stosować albo jaki jest wynik; jeśli pytanie zawiera dość danych, by odpowiedzieć na to, o co pytano, nie zgłaszaj
     danych niezbędnych;
   - "do_pelnej_oceny": informacja potrzebna do pełnej oceny indywidualnej sprawy albo dotycząca innej kwestii niż
     główne pytanie, ale zbędna do odpowiedzi na zadane pytanie.
   Nie zgłaszaj danych, które już podano w pytaniu (wiek, rodzaj zatrudnienia, to, że osoba pracuje), ani takich, które
   nie zmieniają odpowiedzi na zadane pytanie. Nie dopytuj ogólnie o płeć, staż ani rodzaj świadczenia: tylko wtedy,
   gdy od tej danej zależy odpowiedź na to, o co użytkownik pyta, a nie warunki nabycia świadczenia, o które nie pytał. Przy każdej
   brakującej danej napisz w "dlaczego" jednym zdaniem, jak zmienia odpowiedź. Brak danych nie blokuje odpowiedzi
   na tę część pytania, którą da się rozstrzygnąć bez nich.
3. "pytanie_doprecyzowujace" wypełnij tylko wtedy, gdy jest dana "niezbedna"; w przeciwnym razie "".

Zwróć JSON: {{"warianty": ["..."], "brakujace_dane": [{{"dana": "...", "zakres": "niezbedna", "dlaczego": "..."}}], "pytanie_doprecyzowujace": ""}}
"""

PROMPT_ODPOWIEDZ = """Pytanie:
{pytanie}

Fragmenty przepisów (każdy ma id, ścieżkę i treść). Używaj tylko ich:
{fragmenty}

Zasady odpowiedzi:
1. Odpowiadaj WYŁĄCZNIE na podstawie fragmentów. Każde zdanie musi wynikać wprost z któregoś fragmentu.
2. Najpierw odpowiedz na główną wątpliwość użytkownika i na tym poprzestań, chyba że pominięcie czegoś wprowadziłoby
   w błąd (np. istotny wyjątek dla opisanej sytuacji). Obecność fragmentu w kontekście nie oznacza obowiązku jego
   wykorzystania: pomiń przepisy, które nie dotyczą sytuacji z pytania. Każde zdanie musi być potrzebne do odpowiedzi
   na konkretną część pytania. Nie dodawaj przepisów o innym zagadnieniu tylko dlatego, że zawierają podobne słowa albo
   odsyłają do przepisu, który już przytoczono (np. o obliczaniu lub ponownym obliczaniu świadczenia, gdy pytano
   o jego wypłatę). Jeśli jeden przepis rozstrzyga wszystkie części pytania, poprzestań na nim.
3. Na pytanie zamknięte („czy…”) odpowiedz wprost („tak”, „nie” albo „zależy od…” ze wskazaniem warunku), jeśli
   pozwalają na to fragmenty i podane dane. Jeśli nie pozwalają, napisz jasno, czego brakuje i dlaczego to zmienia
   odpowiedź.
4. Rozdziel odpowiedź na wąskie pytanie, które zadano, od pełnej oceny indywidualnego uprawnienia (inne przesłanki,
   daty, dokumenty). Tej drugiej nie rozstrzygaj za użytkownika; jeśli to istotne, wskaż w jednym zdaniu na końcu, co
   jeszcze trzeba by sprawdzić.
5. Każdy przepis (podstawa prawna) ma własne przesłanki. Nie przenoś warunków z jednego przepisu na inny i nie
   przedstawiaj przesłanki odrębnej podstawy jako dodatkowego warunku omawianej; inną podstawę możesz wymienić osobno
   jako inną możliwość, ale tylko wtedy, gdy dotyczy tego, o co pytano.
   Odesłania przenoś dokładnie: gdy przepis odsyła do konkretnych jednostek („ust. 1–3”, „art. 26”), podaj te jednostki,
   a nie zwrot „te przepisy”, „powyższe przepisy” ani „przepisy te”, który mógłby wskazywać inny przepis. Nie pomijaj
   warunku, od którego przepis uzależnia swoje zastosowanie.
6. Zachowaj dokładnie: spójniki „i”, „lub”, „albo”, negacje, liczby, progi, terminy, kwoty, procenty, zakres
   podmiotowy (kogo dotyczy) i zastrzeżenia („chyba że”, „z zastrzeżeniem”). Nie zamieniaj warunku alternatywnego na
   łączny ani odwrotnie; progi przypisuj do właściwej grupy osób i właściwego warunku. Nie streszczaj warunków
   słowami typu „określone warunki”: podaj je albo o nich nie wspominaj. Jeśli przepis ma warunki lub wyjątki istotne
   dla pytania, wymień je.
7. Odpowiadaj z przepisu ogólnego (reguły), chyba że pytanie wprost dotyczy sytuacji szczególnej albo opisana
   sytuacja podpada pod przepis szczególny (konkretną grupę osób, wyjątek). Przepisy o ponownym ustaleniu lub
   przeliczeniu świadczenia oraz przepisy przejściowe cytuj jako podstawę tylko wtedy, gdy pytanie o nie pyta albo
   opisana sytuacja ich wymaga. Gdy pytanie nie opisuje sytuacji szczególnej (konkretnej grupy osób, daty, okoliczności),
   odpowiedz z przepisu ogólnego i nie wymieniaj przepisów dla grup szczególnych.
8. Nie dopytuj o informacje, które już podano w pytaniu, ani o takie, które nie zmieniają odpowiedzi.
9. Odpowiedź zwięzła: 1–6 zdań, po polsku, bez numerowania zdań; podaj numer artykułu i ustępu w treści w formie
   „art. 24 ust. 1”, nigdy jako id w nawiasach kwadratowych ani w formie „fus:art24:ust1”.
10. "cytowane_id": id fragmentów, na których opierasz odpowiedź (tylko id z listy powyżej).
11. Jeśli fragmenty nie pozwalają odpowiedzieć, ustaw "brak_pokrycia": true i w "uwagi" napisz, czego brakuje.

Zwróć JSON: {{"odpowiedz": "...", "cytowane_id": ["..."], "brak_pokrycia": false, "uwagi": ""}}
"""

PROMPT_PRZEGLAD = """
Uwaga: użytkownik podał samo hasło albo poprosił o przegląd. Zamiast limitu z zasady 9 odpowiedz w 3–6 zdaniach,
ale wyłącznie w zakresie pytania:
- zacznij od zasady ogólnej dotyczącej tego tematu;
- omów tylko istotne wyjątki i grupy szczególne tego samego tematu, każde z numerem artykułu i ustępu;
- obecność fragmentu w kontekście nie oznacza obowiązku jego wykorzystania: pomiń przepisy o innych świadczeniach
  i innych kwestiach;
- nie zakładaj danych, których pytanie nie podaje (wiek, staż, daty, rodzaj umowy), ale nie dopytuj, jeśli możesz
  udzielić użytecznej odpowiedzi bez dopytania.
"""

_PYTAJNIKI = {"czy", "ile", "kiedy", "kto", "komu", "kogo", "jak", "jaki", "jaka", "jakie", "jakim", "jakiej",
              "jakiego", "co", "czego", "czemu", "gdzie", "dlaczego", "który", "która", "które", "którego",
              "skąd", "dokąd", "odkąd", "dokiedy"}
# jawna prośba o przegląd: czasownik na początku („wymień…”, „omów…”) albo samo słowo „przegląd” / „wymień”
_RE_PRZEGLAD = re.compile(
    r"^\s*(?:proszę\s+)?(?:wymień|omów|przedstaw|wylicz|podsumuj|scharakteryzuj)\b|\bprzegląd\w*\b|\bwymień\b",
    re.I)


def pytanie_haslowe(pytanie: str) -> bool:
    """Pytanie hasłowe: kilka słów bez zaimka pytajnego i bez znaku zapytania, np. „wcześniejsza emerytura”.
    Takie pytanie dostaje odpowiedź przeglądową (zasada ogólna i istotne wyjątki tematu) zamiast strzału w jeden ustęp."""
    if pytanie.strip().endswith("?"):
        return False
    slowa = re.findall(r"\w+", pytanie.lower())
    return 0 < len(slowa) <= 6 and not any(s in _PYTAJNIKI for s in slowa)


def prosba_o_przeglad(pytanie: str) -> bool:
    """Użytkownik wprost prosi o przegląd („Wymień rodzaje…”, „Omów warunki…”)."""
    return bool(_RE_PRZEGLAD.search(pytanie))


def tryb_przegladowy(pytanie: str) -> bool:
    """Tryb przeglądowy włączają tylko hasło i jawna prośba o przegląd; brak danych o sytuacji osoby go NIE włącza."""
    return pytanie_haslowe(pytanie) or prosba_o_przeglad(pytanie)


PROMPT_WERYFIKACJA = """Sprawdzasz odpowiedź asystenta na pytanie użytkownika. Oceniasz ją niezależnie: jedynym punktem
odniesienia są pytanie i fragmenty przepisów poniżej.

Pytanie użytkownika:
{pytanie}

Pytanie doprecyzowujące przygotowane przez system (może być puste): {dopytanie}

Fragmenty przepisów:
{fragmenty}

Zdania odpowiedzi (numerowane):
{zdania}

Zadanie 1 – dla KAŻDEGO zdania ("nr" = numer zdania; "id" = id fragmentu z listy powyżej, który je pokrywa).
Najpierw wypełnij dwa pola uzasadnienia, dopiero potem oceny:
- "czesc_pytania": własnymi słowami, na którą część pytania użytkownika odpowiada to zdanie; "" (pusty napis), gdy nie
  odpowiada na żadną (dotyczy innego zagadnienia). Przy haśle albo prośbie o przegląd: temat, który zdanie omawia.
- "odeslanie_w_zrodle": jeśli fragment, który pokrywa zdanie, zawiera odesłanie albo zwrot wskazujący („ust. 1–3”,
  „art. 26”, „zgodnie z art. 103a”), przepisz go dokładnie i napisz, do czego się odnosi; inaczej "".
Oceny:
- "pokryte": true, gdy treść zdania wynika wprost z podanego fragmentu (parafraza jest w porządku); false, gdy zdanie
  dodaje fakty, liczby, terminy, warunki lub wyjątki, których we fragmentach nie ma, albo przeinacza ich sens. Zdania
  czysto porządkujące ("Poniżej warunki:") traktuj jako pokryte. Dla zdania pokrytego "id" jest obowiązkowe i musi być
  z listy; dla niepokrytego "".
- "struktura_warunkow_ok": czy zdanie zachowuje strukturę warunków przepisu: warunki łączne ("oraz", "i") zostają
  łączne, alternatywne ("albo", "lub") zostają alternatywne, "jeżeli … to" nie jest odwrócone. False także wtedy, gdy
  zdanie pomija jeden z warunków łącznych albo upraszcza alternatywę tak, że zmienia się krąg osób spełniających
  warunki (np. „określone warunki” w miejsce konkretnych). Zdanie bez warunków: true.
- "negacje_ok": czy zdanie zachowuje negacje i wyłączenia przepisu ("nie", "z wyjątkiem", "chyba że"). Brak negacji
  w przepisie i w zdaniu: true.
- "liczby_przypisane_ok": czy każda liczba, wiek, okres, kwota i procent w zdaniu jest przypisana do właściwej grupy
  osób i właściwego warunku, tak jak w przepisie. To, że liczba występuje gdzieś we fragmentach, nie wystarcza.
  Zdanie bez liczb: true.
- "zakres_podmiotowy_ok": czy zdanie dotyczy tych samych osób i sytuacji co przepis (kto, w jakich okolicznościach);
  false, gdy rozszerza przepis na inną grupę albo zawęża go bez podstawy.
- "odeslania_ok": czy zdanie wiernie oddaje odesłania i zwroty wskazujące (do których ustępów, artykułów, przepisów
  się odnoszą) oraz warunki, od których przepis uzależnia swoje zastosowanie. Porównaj ze swoim opisem w
  "odeslanie_w_zrodle". False, gdy zdanie zmienia adresata
  odesłania (np. przepis odsyła do „ust. 1–3” tego samego artykułu, a zdanie pisze „przepisy te” po opisie innego
  artykułu, więc wskazuje inny przepis) albo pomija warunek zastosowania przepisu (np. że dotyczy tylko osoby, której
  ustalono określone prawo), przez co przepis wygląda na szerszy niż jest. Zdanie bez odesłań i bez takich warunków: true.
- "potrzebne_do_pytania": czy zdanie jest potrzebne do odpowiedzi na którąś część pytania użytkownika; false, gdy
  "czesc_pytania" jest puste. False także, gdy
  dotyczy innego zagadnienia niż pytanie (np. obliczania wysokości świadczenia, gdy pytano o jego wypłatę) i zostało
  dodane tylko dlatego, że przepis zawiera podobne słowa albo odsyła do przepisu głównego. Przy haśle albo prośbie
  o przegląd tematu istotne wyjątki tego samego tematu są potrzebne: true.

Zadanie 2 – ocena całości:
- "odpowiada_na_pytanie": true, gdy odpowiedź dotyczy tego, o co pytano (ta sama kwestia, to samo świadczenie, ten sam
  warunek lub termin); dla pytania w formie hasła (temat bez pytania): gdy omawia ten temat.
- "odpowiada_na_glowna_watpliwosc": true, gdy odpowiedź zawiera to, czego użytkownik potrzebuje: na pytanie zamknięte
  ("czy…") rozstrzyga wprost (tak / nie / zależy od wskazanego warunku) na podstawie fragmentów; na pytanie otwarte
  podaje szukaną informację; na hasło podaje zasadę ogólną. Gdy fragmenty pozwalały rozstrzygnąć, a odpowiedź
  tego nie robi: false.
- "nie_laczy_odrebnych_podstaw": true, gdy odpowiedź nie przedstawia warunku jednego przepisu jako warunku koniecznego
  innego, odrębnego przepisu (innej podstawy prawnej); false, gdy tak robi. Wspomnienie innego przepisu jako osobnej
  możliwości jest w porządku.
- "odpowiedz_deklaruje_brak_danych": true, gdy TREŚĆ odpowiedzi twierdzi, że potrzebne są dodatkowe dane albo
  potwierdzenie czegoś (pytanie doprecyzowujące przygotowane przez system oceniasz w następnym polu, nie tu).
- "dopytanie_jest_konieczne": oceniaj, gdy podano pytanie doprecyzowujące systemu albo odpowiedź deklaruje brak danych;
  w przeciwnym razie false. true tylko wtedy, gdy bez tych danych NIE da się odpowiedzieć na główną wątpliwość
  z pytania, nawet warunkowo, a danych nie ma w pytaniu. Gdy odpowiedź już rozstrzyga główną wątpliwość ("tak", "nie"
  albo "zależy od…" z wyjaśnieniem), pytanie doprecyzowujące nie jest konieczne: false. Dane przydatne tylko do pełnej
  oceny sprawy albo dotyczące innej kwestii niż pytanie: false.
- "istotne_pominiecia": lista warunków, wyjątków lub zastrzeżeń z TYCH SAMYCH przepisów, które odpowiedź stosuje,
  bez których odpowiedź wprowadza w błąd; pusta, gdy ich nie ma.
- "bledy": lista konkretnych błędów, po jednym na pozycję: "zdanie" (numer; 0 dla błędu całości), "typ" (pokrycie |
  struktura_warunkow | negacja | liczby | zakres_podmiotowy | odrebna_podstawa | odeslanie | zbedne_zdanie |
  pominiecie | niepotrzebne_dopytanie | nie_na_temat | inne), "opis" (co jest źle i jak powinno być) i "zrodlo" (id fragmentu, na który się powołujesz, albo
  ""). Pusta, gdy błędów nie ma.

Zwróć JSON: {{"zdania": [{{"nr": 1, "id": "...", "czesc_pytania": "...", "odeslanie_w_zrodle": "", "pokryte": true, "struktura_warunkow_ok": true, "negacje_ok": true, "liczby_przypisane_ok": true, "zakres_podmiotowy_ok": true, "odeslania_ok": true, "potrzebne_do_pytania": true}}], "odpowiada_na_pytanie": true, "odpowiada_na_glowna_watpliwosc": true, "nie_laczy_odrebnych_podstaw": true, "odpowiedz_deklaruje_brak_danych": false, "dopytanie_jest_konieczne": false, "istotne_pominiecia": [], "bledy": []}}
"""

PROMPT_WERYFIKACJA_ZDANIE = """Sprawdzasz jedno zdanie odpowiedzi asystenta na pytanie użytkownika.

Pytanie użytkownika:
{pytanie}

Fragmenty przepisów:
{fragmenty}

Zdanie:
{zdanie}

- "czesc_pytania": na którą część pytania odpowiada zdanie; "" gdy na żadną.
- "odeslanie_w_zrodle": odesłanie we fragmencie (np. „ust. 1–3”) przepisane dokładnie wraz z adresatem; inaczej "".
- "pokryte": true, gdy treść zdania wynika wprost z podanego fragmentu (parafraza jest w porządku); false, gdy zdanie
  dodaje fakty, liczby, terminy, warunki lub wyjątki, których we fragmentach nie ma, albo przeinacza ich sens.
  "id" = id fragmentu z listy powyżej, który pokrywa zdanie (obowiązkowe dla zdania pokrytego; inaczej "").
- "struktura_warunkow_ok": warunki łączne ("oraz", "i") zostają łączne, alternatywne ("albo", "lub") alternatywne.
- "negacje_ok": zdanie zachowuje negacje i wyłączenia przepisu.
- "liczby_przypisane_ok": każda liczba, wiek, okres i kwota jest przypisana do właściwej grupy osób i warunku.
- "zakres_podmiotowy_ok": zdanie dotyczy tych samych osób i sytuacji co przepis.
- "odeslania_ok": zdanie wiernie oddaje odesłania („ust. 1–3”, „te przepisy”) i warunki zastosowania przepisu; false, gdy
  zmienia adresata odesłania albo pomija warunek zastosowania.
- "potrzebne_do_pytania": false, gdy zdanie dotyczy innego zagadnienia niż pytanie i jest zbędne do odpowiedzi.
Zdanie bez warunków, negacji, liczb albo odesłań: odpowiednie pole = true.

Zwróć JSON: {{"id": "...", "czesc_pytania": "...", "odeslanie_w_zrodle": "", "pokryte": true, "struktura_warunkow_ok": true, "negacje_ok": true, "liczby_przypisane_ok": true, "zakres_podmiotowy_ok": true, "odeslania_ok": true, "potrzebne_do_pytania": true}}
"""

PROMPT_NAPRAWA = """

Poprzednia wersja odpowiedzi nie przeszła kontroli.

Poprzednia odpowiedź:
{poprzednia}

Błędy wykazane przez kontrolę (z numerem zdania i źródłem):
{bledy}

Napisz odpowiedź od nowa zgodnie z zasadami powyżej, usuwając te błędy: odpowiedz najpierw na główną wątpliwość
(przy pytaniu „czy” – wprost), stosuj przepis właściwy dla opisanej sytuacji, nie przenoś warunków między odrębnymi
podstawami prawnymi i zachowaj spójniki, negacje oraz progi. Zdania, które były poprawne, zachowaj; nie dodawaj faktów
spoza fragmentów.
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
    przeglad: bool = False                      # odpowiedź przeglądowa (hasło albo jawna prośba o przegląd)
    dopytanie: str = ""                         # pytanie doprecyzowujące o dane NIEZBĘDNE (z analizy), obok odpowiedzi
    dane_do_pelnej_oceny: list[dict] = field(default_factory=list)  # [{"dana", "dlaczego"}]: potrzebne do pełnej oceny sprawy, nie do odpowiedzi
    status_weryfikacji: str = ""                # verified | unverified | insufficient_sources ("" = nie dotyczy, np. retrieval-only)
    ostrzezenia: list[dict] = field(default_factory=list)  # [{"kod", "tekst"}]: ograniczenia i uwagi do odpowiedzi
    naprawa: bool = False                       # odpowiedź pochodzi z cyklu naprawczego (druga wersja po liście błędów)
    slad: dict = field(default_factory=dict)    # pełny ślad diagnostyczny: trace_id, analiza, ranking, kontekst, surowe wyjścia, werdykty


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


# zdanie otwarte zwrotem wskazującym („Przepisy te…”, „Powyższy przepis…”) nie jest samodzielne: w odpowiedzi złożonej
# z cytowanych przepisów łatwo wskazuje inny przepis niż ten, do którego odsyła źródło (np. „ust. 1–3” tego samego artykułu)
_RE_ZWROT_WSKAZUJACY = re.compile(
    r"^\s*(?:(?:przepis(?:y|u|ów)?|ustęp(?:y|u|ów)?|artyku(?:ł|ły|łu|łów))\s+(?:te|ten|tego|tych|powyższe|powyższy|powyższego|powyższych)"
    r"|(?:te|ten|tego|tych|powyższe|powyższy|powyższego|powyższych)\s+(?:przepis(?:y|u|ów)?|ustęp(?:y|u|ów)?|artyku(?:ł|ły|łu|łów)))\b",
    re.I)


def zaczyna_sie_zwrotem_wskazujacym(zdanie: str) -> bool:
    return bool(_RE_ZWROT_WSKAZUJACY.match(zdanie))


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
                     model=model, max_tokens=700, schema=SCHEMA_ANALIZA, etykieta="analiza")


def znormalizuj_analize(a: dict) -> tuple[list[str], list[dict], list[dict], str]:
    """Z wyniku analizy: (warianty, dane niezbędne, dane do pełnej oceny, pytanie doprecyzowujące).
    Brak danych bez uzasadnienia („dlaczego”) jest odrzucany; pytanie doprecyzowujące zostaje tylko wtedy, gdy jest
    dana niezbędna. Akceptowany jest też stary format (lista nazw w „brakujace_parametry”) – dane są wtedy niezbędne."""
    warianty = [w.strip() for w in a.get("warianty", []) if isinstance(w, str) and w.strip()][:5]
    surowe = a.get("brakujace_dane")
    if surowe is None:
        surowe = a.get("brakujace_parametry", [])
    niezbedne: list[dict] = []
    do_oceny: list[dict] = []
    for d in surowe if isinstance(surowe, list) else []:
        if isinstance(d, str) and d.strip():
            niezbedne.append({"dana": d.strip(), "dlaczego": ""})
        elif isinstance(d, dict) and str(d.get("dana", "")).strip() and str(d.get("dlaczego", "")).strip():
            wpis = {"dana": str(d["dana"]).strip(), "dlaczego": str(d["dlaczego"]).strip()}
            (do_oceny if d.get("zakres") == "do_pelnej_oceny" else niezbedne).append(wpis)
    pyt = (a.get("pytanie_doprecyzowujace") or "").strip() if niezbedne else ""
    return warianty, niezbedne, do_oceny, pyt


def generuj(pytanie: str, kontekst: list[Jednostka], przeglad: bool = False,
            naprawa: Optional[dict] = None) -> dict:
    """Odpowiedź z id cytatów; `przeglad` dokleja polecenie przeglądowe (hasło, prośba o przegląd), `naprawa`
    ({"poprzednia": tekst, "bledy": [...]}) dokleja poprzednią wersję i listę błędów z kontroli."""
    tresc = PROMPT_ODPOWIEDZ.format(pytanie=pytanie, fragmenty=_fragmenty_do_promptu(kontekst))
    if przeglad:
        tresc += PROMPT_PRZEGLAD
    if naprawa:
        tresc += PROMPT_NAPRAWA.format(poprzednia=naprawa.get("poprzednia", ""),
                                       bledy="\n".join(f"- {b}" for b in naprawa.get("bledy", [])) or "- (brak listy)")
    return czat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": tresc}],
                     max_tokens=1600 if (przeglad or naprawa) else 1200, schema=SCHEMA_ODPOWIEDZ,
                     etykieta="naprawa" if naprawa else "odpowiedz")


def _schema_weryfikacji(n: int) -> dict:
    """Schemat z wymuszoną liczbą werdyktów – bez tego małe modele oceniają tylko pierwsze zdanie."""
    sch = copy.deepcopy(SCHEMA_WERYFIKACJA)
    sch["properties"]["zdania"]["minItems"] = n
    sch["properties"]["zdania"]["maxItems"] = n
    return sch


def _zmapuj_werdykty(wpisy: list[dict], n: int) -> dict[int, dict]:
    """Werdykty po numerze zdania („nr”), nie po kolejności tablicy: numer musi mieścić się w 1..n i nie może się
    powtórzyć (przy powtórzeniu wygrywa pierwszy wpis, a zdania bez werdyktu są dopytywane osobno).
    Kolejność służy tylko wtedy, gdy ŻADEN wpis nie ma poprawnego numeru, a wpisów jest tyle co zdań."""
    po_numerach: dict[int, dict] = {}
    for w in wpisy:
        try:
            nr = int(w["nr"])
        except (KeyError, ValueError, TypeError):
            continue
        if 1 <= nr <= n and nr not in po_numerach:
            po_numerach[nr] = w
    if not po_numerach and len(wpisy) == n:
        return {i + 1: w for i, w in enumerate(wpisy)}
    return po_numerach


def _powod_usuniecia(w: dict, dozwolone: set[str]) -> Optional[str]:
    """Czy zdanie wypada wg werdyktu: brak pokrycia, źródło spoza listy fragmentów albo jedna z jawnych kontroli
    (struktura warunków, negacje, przypisanie liczb, zakres podmiotowy, odesłania, potrzeba w odpowiedzi) na „false”."""
    if not w.get("pokryte"):
        return "bez pokrycia wg weryfikacji"
    if str(w.get("id") or "").strip() not in dozwolone:
        return "weryfikator nie wskazał poprawnego źródła zdania"
    if w.get("struktura_warunkow_ok") is False:
        return "zmieniona struktura warunków (łączne / alternatywne)"
    if w.get("negacje_ok") is False:
        return "zgubiona lub zmieniona negacja"
    if w.get("liczby_przypisane_ok") is False:
        return "liczba przypisana do złej grupy osób albo złego warunku"
    if w.get("zakres_podmiotowy_ok") is False:
        return "zmieniony zakres podmiotowy"
    if w.get("odeslania_ok") is False:
        return "zmienione odesłanie albo pominięty warunek zastosowania przepisu"
    if w.get("potrzebne_do_pytania") is False or ("czesc_pytania" in w and not str(w["czesc_pytania"] or "").strip()):
        return "zdanie zbędne do odpowiedzi na pytanie (inne zagadnienie)"
    return None


def _normalizuj_bledy(surowe, n: int) -> list[dict]:
    out: list[dict] = []
    for b in surowe if isinstance(surowe, list) else []:
        if isinstance(b, str) and b.strip():
            out.append({"zdanie": 0, "typ": "inne", "opis": b.strip(), "zrodlo": ""})
        elif isinstance(b, dict) and str(b.get("opis", "")).strip():
            try:
                nr = int(b.get("zdanie", 0))
            except (ValueError, TypeError):
                nr = 0
            out.append({"zdanie": nr if 0 <= nr <= n else 0, "typ": str(b.get("typ") or "inne"),
                        "opis": str(b["opis"]).strip(), "zrodlo": str(b.get("zrodlo") or "").strip()})
    return out


def _flaga(wynik: dict, klucz: str, domyslnie: bool) -> bool:
    v = wynik.get(klucz)
    return domyslnie if v is None else bool(v)


def weryfikuj(zdania: list[str], kontekst: list[Jednostka], pytanie: str = "", dopytanie: str = "") -> dict:
    """Osobny przebieg LLM, który dostaje ORYGINALNE pytanie użytkownika (bez dopisków o trybie odpowiedzi).
    Werdykt dla każdego zdania mapowany po numerze zdania: pokrycie w cytowanym fragmencie (id musi być z listy),
    struktura warunków, negacje, przypisanie liczb, zakres podmiotowy. Dla całości: czy dotyczy pytania, czy
    rozstrzyga główną wątpliwość, czy nie łączy odrębnych podstaw, czy dopytanie jest konieczne, istotne pominięcia
    i lista błędów (z numerem zdania i źródłem). Zdania bez poprawnego werdyktu są dopytywane osobno.
    Zwraca słownik: zostaja, usuniete [(zdanie, powód)], odpowiada_na_pytanie, glowna, bez_laczenia, dopytanie_zbedne,
    pominiecia, bledy, surowe, fragmenty."""
    if not zdania or not kontekst:
        return dict(zostaja=[], usuniete=[(z, "brak kontekstu do weryfikacji") for z in zdania],
                    odpowiada_na_pytanie=True, glowna=True, bez_laczenia=True, dopytanie_zbedne=False,
                    pominiecia=[], bledy=[], surowe=None, fragmenty=[j.id for j in kontekst])
    model = KONFIG.llm_model_maly or None
    dozwolone = {j.id for j in kontekst}
    fragmenty = _fragmenty_do_promptu(kontekst)
    numerowane = "\n".join(f"{i + 1}. {z}" for i, z in enumerate(zdania))
    wynik = czat_json([{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": PROMPT_WERYFIKACJA.format(
                           pytanie=pytanie or "(brak)", dopytanie=dopytanie or "(brak)",
                           fragmenty=fragmenty, zdania=numerowane)}],
                      model=model, max_tokens=1500, schema=_schema_weryfikacji(len(zdania)), etykieta="weryfikacja")

    def pytaj_o_zdanie(z: str) -> dict:
        return czat_json([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": PROMPT_WERYFIKACJA_ZDANIE.format(
                              pytanie=pytanie or "(brak)", fragmenty=fragmenty, zdanie=z)}],
                         model=model, max_tokens=300, schema=SCHEMA_WERYFIKACJA_ZDANIE, etykieta="weryfikacja_zdania")

    wpisy = wynik.get("zdania", [])
    wpisy = [w for w in wpisy if isinstance(w, dict)] if isinstance(wpisy, list) else []
    werdykty = _zmapuj_werdykty(wpisy, len(zdania))
    for i, z in enumerate(zdania, start=1):
        w = werdykty.get(i)
        if w is None:
            log.info("Weryfikacja: brak werdyktu dla zdania %d – pytam osobno.", i)
            w = pytaj_o_zdanie(z)
        elif w.get("pokryte") and str(w.get("id") or "").strip() not in dozwolone:
            log.info("Weryfikacja: werdykt zdania %d bez poprawnego źródła – pytam osobno.", i)
            w = pytaj_o_zdanie(z)
        werdykty[i] = w

    bledy = _normalizuj_bledy(wynik.get("bledy"), len(zdania))
    powody: dict[int, str] = {}
    for i in range(1, len(zdania) + 1):
        p = _powod_usuniecia(werdykty[i], dozwolone)
        if p:
            powody[i] = p
    for b in bledy:                                  # błąd z numerem zdania i typem „zdaniowym” też usuwa zdanie
        if b["typ"] in TYPY_BLEDOW_ZDANIA and 1 <= b["zdanie"] <= len(zdania) and b["zdanie"] not in powody:
            powody[b["zdanie"]] = f"{b['typ']}: {b['opis']}"
    zostaja = [z for i, z in enumerate(zdania, start=1) if i not in powody]
    usuniete = [(z, powody[i]) for i, z in enumerate(zdania, start=1) if i in powody]
    pyta = _flaga(wynik, "odpowiedz_deklaruje_brak_danych", False) or bool(dopytanie.strip())
    pominiecia = wynik.get("istotne_pominiecia") or []
    return dict(zostaja=zostaja, usuniete=usuniete,
                odpowiada_na_pytanie=_flaga(wynik, "odpowiada_na_pytanie", True),
                glowna=_flaga(wynik, "odpowiada_na_glowna_watpliwosc", True),
                bez_laczenia=_flaga(wynik, "nie_laczy_odrebnych_podstaw", True),
                dopytanie_zbedne=pyta and not _flaga(wynik, "dopytanie_jest_konieczne", True),
                pominiecia=[p.strip() for p in pominiecia if isinstance(p, str) and p.strip()]
                if isinstance(pominiecia, list) else [],
                bledy=bledy, surowe=wynik, werdykty=[werdykty[i] for i in sorted(werdykty)],
                fragmenty=[j.id for j in kontekst])


def _z_rodzicami(indeks: Indeks, jednostki: list[Jednostka]) -> list[Jednostka]:
    out: list[Jednostka] = []
    widziane: set[str] = set()
    for j in jednostki:
        for ident in (j.id, j.rodzic):
            if ident and ident not in widziane and ident in indeks.jednostki:
                widziane.add(ident)
                out.append(indeks.jednostki[ident])
    return out


# --- kontrola jednej wersji odpowiedzi: bramki deterministyczne + weryfikacja LLM ----------
def _wynik_generatora(indeks: Indeks, g: dict, kontekst: list[Jednostka]) -> tuple[str, list[Jednostka], str]:
    tekst = (g.get("odpowiedz") or "").strip()
    surowe = g.get("cytowane_id") or []
    cyt_id = odzyskaj_cytaty(tekst, surowe if isinstance(surowe, list) else [surowe], kontekst)
    return tekst, [indeks.jednostki[c] for c in cyt_id], (g.get("uwagi") or "").strip()


def _opis_bledu(b: dict) -> str:
    zrodlo = f" (źródło: {b['zrodlo']})" if b["zrodlo"] else ""
    if b["zdanie"]:
        return f"zdanie {b['zdanie']} [{b['typ']}]: {b['opis']}{zrodlo}"
    return f"[{b['typ']}]: {b['opis']}{zrodlo}"


def _kontrola(indeks: Indeks, pytanie: str, tekst: str, cytaty: list[Jednostka], kontekst: list[Jednostka],
              weryfikacja: bool, dopytanie: str = "") -> dict:
    """Bramki deterministyczne (echo pytania, liczby spoza kontekstu – to warunek konieczny, nie dowód poprawności)
    i weryfikacja LLM jednego tekstu. Gdy weryfikator usunął zdania, to, co zostało, jest weryfikowane jeszcze raz:
    odpowiedź skrócona o kluczowe zdanie nie może przejść jako poprawna, jeśli przestała odpowiadać na pytanie."""
    usuniete: list[str] = []
    zdania: list[str] = []
    for z in _zdania(tekst):
        if powtarza_pytanie(z, pytanie):
            usuniete.append(f"{z}  [powtórzenie pytania]")
        elif zaczyna_sie_zwrotem_wskazujacym(z):
            usuniete.append(f"{z}  [zwrot wskazujący („przepisy te”) bez jednoznacznego odesłania]")
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
    k: dict = {"zdania_wejscie": list(zdania), "zdania": zdania, "usuniete": usuniete, "usuniete_weryfikacja": 0,
               "weryfikacja": "pominieta", "na_temat": True, "glowna": True, "bez_laczenia": True,
               "dopytanie_zbedne": False, "pominiecia": [], "bledy": [], "surowe": [], "fragmenty": [],
               "po_usunieciach": False}
    if weryfikacja and zdania:
        fragmenty = _z_rodzicami(indeks, cytaty)
        try:
            w = weryfikuj(zdania, fragmenty, pytanie, dopytanie)
            glob, usun, zostaja_w = w, list(w["usuniete"]), w["zostaja"]
            surowe, bledy = [w["surowe"]], list(w["bledy"])
            if usun and zostaja_w:                       # spójność tego, co zostało po usunięciach
                w2 = weryfikuj(zostaja_w, fragmenty, pytanie, dopytanie)
                glob, zostaja_w = w2, w2["zostaja"]
                usun += w2["usuniete"]
                surowe.append(w2["surowe"])
                bledy += w2["bledy"]
                k["po_usunieciach"] = True
            k.update(zdania=zostaja_w, weryfikacja="ok", na_temat=glob["odpowiada_na_pytanie"], glowna=glob["glowna"],
                     bez_laczenia=glob["bez_laczenia"], dopytanie_zbedne=glob["dopytanie_zbedne"],
                     pominiecia=glob["pominiecia"], bledy=bledy, surowe=surowe, fragmenty=glob["fragmenty"],
                     usuniete_weryfikacja=len(usun))
            k["usuniete"] = usuniete + [f"{z}  [{powod}]" for z, powod in usun]
        except (BrakLLM, ValueError) as e:
            log.warning("Weryfikacja nieudana (%s) – odpowiedź oznaczona jako niezweryfikowana.", e)
            k.update(weryfikacja="awaria", blad=str(e))
    powody = [_opis_bledu(b) for b in k["bledy"]]
    powody += [f"istotne pominięcie: {p}" for p in k["pominiecia"]]
    if not k["na_temat"]:
        powody.append("odpowiedź dotyczy innej kwestii niż pytanie")
    if not k["glowna"]:
        powody.append("odpowiedź nie rozstrzyga głównej wątpliwości (przy pytaniu „czy” – wprost: tak / nie / "
                      "zależy od wskazanego warunku)")
    if not k["bez_laczenia"]:
        powody.append("odpowiedź przedstawia warunek odrębnej podstawy prawnej jako warunek omawianej")
    powody += [f"zdanie usunięte przez kontrolę: {u}" for u in k["usuniete"]]
    k["powody"] = list(dict.fromkeys(powody))
    k["potrzebna_naprawa"] = k["weryfikacja"] == "ok" and (
        not k["na_temat"] or not k["glowna"] or not k["bez_laczenia"] or bool(k["pominiecia"])
        or k["usuniete_weryfikacja"] > 0 or not k["zdania"])
    k["wynik_kontroli"] = ((0 if k["na_temat"] else 10) + (0 if k["glowna"] else 6) + (0 if k["bez_laczenia"] else 6)
                           + 3 * len(k["pominiecia"]) + len(k["usuniete"]) + (8 if not k["zdania"] else 0))
    return k


_RE_SPOJNIK_NA_POCZATKU = re.compile(
    r"^(?:Jednakże|Jednak|Ponadto|Dodatkowo|Natomiast|Zatem|Również|Poza tym|Co więcej|Dlatego|W związku z tym)\b[,\s]+")


def _bez_wiszacego_spojnika(zdania: list[str], oryginalne: list[str]) -> list[str]:
    """Gdy usunięto pierwsze zdanie odpowiedzi, następne nie może zaczynać się od „Jednakże…”, „Ponadto…” itp.;
    zdejmujemy sam spójnik (nie dodajemy żadnej treści)."""
    if not zdania or not oryginalne or zdania[0] == oryginalne[0]:
        return zdania
    nowe = _RE_SPOJNIK_NA_POCZATKU.sub("", zdania[0], count=1)
    return [nowe[:1].upper() + nowe[1:]] + zdania[1:] if nowe != zdania[0] else zdania


def _status_koncowy(k: dict) -> tuple[str, list[dict]]:
    """verified tylko, gdy weryfikator zadziałał i nie zostały żadne zastrzeżenia; awaria, wyłączona weryfikacja
    albo zastrzeżenia po (ewentualnej) naprawie dają unverified z jawnym ograniczeniem."""
    ostrz: list[dict] = []
    if k["weryfikacja"] == "awaria":
        ostrz.append({"kod": "weryfikacja_nieudana",
                      "tekst": "Kontrola odpowiedzi nie zadziałała (" + k.get("blad", "błąd modelu") + "). To szkic do "
                               "sprawdzenia przez pracownika: czytaj go wyłącznie razem z cytowanymi przepisami."})
        return "unverified", ostrz
    if k["weryfikacja"] == "pominieta":
        ostrz.append({"kod": "weryfikacja_wylaczona", "tekst": "Weryfikacja odpowiedzi była wyłączona."})
        return "unverified", ostrz
    if not k["glowna"]:
        ostrz.append({"kod": "nie_rozstrzyga",
                      "tekst": "Odpowiedź nie rozstrzyga głównej wątpliwości wprost. Sprawdź cytaty albo przekaż "
                               "sprawę referentowi."})
    if not k["bez_laczenia"]:
        ostrz.append({"kod": "laczy_odrebne_podstawy",
                      "tekst": "Kontrola podejrzewa, że odpowiedź przenosi warunek odrębnej podstawy prawnej na "
                               "omawianą. Sprawdź cytaty albo przekaż sprawę referentowi."})
    if k["pominiecia"]:
        ostrz.append({"kod": "istotne_pominiecia",
                      "tekst": "Kontrola wskazuje możliwe pominięcia istotnych warunków: " + "; ".join(k["pominiecia"])})
    return ("unverified" if ostrz else "verified"), ostrz


# --- wersje aktów: co system wie o brzmieniu przepisów ------------------------------------
def _wersje_aktow(indeks: Indeks) -> dict[str, Jednostka]:
    """Pierwsza jednostka każdego aktu (daty i etykieta wersji są wspólne dla całego aktu)."""
    akty: dict[str, Jednostka] = {}
    for j in indeks.jednostki.values():
        akty.setdefault(j.akt_kod, j)
    return akty


def _opis_wersji(j: Jednostka) -> str:
    return (f"{j.akt}: wersja {j.wersja or 'bez oznaczenia'}, obowiązuje od {j.obowiazuje_od or 'brak daty'} "
            f"do {j.obowiazuje_do or 'brak daty końca'}")


def _brak_wersji_na_date(indeks: Indeks, na_dzien: str) -> Optional[str]:
    """Komunikat, gdy żaden akt w indeksie nie obowiązuje w podanym dniu (wszystkie mają daty i żaden ich nie obejmuje)."""
    akty = _wersje_aktow(indeks)
    if not akty or any(j.obowiazuje(na_dzien) for j in akty.values()):
        return None
    return (f"W indeksie nie ma wersji aktu obowiązującej na dzień {na_dzien}; system nie potwierdza brzmienia "
            "przepisów z tego dnia. Dostępne wersje: " + "; ".join(_opis_wersji(j) for j in akty.values()) + ".")


def _ostrzezenia_wersji(cytaty: list[Jednostka], na_dzien: Optional[str], data_wskazana: bool) -> list[dict]:
    """Ostrzeżenia o stanie prawnym, gdy użytkownik wskazał datę: system ma jedno brzmienie aktu, więc nie weryfikuje
    historii zmian (daty obowiązywania nie są wnioskowane z domysłów ani z daty pobrania)."""
    if not (data_wskazana and na_dzien):
        return []
    out: list[dict] = []
    for kod in dict.fromkeys(j.akt_kod for j in cytaty):
        j = next(x for x in cytaty if x.akt_kod == kod)
        if not j.obowiazuje_od and not j.obowiazuje_do:
            out.append({"kod": "brak_wersji_historycznej",
                        "tekst": f"Akt „{j.akt}” jest w indeksie w jednej wersji ({j.wersja or 'bez oznaczenia'}) "
                                 f"bez dat obowiązywania przepisów; system nie potwierdza jego brzmienia na dzień "
                                 f"{na_dzien}."})
        else:
            out.append({"kod": "jedna_wersja_aktu",
                        "tekst": f"{_opis_wersji(j)}. W indeksie nie ma wcześniejszych brzmień, więc system nie "
                                 f"weryfikuje historii zmian przepisów przed dniem {na_dzien}."})
    return out


def _slad_kontroli(k: dict) -> dict:
    return {kk: k.get(kk) for kk in ("zdania_wejscie", "zdania", "usuniete", "usuniete_weryfikacja", "weryfikacja",
                                     "na_temat", "glowna", "bez_laczenia", "dopytanie_zbedne", "pominiecia", "bledy",
                                     "surowe", "fragmenty", "po_usunieciach", "powody", "blad", "wynik_kontroli",
                                     "potrzebna_naprawa")}


def _slad_trafien(traf: list[Trafienie]) -> list[dict]:
    return [{"id": t.id, "wynik": round(float(t.wynik), 4), "zrodlo": t.zrodlo} for t in traf]


# --- pełny przebieg -----------------------------------------------------------------------
def odpowiedz(indeks: Indeks, pytanie: str, na_dzien: Optional[str] = None,
              uzyj_emb: bool = True, uzyj_reranker: bool = True,
              dopytuj: bool = True, weryfikacja: bool = True,
              top_n: Optional[int] = None, uzyj_llm: bool = True,
              analiza: Optional[bool] = None, naprawa: Optional[bool] = None,
              data_wskazana: Optional[bool] = None) -> Odpowiedz:
    """Pełny przebieg. Bez skonfigurowanego LLM zwraca same trafienia (retrieval-only).
    `data_wskazana` – czy datę „na dzień” podał użytkownik (domyślnie: tak, gdy `na_dzien` jest ustawione); tylko wtedy
    odpowiedź dostaje ostrzeżenia o stanie prawnym. Pole `slad` wyniku ma komplet danych diagnostycznych (trace_id,
    analiza, ranking, kontekst, wszystkie wiadomości do modelu i surowe odpowiedzi, werdykty, naprawa, wynik, konfiguracja
    bez sekretów); zapisuje je dopiero warstwa wywołująca, gdy diagnostyka jest włączona."""
    czasy: dict[str, float] = {}
    if data_wskazana is None:
        data_wskazana = na_dzien is not None
    slad: dict = {"trace_id": uuid.uuid4().hex, "czas": datetime.now().isoformat(timespec="seconds"),
                  "pytanie": pytanie, "na_dzien": na_dzien, "data_wskazana": data_wskazana,
                  "konfiguracja": konfiguracja_do_sladu()}
    t0 = time.time()
    if not uzyj_llm or not llm_dostepny():
        traf = szukaj_pelnie(indeks, pytanie, na_dzien=na_dzien, uzyj_emb=uzyj_emb,
                             uzyj_reranker=uzyj_reranker, top_n=top_n)
        kontekst = rozszerz(indeks, traf, na_dzien=na_dzien)
        czasy["szukanie"] = round(time.time() - t0, 1)
        slad.update(zapytania=[pytanie], trafienia=_slad_trafien(traf), kontekst=[j.id for j in kontekst],
                    wynik={"tryb": "retrieval-only", "czasy": czasy})
        return Odpowiedz(tryb="retrieval-only", trafienia=traf, kontekst=kontekst, czasy=czasy, slad=slad)
    rozpocznij_slad()
    try:
        w = _odpowiedz_llm(indeks, pytanie, na_dzien, data_wskazana, uzyj_emb, uzyj_reranker, dopytuj, weryfikacja,
                           top_n, analiza, naprawa, czasy, slad, t0)
        slad["wynik"] = {"tryb": w.tryb, "status_weryfikacji": w.status_weryfikacji, "tekst": w.tekst, "uwagi": w.uwagi,
                         "cytaty": [j.id for j in w.cytaty], "zdania_usuniete": w.zdania_usuniete,
                         "ostrzezenia": w.ostrzezenia, "naprawa": w.naprawa, "dopytanie": w.dopytanie,
                         "brakujace_parametry": w.brakujace_parametry, "dane_do_pelnej_oceny": w.dane_do_pelnej_oceny,
                         "czasy": w.czasy}
        return w
    finally:
        slad["wywolania"] = zakoncz_slad()


def _odpowiedz_llm(indeks: Indeks, pytanie: str, na_dzien: Optional[str], data_wskazana: bool, uzyj_emb: bool,
                   uzyj_reranker: bool, dopytuj: bool, weryfikacja: bool, top_n: Optional[int],
                   analiza: Optional[bool], naprawa: Optional[bool], czasy: dict, slad: dict, t0: float) -> Odpowiedz:
    # krok 1 – analiza pytania i warianty (pomijalny: --bez-analizy / LEXRAG_SKIP_ANALYSIS=1).
    # Tryb przeglądowy włączają WYŁĄCZNIE hasło i jawna prośba o przegląd; brak danych wg analizy daje zwykłą odpowiedź
    # (warunkową) z pytaniem doprecyzowującym obok niej, a dane „do pełnej oceny” tylko jako uwaga.
    warianty: list[str] = []
    niezbedne: list[dict] = []
    do_oceny: list[dict] = []
    pyt_dop = ""
    przeglad = tryb_przegladowy(pytanie)
    if analiza is None:
        analiza = not KONFIG.pomin_analize
    if naprawa is None:
        naprawa = KONFIG.naprawa
    if analiza:
        try:
            a = analizuj_pytanie(pytanie)
            warianty, niezbedne, do_oceny, pyt_dop = znormalizuj_analize(a)
            slad["analiza"] = a
        except (BrakLLM, ValueError) as e:
            log.warning("Analiza pytania nieudana (%s) – szukam samym pytaniem.", e)
            slad["analiza"] = {"blad": str(e)}
        czasy["analiza"] = round(time.time() - t0, 1)
    brakujace = [d["dana"] for d in niezbedne]
    slad.update(analiza_wykonana=bool(analiza), pomin_analize_efektywne=not analiza)   # efektywne, nie domyślne z konfiguracji

    # kroki 2–4 – wyszukiwanie i kontekst
    t1 = time.time()
    traf = szukaj_pelnie(indeks, pytanie, warianty, na_dzien=na_dzien,
                         uzyj_emb=uzyj_emb, uzyj_reranker=uzyj_reranker, top_n=top_n)
    kontekst = rozszerz(indeks, traf, na_dzien=na_dzien)
    czasy["szukanie"] = round(time.time() - t1, 1)
    slad.update(zapytania=[pytanie] + warianty, trafienia=_slad_trafien(traf),
                kontekst=[j.id for j in kontekst], przeglad=przeglad)
    wspolne = dict(warianty=warianty, trafienia=traf, kontekst=kontekst, czasy=czasy, przeglad=przeglad,
                   brakujace_parametry=brakujace, dane_do_pelnej_oceny=do_oceny,
                   dopytanie=(pyt_dop if dopytuj else ""), slad=slad)

    def _odmowa(**kw) -> Odpowiedz:
        """Brak podstaw do odpowiedzi (insufficient_sources). Gdy analiza przygotowała dopytanie o dane niezbędne,
        oddajemy je zamiast samej odmowy (powód trafia do uwag)."""
        kw["status_weryfikacji"] = "insufficient_sources"
        if dopytuj and pyt_dop:
            kw["uwagi"] = " ".join(x for x in (kw.get("tekst", ""), kw.get("uwagi", "")) if x).strip()
            kw["tekst"] = pyt_dop
            return Odpowiedz(**kw, tryb="dopytanie")
        return Odpowiedz(**kw, tryb="odmowa")

    if not kontekst:
        brak_daty = _brak_wersji_na_date(indeks, na_dzien) if na_dzien else None
        if brak_daty:
            komunikat = brak_daty if data_wskazana else (
                f"Domyślnie sprawdzono stan prawny na dziś ({na_dzien}), ale {brak_daty[:1].lower() + brak_daty[1:]}"
            )
            return _odmowa(tekst=komunikat, ostrzezenia=[{"kod": "brak_wersji_na_date", "tekst": komunikat}], **wspolne)
        return _odmowa(tekst="Nie znaleziono przepisów pasujących do pytania.", **wspolne)
    if KONFIG.prog_rerank > 0 and traf and traf[0].zrodlo == "rerank":
        p = _sigmoid(traf[0].wynik)
        if p < KONFIG.prog_rerank:
            return _odmowa(**wspolne,
                           tekst=f"Nie znaleziono przepisu pasującego do pytania (najlepszy kandydat: "
                                 f"{p:.0%} trafności, próg {KONFIG.prog_rerank:.0%}).")

    # krok 5 – odpowiedź z id cytatów
    t2 = time.time()
    try:
        g = generuj(pytanie, kontekst, przeglad=przeglad)
    except (BrakLLM, ValueError) as e:
        return Odpowiedz(tryb="blad", tekst=f"Błąd LLM: {e}", **wspolne)
    czasy["odpowiedz"] = round(time.time() - t2, 1)
    slad["generator"] = g
    tekst, cytaty, uwagi = _wynik_generatora(indeks, g, kontekst)
    if g.get("brak_pokrycia") or not tekst:
        return _odmowa(uwagi=uwagi, **wspolne,
                       tekst="Dostarczone przepisy nie pozwalają odpowiedzieć jednoznacznie. " + uwagi)
    if not cytaty:
        return _odmowa(uwagi=uwagi, **wspolne,
                       tekst="Model nie wskazał podstawy prawnej spośród dostarczonych fragmentów – "
                             "sprawa wymaga referenta.")

    # krok 6 – bramki deterministyczne i weryfikacja (z ponowną weryfikacją tego, co zostało po usunięciach)
    t3 = time.time()
    k = _kontrola(indeks, pytanie, tekst, cytaty, kontekst, weryfikacja, pyt_dop)
    czasy["weryfikacja"] = round(time.time() - t3, 1)
    slad["kontrola"] = [_slad_kontroli(k)]

    # krok 7 – jedna próba naprawy: oryginalne pytanie, kontekst, poprzednia odpowiedź i lista błędów → nowa wersja →
    # ponowna weryfikacja; po drugim niepowodzeniu zostaje jawne ograniczenie (unverified) albo odmowa, nigdy kolejna pętla
    naprawiono = False
    if naprawa and k["potrzebna_naprawa"]:
        t4 = time.time()
        try:
            g2 = generuj(pytanie, kontekst, przeglad=przeglad, naprawa={"poprzednia": tekst, "bledy": k["powody"]})
            slad["naprawa"] = {"powody": k["powody"], "generator": g2}
            tekst2, cytaty2, uwagi2 = _wynik_generatora(indeks, g2, kontekst)
            if tekst2 and cytaty2 and not g2.get("brak_pokrycia"):
                k2 = _kontrola(indeks, pytanie, tekst2, cytaty2, kontekst, weryfikacja, pyt_dop)
                slad["kontrola"].append(_slad_kontroli(k2))
                if k2["weryfikacja"] == "ok" and k2["wynik_kontroli"] < k["wynik_kontroli"]:
                    k, cytaty, uwagi, naprawiono = k2, cytaty2, uwagi2, True
                else:
                    slad["naprawa"]["odrzucona"] = "poprawiona wersja nie wypadła lepiej w kontroli"
            else:
                slad["naprawa"]["odrzucona"] = "model nie dał poprawionej odpowiedzi z cytatami"
        except (BrakLLM, ValueError) as e:
            slad["naprawa"] = {"powody": k["powody"], "blad": str(e)}
        czasy["naprawa"] = round(time.time() - t4, 1)

    wspolne["cytaty"] = cytaty
    usuniete = k["usuniete"]
    if k["dopytanie_zbedne"]:                                    # weryfikator: dopytanie niepotrzebne – nie męczymy użytkownika
        wspolne["dopytanie"] = ""
        wspolne["brakujace_parametry"] = []                      # ta sama decyzja dla listy „brakuje: …” w odpowiedzi
        pyt_dop = ""                                             # ani w odmowie, która zastępuje się dopytaniem
    if k["weryfikacja"] == "ok" and not k["na_temat"]:
        return _odmowa(uwagi=uwagi, **wspolne,
                       zdania_usuniete=usuniete + [f"{z}  [nie na temat pytania]" for z in k["zdania"]],
                       tekst="Odpowiedź modelu dotyczyła innej kwestii niż pytanie – sprawa wymaga referenta.")
    tekst = " ".join(_bez_wiszacego_spojnika(k["zdania"], k["zdania_wejscie"])).strip()
    if not tekst:
        return _odmowa(zdania_usuniete=usuniete, uwagi=uwagi, **wspolne,
                       tekst="Żadne zdanie odpowiedzi nie znalazło pokrycia w przepisach – sprawa wymaga referenta.")
    status, ostrzezenia = _status_koncowy(k)
    ostrzezenia = _ostrzezenia_wersji(cytaty, na_dzien, data_wskazana) + ostrzezenia
    if naprawiono:
        ostrzezenia.append({"kod": "po_cyklu_naprawczym",
                            "tekst": "Pierwsza wersja odpowiedzi nie przeszła kontroli; to druga, napisana po liście błędów."})
    if status == "unverified" and k["weryfikacja"] == "awaria":
        uwagi = (uwagi + " NIEZWERYFIKOWANA: " + k.get("blad", "")).strip()
    return Odpowiedz(tryb="odpowiedz", tekst=tekst, zdania_usuniete=usuniete, uwagi=uwagi,
                     status_weryfikacji=status, ostrzezenia=ostrzezenia, naprawa=naprawiono, **wspolne)
