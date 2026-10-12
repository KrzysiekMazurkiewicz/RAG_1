"""Testy z PRAWDZIWYM modelem. Nie wchodzą w zwykłe `pytest`: wymagają serwera modelu i zużywają jego czas.

Uruchomienie (model on-premise albo tunel do serwera w Colabie, tylko akty publiczne):

    LEXRAG_TEST_MODEL=1 LEXRAG_LLM_BASE_URL=http://localhost:11434/v1 LEXRAG_LLM_API=ollama \\
    LEXRAG_LLM_MODEL=gemma4:12b python -m pytest tests/test_model_rzeczywisty.py -v -s

Sprawdzają to, czego mocki nie mogą: czy prawdziwy weryfikator odrzuca celowo błędne zdania (D, E, B) i nie odrzuca
poprawnego (kontrola), oraz czy cała ścieżka z prawdziwym generatorem trzyma niezmienniki (A, C). Zdania do oceny
i oczekiwane twierdzenia pochodzą z tests/data/fus_wyciag.txt, a nie z pamięci. Wynik zależy od modelu: porażka testu
oznacza, że ten model nie wykrył danego błędu, a nie błąd w kodzie."""
import os

import pytest

from lexrag import generate
from pomoce import WERSJA_FUS, zbuduj_indeks_fus

pytestmark = pytest.mark.skipif(
    not (os.environ.get("LEXRAG_TEST_MODEL") and os.environ.get("LEXRAG_LLM_MODEL")),
    reason="test z prawdziwym modelem: ustaw LEXRAG_TEST_MODEL=1, LEXRAG_LLM_MODEL i LEXRAG_LLM_BASE_URL")

PYTANIE_MAMA = ("Dzień dobry, pytam o mamę. Urodziła się w 1948 roku. Jak skończyła 55 lat, miała już 30 lat okresów "
                "składkowych i nieskładkowych. Przez cały ten czas była zatrudniona na umowie o pracę. Czy mogła wtedy "
                "przejść na wcześniejszą emeryturę, mimo że była zdrowa i nie miała orzeczenia o niezdolności do pracy?")
PYTANIE_25_LAT = ("Moja mama urodziła się w 1948 roku, ma 25 lat okresów składkowych i nieskładkowych, była zatrudniona "
                  "na umowę o pracę i nie ma orzeczenia o niezdolności do pracy. Czy mogła przejść na emeryturę po "
                  "ukończeniu 55 lat?")
POPRAWNE = ("Kobieta urodzona przed dniem 1 stycznia 1949 r. może przejść na emeryturę po osiągnięciu wieku 55 lat, jeżeli "
            "ma co najmniej 30-letni okres składkowy i nieskładkowy (art. 29 ust. 1 pkt 1).")
ALBO_NA_I = ("Kobieta może przejść na emeryturę po osiągnięciu wieku 55 lat, jeżeli ma co najmniej 30-letni okres składkowy "
             "i nieskładkowy i ma co najmniej 20-letni okres składkowy i nieskładkowy oraz została uznana za całkowicie "
             "niezdolną do pracy (art. 29 ust. 1 pkt 1).")
ODREBNA_PODSTAWA = ("Warunkiem przejścia na emeryturę jest dodatkowo prawo do renty z tytułu niezdolności do pracy "
                    "spowodowanej wypadkiem przy pracy (art. 31).")
STAZ_25_NA_WARIANT_30 = ("Mama mogła przejść na emeryturę po osiągnięciu wieku 55 lat, bo spełniła wariant z co najmniej "
                         "30-letnim okresem składkowym i nieskładkowym (art. 29 ust. 1 pkt 1).")


@pytest.fixture(scope="module")
def indeks(tmp_path_factory):
    return zbuduj_indeks_fus(tmp_path_factory.mktemp("fus_model"))


def fragmenty(indeks, *ids):
    return [indeks.jednostki[i] for i in ids]


def sprawdz(indeks, pytanie, *zdania):
    return generate.weryfikuj(list(zdania), fragmenty(indeks, "fus:art29:ust1", "fus:art29", "fus:art31"), pytanie)


def test_kontrola_poprawne_zdanie_przechodzi(indeks):
    w = sprawdz(indeks, PYTANIE_MAMA, POPRAWNE)
    print("\nPOPRAWNE:", w["usuniete"], w["glowna"], w["bledy"])
    assert w["zostaja"] == [POPRAWNE]


def test_D_weryfikator_odrzuca_zamiane_albo_na_i(indeks):
    w = sprawdz(indeks, "Czy kobieta z 30-letnim stażem może przejść na emeryturę po 55 latach?", ALBO_NA_I)
    print("\nALBO->I:", w["usuniete"], w["bledy"])
    assert w["zostaja"] == [] and w["usuniete"]


def test_E_weryfikator_wykrywa_przeslanke_z_odrebnej_podstawy(indeks):
    w = sprawdz(indeks, PYTANIE_MAMA, POPRAWNE, ODREBNA_PODSTAWA)
    print("\nODREBNA PODSTAWA:", w["usuniete"], w["bez_laczenia"], w["bledy"])
    assert ODREBNA_PODSTAWA not in w["zostaja"]


def test_B_weryfikator_wykrywa_przeniesienie_wariantu_30_lat_na_25(indeks):
    w = sprawdz(indeks, PYTANIE_25_LAT, STAZ_25_NA_WARIANT_30)
    print("\n25 LAT NA WARIANT 30:", w["usuniete"], w["glowna"], w["bledy"])
    assert w["zostaja"] == [] and w["usuniete"]


def test_A_sciezka_z_prawdziwym_modelem_pytanie_o_mame(indeks):
    w = generate.odpowiedz(indeks, PYTANIE_MAMA, na_dzien="2010-05-01", uzyj_emb=False, uzyj_reranker=False, analiza=True)
    print(f"\nA: tryb={w.tryb} status={w.status_weryfikacji} naprawa={w.naprawa} przeglad={w.przeglad}\n  {w.tekst}\n"
          f"  cytaty={[c.id for c in w.cytaty]} dopytanie={w.dopytanie!r}\n  ostrzezenia={[o['kod'] for o in w.ostrzezenia]}\n"
          f"  do pelnej oceny={w.dane_do_pelnej_oceny}\n  usuniete={w.zdania_usuniete}")
    assert not w.przeglad and w.tryb in ("odpowiedz", "dopytanie", "odmowa")
    assert w.status_weryfikacji in ("verified", "unverified", "insufficient_sources")
    if w.tryb == "odpowiedz":
        assert any(o["kod"] == "brak_wersji_historycznej" and WERSJA_FUS in o["tekst"] for o in w.ostrzezenia)


def test_B_sciezka_z_prawdziwym_modelem_staz_25_lat(indeks):
    w = generate.odpowiedz(indeks, PYTANIE_25_LAT, uzyj_emb=False, uzyj_reranker=False, analiza=True)
    print(f"\nB: tryb={w.tryb} status={w.status_weryfikacji} naprawa={w.naprawa}\n  {w.tekst}\n  usuniete={w.zdania_usuniete}")
    assert w.status_weryfikacji in ("verified", "unverified", "insufficient_sources")


def test_C_sciezka_z_prawdziwym_modelem_pytanie_ogolne(indeks):
    w = generate.odpowiedz(indeks, "W jakim wieku idzie się na emeryturę?", uzyj_emb=False, uzyj_reranker=False, analiza=True)
    print(f"\nC: tryb={w.tryb} status={w.status_weryfikacji} przeglad={w.przeglad}\n  {w.tekst}\n  dopytanie={w.dopytanie!r}")
    assert not w.przeglad and w.status_weryfikacji in ("verified", "unverified", "insufficient_sources")
    assert len(generate._zdania(w.tekst)) <= 6
