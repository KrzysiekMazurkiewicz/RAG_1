"""Testy regresyjne kontroli odpowiedzi (przypadki A–H z audytu) na mockach modelu.

Model jest skryptowany, więc te testy sprawdzają ORKIESTRACJĘ: co system robi z werdyktami (usuwanie zdań, naprawa,
ponowna weryfikacja, status, ostrzeżenia, brak przeglądu przy braku danych, niezależność weryfikatora), a nie to, czy
prawdziwy model umie wykryć błąd. To sprawdzają testy z prawdziwym modelem (tests/test_model_rzeczywisty.py).
Oczekiwane twierdzenia prawne pochodzą z tests/data/fus_wyciag.txt (dosłowny tekst jednolity), nie z pamięci."""
import pytest

from lexrag import generate, llm
from lexrag.generate import (TYPY_BLEDOW, _brak_wersji_na_date, _ostrzezenia_wersji, _zmapuj_werdykty,
                             prosba_o_przeglad, tryb_przegladowy, znormalizuj_analize)
from pomoce import (WERSJA_FUS, analiza, blad, odpowiedz_modelu, skrypt_llm, werdykt, weryfikacja, zbuduj_indeks_fus)

PYTANIE_MAMA = ("Dzień dobry, pytam o mamę. Urodziła się w 1948 roku. Jak skończyła 55 lat, miała już 30 lat okresów "
                "składkowych i nieskładkowych. Przez cały ten czas była zatrudniona na umowie o pracę. Czy mogła wtedy "
                "przejść na wcześniejszą emeryturę, mimo że była zdrowa i nie miała orzeczenia o niezdolności do pracy?")
PYTANIE_25_LAT = ("Moja mama urodziła się w 1948 roku, ma 25 lat okresów składkowych i nieskładkowych, była zatrudniona "
                  "na umowę o pracę i nie ma orzeczenia o niezdolności do pracy. Czy mogła przejść na emeryturę po "
                  "ukończeniu 55 lat?")
A29, A29U1, A31 = "fus:art29", "fus:art29:ust1", "fus:art31"
POPRAWNE_30 = ("Tak, kobieta urodzona przed dniem 1 stycznia 1949 r. mogła przejść na emeryturę po osiągnięciu wieku "
               "55 lat, jeżeli miała co najmniej 30-letni okres składkowy i nieskładkowy (art. 29 ust. 1 pkt 1).")


@pytest.fixture(scope="module")
def indeks_fus(tmp_path_factory):
    return zbuduj_indeks_fus(tmp_path_factory.mktemp("fus"))                       # jedna wersja, bez dat obowiązywania


def pytaj(indeks, pytanie, **kw):
    kw.setdefault("uzyj_emb", False)
    kw.setdefault("uzyj_reranker", False)
    return generate.odpowiedz(indeks, pytanie, **kw)


def kody(w):
    return [o["kod"] for o in w.ostrzezenia]


# --- A: pytanie o mamę -------------------------------------------------------------------------
def test_A_pytanie_o_mame_odpowiedz_konkretna_bez_przegladu_z_ograniczeniem_wersji(indeks_fus, monkeypatch):
    wywolania = skrypt_llm(
        monkeypatch,
        analiza=analiza(do_oceny=[("dokładna data zgłoszenia wniosku o emeryturę",
                                   "od niej zależy ocena warunków zatrudnienia z ust. 2")]),
        odpowiedz=odpowiedz_modelu(
            POPRAWNE_30 + " Orzeczenie o całkowitej niezdolności do pracy jest potrzebne dopiero w wariancie z 20-letnim "
            "okresem składkowym i nieskładkowym (art. 29 ust. 1 pkt 1). Pełna ocena uprawnienia wymaga jeszcze "
            "sprawdzenia warunków z art. 29 ust. 2 i 3.", [A29U1, A29]),
        weryfikacja=weryfikacja([werdykt(1, A29U1), werdykt(2, A29U1), werdykt(3, A29)]))
    w = pytaj(indeks_fus, PYTANIE_MAMA, na_dzien="2010-05-01")
    assert w.tryb == "odpowiedz" and w.status_weryfikacji == "verified"
    assert not w.przeglad                                          # brak „ścisłych” danych nie włącza przeglądu
    assert A31 not in [c.id for c in w.cytaty]                      # odrębna podstawa nie jest cytowana jako warunek
    assert w.dopytanie == "" and w.brakujace_parametry == []         # nic niezbędnego nie brakuje: odpowiedź bez dopytywania
    assert w.dane_do_pelnej_oceny[0]["dana"].startswith("dokładna data")   # pełna ocena rozdzielona od odpowiedzi
    prompt_odp = next(p for e, p, _ in wywolania if e == "odpowiedz")
    assert "Uwaga: użytkownik podał samo hasło" not in prompt_odp and "Najpierw odpowiedz na główną wątpliwość" in prompt_odp
    prompt_wer = next(p for e, p, _ in wywolania if e == "weryfikacja")
    assert PYTANIE_MAMA in prompt_wer                               # weryfikator dostaje oryginalne pytanie...
    assert "[pytanie hasłowe" not in prompt_wer and "Tryb odpowiedzi" not in prompt_wer   # ...bez dopisków o trybie
    ostrz = next(o for o in w.ostrzezenia if o["kod"] == "brak_wersji_historycznej")        # jawne ograniczenie czasowe
    assert "2010-05-01" in ostrz["tekst"] and WERSJA_FUS in ostrz["tekst"]


def test_A_dane_w_pytaniu_nie_sa_ponownie_dopytywane(indeks_fus, monkeypatch):
    """Analiza bez uzasadnienia braku jest odrzucana, więc dopytanie o to, co wiadomo, nie dociera do użytkownika."""
    a = analiza(pytanie="Podaj datę urodzenia mamy.")
    a["brakujace_dane"] = [{"dana": "data urodzenia", "zakres": "niezbedna", "dlaczego": ""}]    # bez uzasadnienia
    skrypt_llm(monkeypatch, analiza=a, odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
               weryfikacja=weryfikacja([werdykt(1, A29U1)]))
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.dopytanie == "" and w.brakujace_parametry == []


# --- B: 25 lat zamiast 30 --------------------------------------------------------------------------
def test_B_krotszy_staz_nie_dostaje_wariantu_30_lat(indeks_fus, monkeypatch):
    zle = ("Tak, mogła przejść na emeryturę po osiągnięciu wieku 55 lat, bo miała co najmniej 30-letni okres "
           "składkowy i nieskładkowy (art. 29 ust. 1 pkt 1).")
    dobre = ("Nie na podstawie wariantu z 30-letnim okresem: przy 25 latach okresów składkowych i nieskładkowych art. 29 "
             "ust. 1 pkt 1 wymaga 20-letniego okresu oraz uznania za całkowicie niezdolną do pracy.")
    wywolania = skrypt_llm(
        monkeypatch, z_naprawa=True, analiza=analiza(),
        odpowiedz=odpowiedz_modelu(zle, [A29U1]),
        naprawa=odpowiedz_modelu(dobre, [A29U1]),
        weryfikacja=[
            weryfikacja([werdykt(1, A29U1, liczby_przypisane_ok=False)], odpowiada_na_glowna_watpliwosc=False,
                        bledy=[blad(1, "liczby", "30-letni okres dotyczy innego stanu faktycznego niż 25 lat z pytania", A29U1)]),
            weryfikacja([werdykt(1, A29U1)])])
    w = pytaj(indeks_fus, PYTANIE_25_LAT)
    assert w.tryb == "odpowiedz" and w.naprawa and w.status_weryfikacji == "verified"
    assert "bo miała co najmniej 30-letni" not in w.tekst and w.tekst.startswith("Nie na podstawie")
    assert [e for e, _, _ in wywolania] == ["analiza", "odpowiedz", "weryfikacja", "naprawa", "weryfikacja"]
    prompt_naprawy = wywolania[3][1]
    assert PYTANIE_25_LAT in prompt_naprawy and zle in prompt_naprawy       # pytanie, kontekst i poprzednia odpowiedź
    assert "zdanie 1 [liczby]" in prompt_naprawy and "25 lat" in prompt_naprawy        # konkretna lista błędów
    assert "po_cyklu_naprawczym" in kody(w)


# --- C: pytanie ogólne --------------------------------------------------------------------------------
def test_C_pytanie_ogolne_o_wiek_dostaje_odpowiedz_podstawowa_bez_katalogu(indeks_fus, monkeypatch):
    wywolania = skrypt_llm(
        monkeypatch, analiza=analiza(),
        odpowiedz=odpowiedz_modelu("Ubezpieczonym urodzonym po dniu 31 grudnia 1948 r. emerytura przysługuje po "
                                   "osiągnięciu wieku co najmniej 60 lat dla kobiet i 65 lat dla mężczyzn (art. 24 ust. 1).",
                                   ["fus:art24:ust1"]),
        weryfikacja=weryfikacja([werdykt(1, "fus:art24:ust1")]))
    w = pytaj(indeks_fus, "W jakim wieku idzie się na emeryturę?")
    assert w.tryb == "odpowiedz" and not w.przeglad and w.status_weryfikacji == "verified"
    prompt = next(p for e, p, _ in wywolania if e == "odpowiedz")
    assert "Uwaga: użytkownik podał samo hasło" not in prompt and "wymień wszystkie" not in prompt.lower()
    assert "Obecność fragmentu w kontekście nie oznacza obowiązku" in " ".join(prompt.split())   # kontekst nie jest listą do wyczerpania
    assert len(generate._zdania(w.tekst)) == 1


def test_C_tryb_przegladowy_tylko_dla_hasla_i_jawnej_prosby():
    assert tryb_przegladowy("wcześniejsza emerytura") and tryb_przegladowy("renta rodzinna")
    assert prosba_o_przeglad("Wymień rodzaje wcześniejszych emerytur") and tryb_przegladowy("Omów warunki renty")
    assert not tryb_przegladowy("W jakim wieku idzie się na emeryturę?")
    assert not tryb_przegladowy(PYTANIE_MAMA)
    assert not tryb_przegladowy("Przyjmij, że wszystkie podane okresy są udowodnione. Czy spełnia warunek stażowy?")


# --- D: „albo” zamienione na „i” ------------------------------------------------------------------------
ZLE_I = ("Kobieta może przejść na emeryturę po osiągnięciu wieku 55 lat, jeżeli ma co najmniej 30-letni okres składkowy "
         "i nieskładkowy i ma co najmniej 20-letni okres składkowy i nieskładkowy oraz została uznana za całkowicie "
         "niezdolną do pracy (art. 29 ust. 1 pkt 1).")


def test_D_zamiana_albo_na_i_jest_odrzucona_i_naprawiona(indeks_fus, monkeypatch):
    wywolania = skrypt_llm(
        monkeypatch, z_naprawa=True, analiza=analiza(),
        odpowiedz=odpowiedz_modelu(ZLE_I, [A29U1]),
        naprawa=odpowiedz_modelu("Kobieta może przejść na emeryturę po osiągnięciu wieku 55 lat, jeżeli ma co najmniej "
                                 "30-letni okres składkowy i nieskładkowy albo jeżeli ma co najmniej 20-letni okres "
                                 "składkowy i nieskładkowy oraz została uznana za całkowicie niezdolną do pracy "
                                 "(art. 29 ust. 1 pkt 1).", [A29U1]),
        weryfikacja=[weryfikacja([werdykt(1, A29U1, struktura_warunkow_ok=False)],
                                 bledy=[blad(1, "struktura_warunkow", "warianty „albo” połączone spójnikiem „i”", A29U1)]),
                     weryfikacja([werdykt(1, A29U1)])])
    w = pytaj(indeks_fus, "Czy kobieta z 30-letnim stażem może przejść na emeryturę po 55 latach?")
    assert w.status_weryfikacji == "verified" and w.naprawa and " albo " in w.tekst
    assert "struktura_warunkow" in wywolania[3][1]                  # lista błędów dotarła do generatora


def test_D_gdy_naprawa_tez_zawodzi_zostaje_odmowa_z_powodem(indeks_fus, monkeypatch):
    wer_zle = weryfikacja([werdykt(1, A29U1, struktura_warunkow_ok=False)],
                          bledy=[blad(1, "struktura_warunkow", "„albo” zamienione na „i”", A29U1)])
    skrypt_llm(monkeypatch, z_naprawa=True, analiza=analiza(), odpowiedz=odpowiedz_modelu(ZLE_I, [A29U1]),
               naprawa=odpowiedz_modelu(ZLE_I, [A29U1]), weryfikacja=wer_zle)
    w = pytaj(indeks_fus, "Czy kobieta z 30-letnim stażem może przejść na emeryturę po 55 latach?")
    assert w.tryb == "odmowa" and w.status_weryfikacji == "insufficient_sources" and not w.naprawa
    assert any("zmieniona struktura warunków" in z for z in w.zdania_usuniete)


# --- E: przesłanka z odrębnej podstawy ---------------------------------------------------------------------
def test_E_przeslanka_odrebnej_podstawy_jest_wykryta_i_zdanie_usuniete(indeks_fus, monkeypatch):
    drugie = ("Dodatkowo musi być uprawniona do renty z tytułu niezdolności do pracy spowodowanej wypadkiem przy pracy "
              "(art. 31).")
    skrypt_llm(
        monkeypatch, analiza=analiza(),
        odpowiedz=odpowiedz_modelu(POPRAWNE_30 + " " + drugie, [A29U1, A31]),
        weryfikacja=[weryfikacja([werdykt(1, A29U1), werdykt(2, A31)], nie_laczy_odrebnych_podstaw=False,
                                 bledy=[blad(2, "odrebna_podstawa", "warunek art. 31 przedstawiony jako warunek art. 29", A31)]),
                     weryfikacja([werdykt(1, A29U1)])])                    # ponowna weryfikacja tego, co zostało
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.tryb == "odpowiedz" and "renty" not in w.tekst and w.status_weryfikacji == "verified"
    assert any(z.startswith(drugie) and "odrebna_podstawa" in z for z in w.zdania_usuniete)


def test_E_podejrzenie_laczenia_podstaw_bez_wskazania_zdania_nie_daje_verified(indeks_fus, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
               weryfikacja=weryfikacja([werdykt(1, A29U1)], nie_laczy_odrebnych_podstaw=False))
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.tryb == "odpowiedz" and w.status_weryfikacji == "unverified" and "laczy_odrebne_podstawy" in kody(w)


# --- F: awaria weryfikatora ---------------------------------------------------------------------------------
@pytest.mark.parametrize("wyjatek", [llm.BrakLLM("timeout"), ValueError("Model nie zwrócił poprawnego JSON")])
def test_F_awaria_weryfikatora_to_unverified_nigdy_verified(indeks_fus, monkeypatch, wyjatek):
    skrypt_llm(monkeypatch, z_naprawa=True, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
               weryfikacja=wyjatek)
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    # kontrola liczb i identyfikatorów przeszła (liczby są w kontekście), a mimo to status NIE jest verified
    assert w.tryb == "odpowiedz" and w.tekst and w.status_weryfikacji == "unverified" and not w.naprawa
    assert "weryfikacja_nieudana" in kody(w) and "NIEZWERYFIKOWANA" in w.uwagi
    assert w.slad["kontrola"][0]["weryfikacja"] == "awaria"


def test_F_wylaczona_weryfikacja_to_unverified(indeks_fus, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]))
    w = pytaj(indeks_fus, PYTANIE_MAMA, weryfikacja=False)
    assert w.status_weryfikacji == "unverified" and "weryfikacja_wylaczona" in kody(w)


# --- G: data bez wersji historycznej ---------------------------------------------------------------------------
def test_G_data_wskazana_przez_uzytkownika_bez_wersji_korpusu_daje_jawne_ograniczenie(indeks_fus, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
               weryfikacja=weryfikacja([werdykt(1, A29U1)]))
    w = pytaj(indeks_fus, PYTANIE_MAMA, na_dzien="2010-05-01")
    ostrz = [o for o in w.ostrzezenia if o["kod"] == "brak_wersji_historycznej"]
    assert len(ostrz) == 1 and "2010-05-01" in ostrz[0]["tekst"] and "nie potwierdza" in ostrz[0]["tekst"]
    assert w.status_weryfikacji == "verified"                       # to ograniczenie źródła, nie błąd odpowiedzi


def test_G_data_domyslna_nie_dostaje_ostrzezenia_o_historii(indeks_fus, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
               weryfikacja=weryfikacja([werdykt(1, A29U1)]))
    w = pytaj(indeks_fus, PYTANIE_MAMA, na_dzien="2026-10-12", data_wskazana=False)     # data „na dziś” z warstwy WWW
    assert "brak_wersji_historycznej" not in kody(w)


def test_G_data_poza_zakresem_jedynej_wersji_to_insufficient_sources(tmp_path, monkeypatch):
    idx = zbuduj_indeks_fus(tmp_path, obowiazuje_od="2026-06-19")            # data z metadanych źródła
    wywolania = skrypt_llm(monkeypatch)
    w = pytaj(idx, PYTANIE_MAMA, na_dzien="2020-01-01", analiza=False)
    assert w.tryb == "odmowa" and w.status_weryfikacji == "insufficient_sources" and wywolania == []
    assert "brak_wersji_na_date" in kody(w)
    assert "2020-01-01" in w.tekst and WERSJA_FUS in w.tekst and "2026-06-19" in w.tekst
    assert _brak_wersji_na_date(idx, "2026-07-01") is None


def test_G_data_w_zakresie_jedynej_wersji_ostrzega_o_braku_historii(tmp_path, monkeypatch):
    idx = zbuduj_indeks_fus(tmp_path, obowiazuje_od="2026-06-19")
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
               weryfikacja=weryfikacja([werdykt(1, A29U1)]))
    w = pytaj(idx, PYTANIE_MAMA, na_dzien="2026-07-01")
    assert "jedna_wersja_aktu" in kody(w) and "2026-06-19" in next(o for o in w.ostrzezenia if o["kod"] == "jedna_wersja_aktu")["tekst"]


# --- H: usunięcie kluczowego zdania ------------------------------------------------------------------------------
def test_H_po_usunieciu_kluczowego_zdania_odpowiedz_nie_przechodzi_jako_poprawna(indeks_fus, monkeypatch):
    drugie = "Pełna ocena uprawnienia wymaga sprawdzenia dalszych warunków (art. 29 ust. 2)."
    wywolania = skrypt_llm(
        monkeypatch, analiza=analiza(),
        odpowiedz=odpowiedz_modelu(POPRAWNE_30 + " " + drugie, [A29U1, A29]),
        weryfikacja=[weryfikacja([werdykt(1, A29U1, pokryte=False, id=""), werdykt(2, A29)]),     # kluczowe zdanie wypada
                     weryfikacja([werdykt(1, A29)], odpowiada_na_glowna_watpliwosc=False)])     # reszta nie odpowiada
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.tryb == "odpowiedz" and w.status_weryfikacji == "unverified" and "nie_rozstrzyga" in kody(w)
    assert w.tekst == drugie and any(z.startswith(POPRAWNE_30) for z in w.zdania_usuniete)
    assert [e for e, _, _ in wywolania] == ["analiza", "odpowiedz", "weryfikacja", "weryfikacja"]
    assert POPRAWNE_30 not in wywolania[3][1] and drugie in wywolania[3][1]       # druga kontrola widzi tylko resztę


def test_H_naprawa_po_usunieciu_kluczowego_zdania_i_ponowna_porazka_daje_ograniczenie(indeks_fus, monkeypatch):
    drugie = "Pełna ocena uprawnienia wymaga sprawdzenia dalszych warunków (art. 29 ust. 2)."
    bez_rozstrzygniecia = weryfikacja([werdykt(1, A29)], odpowiada_na_glowna_watpliwosc=False)
    wywolania = skrypt_llm(
        monkeypatch, z_naprawa=True, analiza=analiza(),
        odpowiedz=odpowiedz_modelu(POPRAWNE_30 + " " + drugie, [A29U1, A29]),
        naprawa=odpowiedz_modelu(drugie, [A29]),
        weryfikacja=[weryfikacja([werdykt(1, A29U1, pokryte=False, id=""), werdykt(2, A29)]),
                     bez_rozstrzygniecia, bez_rozstrzygniecia])
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert [e for e, _, _ in wywolania].count("naprawa") == 1               # najwyżej jedna próba naprawy
    assert w.status_weryfikacji == "unverified" and "nie_rozstrzyga" in kody(w)


# --- mapowanie werdyktów, źródła, dopytanie -----------------------------------------------------------------------
def test_werdykty_sa_mapowane_po_numerach_a_nie_po_kolejnosci(indeks_fus, monkeypatch):
    s1 = POPRAWNE_30
    s2 = "Orzeczenie o niezdolności do pracy jest potrzebne w wariancie z 20-letnim okresem (art. 29 ust. 1 pkt 1)."
    skrypt_llm(
        monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(s1 + " " + s2, [A29U1]),
        weryfikacja=[weryfikacja([werdykt(2, A29U1, pokryte=False, id=""), werdykt(1, A29U1)]),   # odwrócona kolejność
                     weryfikacja([werdykt(1, A29U1)])])
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.tekst == s1 and any(z.startswith(s2) for z in w.zdania_usuniete)         # wypadło zdanie 2, nie 1


def test_zmapuj_werdykty_unit():
    w = [{"nr": 2, "x": "b"}, {"nr": 1, "x": "a"}, {"nr": 1, "x": "duplikat"}, {"nr": 9, "x": "poza zakresem"}, {"x": "bez numeru"}]
    assert _zmapuj_werdykty(w, 3) == {2: {"nr": 2, "x": "b"}, 1: {"nr": 1, "x": "a"}}
    assert _zmapuj_werdykty([{"x": 1}, {"x": 2}], 2) == {1: {"x": 1}, 2: {"x": 2}}     # tylko gdy żaden nie ma numeru


def test_zrodlo_spoza_listy_fragmentow_jest_dopytane_osobno_a_potem_usuwa_zdanie(indeks_fus, monkeypatch):
    zle_zrodlo = werdykt(1, "fus:art999")                                              # id spoza fragmentów weryfikacji
    pojedynczy_ok = {k: v for k, v in werdykt(0, A29U1).items() if k != "nr"}
    wywolania = skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
                           weryfikacja=weryfikacja([zle_zrodlo]), weryfikacja_zdania=pojedynczy_ok)
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.status_weryfikacji == "verified" and [e for e, _, _ in wywolania].count("weryfikacja_zdania") == 1
    pojedynczy_zle = {k: v for k, v in werdykt(0, "fus:art999").items() if k != "nr"}
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
               weryfikacja=weryfikacja([zle_zrodlo]), weryfikacja_zdania=pojedynczy_zle)
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.tryb == "odmowa" and any("nie wskazał poprawnego źródła" in z for z in w.zdania_usuniete)


def test_brak_danych_niezbednych_daje_odpowiedz_warunkowa_z_dopytaniem_obok_nie_przeglad(indeks_fus, monkeypatch):
    wywolania = skrypt_llm(
        monkeypatch, analiza=analiza(niezbedne=[("data urodzenia", "od niej zależy, który przepis o wieku emerytalnym stosuje się")],
                                     pytanie="Podaj datę urodzenia."),
        odpowiedz=odpowiedz_modelu("Wiek emerytalny zależy od daty urodzenia: dla urodzonych po dniu 31 grudnia 1948 r. "
                                   "wynosi co najmniej 60 lat dla kobiet i 65 lat dla mężczyzn (art. 24 ust. 1).",
                                   ["fus:art24:ust1"]),
        weryfikacja=weryfikacja([werdykt(1, "fus:art24:ust1")], dopytanie_jest_konieczne=True))
    w = pytaj(indeks_fus, "W jakim wieku kobieta przechodzi na emeryturę?")
    assert w.tryb == "odpowiedz" and not w.przeglad and w.status_weryfikacji == "verified"
    assert w.dopytanie == "Podaj datę urodzenia." and w.brakujace_parametry == ["data urodzenia"]
    assert "Uwaga: użytkownik podał samo hasło" not in next(p for e, p, _ in wywolania if e == "odpowiedz")
    assert "Podaj datę urodzenia." in next(p for e, p, _ in wywolania if e == "weryfikacja")   # weryfikator ocenia dopytanie


def test_dopytanie_uznane_za_zbedne_nie_trafia_do_uzytkownika(indeks_fus, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(niezbedne=[("płeć", "zmienia próg wieku")], pytanie="Podaj płeć."),
               odpowiedz=odpowiedz_modelu(POPRAWNE_30, [A29U1]),
               weryfikacja=weryfikacja([werdykt(1, A29U1)], dopytanie_jest_konieczne=False))
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.tryb == "odpowiedz" and w.dopytanie == ""
    assert w.brakujace_parametry == [] and w.slad["wynik"]["brakujace_parametry"] == []   # „brakuje: płeć” też znika
    assert w.slad["analiza_wykonana"] is True and w.slad["pomin_analize_efektywne"] is False


def test_odmowa_z_dopytaniem_gdy_brak_podstaw_i_dane_niezbedne(indeks_fus, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(niezbedne=[("data urodzenia", "wybiera przepis")], pytanie="Podaj datę urodzenia."),
               odpowiedz=odpowiedz_modelu("", [], brak_pokrycia=True, uwagi="brak przepisu"))
    w = pytaj(indeks_fus, "Kiedy mogę przejść na emeryturę?")
    assert w.tryb == "dopytanie" and w.tekst == "Podaj datę urodzenia." and w.status_weryfikacji == "insufficient_sources"


def test_znormalizuj_analize_stary_i_nowy_format():
    war, nie, do, pyt = znormalizuj_analize(analiza(niezbedne=[("płeć", "zmienia próg")], do_oceny=[("staż", "ocena indywidualna")],
                                                    pytanie="Podaj płeć.", warianty=["a", "b"]))
    assert war == ["a", "b"] and [d["dana"] for d in nie] == ["płeć"] and [d["dana"] for d in do] == ["staż"] and pyt == "Podaj płeć."
    war, nie, do, pyt = znormalizuj_analize({"warianty": [], "brakujace_parametry": ["płeć"], "pytanie_doprecyzowujace": "Podaj płeć."})
    assert [d["dana"] for d in nie] == ["płeć"] and pyt == "Podaj płeć."                # stary format: dane niezbędne
    war, nie, do, pyt = znormalizuj_analize(analiza(do_oceny=[("staż", "ocena")], pytanie="Podaj staż."))
    assert nie == [] and pyt == ""                                                   # dopytanie tylko przy danych niezbędnych


def test_typy_bledow_zgodne_ze_schematem_i_promptem():
    for typ in TYPY_BLEDOW:
        assert typ in generate.PROMPT_WERYFIKACJA
    assert "art. 29" not in generate.PROMPT_WERYFIKACJA and "art. 31" not in generate.PROMPT_WERYFIKACJA   # reguły ogólne
    assert "art. 29" not in generate.PROMPT_ODPOWIEDZ and "art. 31" not in generate.PROMPT_ODPOWIEDZ


def test_po_usunieciu_pierwszego_zdania_nie_zostaje_wiszacy_spojnik(indeks_fus, monkeypatch):
    drugie = "Jednakże przy braku takiego orzeczenia wymagany jest co najmniej 30-letni okres składkowy i nieskładkowy (art. 29 ust. 1 pkt 1)."
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(
        "Tak, mogła przejść na emeryturę po osiągnięciu wieku 55 lat (art. 29 ust. 1 pkt 1). " + drugie, [A29U1]),
        weryfikacja=[weryfikacja([werdykt(1, A29U1, pokryte=False, id=""), werdykt(2, A29U1)]),
                     weryfikacja([werdykt(1, A29U1)])])
    w = pytaj(indeks_fus, PYTANIE_25_LAT)
    assert w.tekst.startswith("Przy braku takiego orzeczenia") and "Jednakże" not in w.tekst


def test_prompt_odpowiedzi_nie_zacheca_do_katalogu_i_zakazuje_id_w_tekscie():
    p = " ".join(generate.PROMPT_ODPOWIEDZ.split())
    assert "na tym poprzestań" in p and "nie wymieniaj przepisów dla grup szczególnych" in p
    assert "nigdy jako id w nawiasach kwadratowych" in p and "określone warunki" in p


def test_zmienione_odeslanie_i_zdanie_zbedne_usuwaja_zdanie(indeks_fus, monkeypatch):
    drugie = "Zasady ustalania wysokości emerytury mają tu zastosowanie (art. 29 ust. 1 pkt 1)."
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30 + " " + drugie, [A29U1]),
               weryfikacja=[weryfikacja([werdykt(1, A29U1), werdykt(2, A29U1, odeslania_ok=False, potrzebne_do_pytania=False)]),
                            weryfikacja([werdykt(1, A29U1)])])
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.tekst == POPRAWNE_30
    assert any("odesłanie" in z for z in w.zdania_usuniete)
    for pole, powod in (("odeslania_ok", "odesłanie"), ("potrzebne_do_pytania", "zbędne")):
        assert powod in generate._powod_usuniecia(werdykt(1, A29U1, **{pole: False}), {A29U1})


def test_prompty_zakazuja_rozszerzania_tematu_i_podmiany_odesłań():
    p = " ".join(generate.PROMPT_ODPOWIEDZ.split())
    assert "Nie dodawaj przepisów o innym zagadnieniu" in p and "„przepisy te”" in p
    a = " ".join(generate.PROMPT_ANALIZA.split())
    assert "nie dodawaj zagadnień, o które nie zapytano" in a


def test_zdanie_otwarte_zwrotem_wskazujacym_jest_usuwane_bramka_deterministyczna(indeks_fus, monkeypatch):
    for z in ("Przepisy te mają zastosowanie do ustalenia wysokości emerytury (art. 26c ust. 4).",
              "Przepis ten dotyczy innej grupy osób.", "Te przepisy stosuje się odpowiednio.",
              "Powyższe przepisy nie dotyczą tej sytuacji."):
        assert generate.zaczyna_sie_zwrotem_wskazujacym(z), z
    for z in ("Prawo do emerytury ulega zawieszeniu (art. 103a).", "Przepisy ust. 1–3 stosuje się odpowiednio.",
              "Przepisy art. 29 dotyczą wieku."):
        assert not generate.zaczyna_sie_zwrotem_wskazujacym(z), z
    drugie = "Przepisy te mają zastosowanie do ustalenia wysokości emerytury (art. 29 ust. 1 pkt 1)."
    skrypt_llm(monkeypatch, analiza=analiza(), odpowiedz=odpowiedz_modelu(POPRAWNE_30 + " " + drugie, [A29U1]),
               weryfikacja=weryfikacja([werdykt(1, A29U1)]))
    w = pytaj(indeks_fus, PYTANIE_MAMA)
    assert w.tekst == POPRAWNE_30 and any("zwrot wskazujący" in z for z in w.zdania_usuniete)
