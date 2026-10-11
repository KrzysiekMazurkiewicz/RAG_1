"""Testy ścieżki pytanie → odpowiedź bez prawdziwego LLM (odpowiedzi modelu są zamockowane),
żeby sprawdzić orkiestrację: dopytanie, cytaty renderowane z indeksu, odrzucanie zdań bez
pokrycia, odmowa. Embeddingi i reranker są wyłączone (BM25 wystarcza na fixture)."""
import json
import dataclasses
import os

import pytest

from lexrag import generate, llm
from lexrag.index import Indeks
from lexrag.models import zapisz_jsonl
from lexrag.parse_doc import parsuj_markdown
from lexrag.parse_ustawa import parsuj_ustawe
from lexrag.retrieve import rozszerz, szukaj

TU = os.path.dirname(__file__)
DANE = os.path.join(TU, "..", "data", "samples")


@pytest.fixture(scope="module")
def indeks(tmp_path_factory):
    katalog = tmp_path_factory.mktemp("idx")
    with open(os.path.join(DANE, "ustawa_testowa.txt"), encoding="utf-8") as f:
        u = parsuj_ustawe(f.read(), "Ustawa testowa", "test")
    with open(os.path.join(DANE, "procedura_testowa.md"), encoding="utf-8") as f:
        p = parsuj_markdown(f.read(), "Procedura P-NPS", "pnps")
    pu, pp = str(katalog / "u.jsonl"), str(katalog / "p.jsonl")
    zapisz_jsonl(u, pu)
    zapisz_jsonl(p, pp)
    return Indeks.zbuduj([pu, pp], str(katalog / "index"), z_embeddingami=False)


def test_bm25_trafia_w_ustep(indeks):
    t = szukaj(indeks, ["za jaki okres można żądać zwrotu nienależnie pobranych świadczeń"], uzyj_emb=False)
    assert t[0].id == "test:art138:ust3"


def test_rozszerzanie_dodaje_rodzica_odwolania_i_definicje(indeks):
    t = szukaj(indeks, ["wiek świadczenie urodzeni po 1948"], uzyj_emb=False)[:3]
    ids = {j.id for j in rozszerz(indeks, t)}
    assert "test:art24:ust1" in ids and "test:art24" in ids          # trafienie + rodzic
    assert "test:art46" in ids or "test:art184" in ids                # odesłania z ust. 1


def test_procedura_z_markdown(indeks):
    t = szukaj(indeks, ["po ilu dniach od upomnienia tytuł wykonawczy"], uzyj_emb=False)
    assert t[0].id.startswith("pnps:")
    assert "Upomnienie" in indeks.jednostki[t[0].id].sciezka


def _mock_llm(monkeypatch, odpowiedzi):
    """Kolejne wywołania czat_json dostają kolejne słowniki z listy."""
    kolejka = list(odpowiedzi)

    def fake_czat_json(wiadomosci, model=None, proby=2, **kw):
        return kolejka.pop(0)

    monkeypatch.setattr(generate, "czat_json", fake_czat_json)
    monkeypatch.setattr(generate, "llm_dostepny", lambda: True)
    monkeypatch.setattr(generate, "KONFIG", dataclasses.replace(generate.KONFIG, pomin_analize=False, naprawa=False))  # niezależnie od LEXRAG_SKIP_ANALYSIS; bez cyklu naprawczego


def test_brak_parametrow_nie_wlacza_przegladu_tylko_dopytanie(indeks, monkeypatch):
    """Brak danych wg analizy NIE przełącza odpowiedzi w tryb przeglądowy (po recenzji audytu z 11.10.2026):
    odpowiedź jest zwykła (warunkowa), pytanie doprecyzowujące idzie obok niej, a weryfikator dostaje
    oryginalne pytanie bez dopisku o trybie."""
    prompty = []
    odpowiedzi = [
        {"brakujace_parametry": ["data urodzenia", "płeć"], "pytanie_doprecyzowujace": "Podaj datę urodzenia i płeć.", "warianty": []},
        {"odpowiedz": "Świadczenie przysługuje po osiągnięciu wieku emerytalnego, który zależy od płci.",
         "cytowane_id": ["test:art24:ust1"], "brak_pokrycia": False},
        {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art24:ust1", "warunki_poprawne": True, "odrebna_podstawa_jako_warunek": False}],
         "odpowiada_na_pytanie": True, "odpowiada_wprost": True, "bledy": []},
    ]

    def fake(wiadomosci, model=None, proby=2, **kw):
        prompty.append(wiadomosci[-1]["content"])
        return odpowiedzi[len(prompty) - 1]

    monkeypatch.setattr(generate, "czat_json", fake)
    monkeypatch.setattr(generate, "llm_dostepny", lambda: True)
    monkeypatch.setattr(generate, "KONFIG", dataclasses.replace(generate.KONFIG, pomin_analize=False, naprawa=False))
    w = generate.odpowiedz(indeks, "od jakiego wieku przysługuje świadczenie osobom urodzonym po 1948", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz" and not w.przeglad and w.status_weryfikacji == "zweryfikowana"
    assert "datę urodzenia" in w.dopytanie and w.brakujace_parametry == ["data urodzenia", "płeć"]
    assert [c.id for c in w.cytaty] == ["test:art24:ust1"]
    assert "PRZEGLĄDOWO" not in prompty[1]                                        # prompt odpowiedzi bez dopisku przeglądowego
    assert "[pytanie hasłowe" not in prompty[2] and "Tryb odpowiedzi" not in prompty[2]   # weryfikator ocenia oryginalne pytanie
    assert w.slad["analiza"]["brakujace_parametry"] == ["data urodzenia", "płeć"]
    assert w.slad["kontrola"][0]["status"] == "zweryfikowana" and w.slad["generator"]["cytowane_id"] == ["test:art24:ust1"]
    assert w.slad["trafienia"][0]["id"] in w.slad["kontekst"]


def test_dopytanie_gdy_odpowiedz_przegladowa_nie_ma_pokrycia(indeks, monkeypatch):
    """Gdy odpowiedź przeglądowa się nie udaje, użytkownik dostaje przynajmniej pytanie doprecyzowujące."""
    _mock_llm(monkeypatch, [
        {"brakujace_parametry": ["data urodzenia"], "pytanie_doprecyzowujace": "Podaj datę urodzenia.", "warianty": []},
        {"odpowiedz": "", "cytowane_id": [], "brak_pokrycia": True, "uwagi": "brak przepisu o wieku"},
    ])
    w = generate.odpowiedz(indeks, "wiek świadczenie urodzeni po 1948", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "dopytanie" and w.tekst == "Podaj datę urodzenia." and "brak przepisu" in w.uwagi
    assert w.status_weryfikacji == "brak_zrodel"


def test_pytanie_haslowe_bez_analizy_idzie_w_tryb_przegladowy(indeks, monkeypatch):
    zebrane = []

    def fake(wiadomosci, model=None, proby=2, **kw):
        zebrane.append(wiadomosci[-1]["content"])
        return [{"odpowiedz": "Świadczenie zależy od wieku i płci.", "cytowane_id": ["test:art24:ust1"], "brak_pokrycia": False},
                {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art24:ust1"}], "odpowiada_na_pytanie": True}][len(zebrane) - 1]

    monkeypatch.setattr(generate, "czat_json", fake)
    monkeypatch.setattr(generate, "llm_dostepny", lambda: True)
    monkeypatch.setattr(generate, "KONFIG", dataclasses.replace(generate.KONFIG, pomin_analize=False, naprawa=False))  # niezależnie od LEXRAG_SKIP_ANALYSIS
    w = generate.odpowiedz(indeks, "wiek świadczenie urodzeni po 1948", uzyj_emb=False, uzyj_reranker=False, analiza=False)
    assert w.tryb == "odpowiedz" and w.przeglad and w.dopytanie == ""
    assert "PRZEGLĄDOWO" in zebrane[0]            # prompt odpowiedzi dostał dopisek przeglądowy
    assert "Tryb odpowiedzi: przeglądowa" in zebrane[1] and "[pytanie hasłowe" not in zebrane[1]   # weryfikator: tryb osobną linią, pytanie bez zmian


def test_odpowiedz_z_cytatami_i_usuwaniem_zdan(indeks, monkeypatch):
    _mock_llm(monkeypatch, [
        {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": ["zwrot nienależnie pobranych świadczeń okres"]},
        {"odpowiedz": "Zwrotu można żądać za okres nie dłuższy niż 3 lata. Jeśli osoba zawiadomiła organ, okres skraca się do 12 miesięcy. Odsetki wynoszą 7% rocznie.",
         "cytowane_id": ["test:art138:ust3", "test:nieistniejacy"], "brak_pokrycia": False},
        {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art138:ust3"},
                    {"nr": 2, "pokryte": True, "id": "test:art138:ust3"},
                    {"nr": 3, "pokryte": False, "id": ""}]},
    ])
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz"
    assert [c.id for c in w.cytaty] == ["test:art138:ust3"]          # nieistniejące id odfiltrowane
    assert w.cytaty[0].tekst.startswith("Nie można żądać zwrotu")      # tekst z indeksu, nie z modelu
    assert len(w.zdania_usuniete) == 1 and "Odsetki" in w.zdania_usuniete[0]
    assert "Odsetki" not in w.tekst


def test_odmowa_gdy_brak_pokrycia(indeks, monkeypatch):
    _mock_llm(monkeypatch, [
        {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": []},
        {"odpowiedz": "Nie wiadomo.", "cytowane_id": [], "brak_pokrycia": True, "uwagi": "brak przepisu o odsetkach"},
    ])
    w = generate.odpowiedz(indeks, "jaka jest stopa odsetek od nienależnie pobranych", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odmowa"


def test_wyciagnij_json_toleruje_ogrodzenie():
    assert llm.wyciagnij_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.wyciagnij_json('Oto wynik: {"a": [1, 2]} koniec') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        llm.wyciagnij_json("nic")


def test_filtr_na_dzien(indeks, tmp_path):
    with open(os.path.join(DANE, "ustawa_testowa.txt"), encoding="utf-8") as f:
        stare = parsuj_ustawe(f.read(), "Ustawa testowa (stara wersja)", "stara",
                              obowiazuje_od="2000-01-01", obowiazuje_do="2017-09-30")
    p = str(tmp_path / "stara.jsonl")
    zapisz_jsonl(stare, p)
    idx = Indeks.zbuduj([p], str(tmp_path / "idx"), z_embeddingami=False)
    traf = szukaj(idx, ["zwrot nienależnie pobranych"], na_dzien="2010-01-01", uzyj_emb=False)
    assert traf
    assert not szukaj(idx, ["zwrot nienależnie pobranych"], na_dzien="2024-01-01", uzyj_emb=False)
    assert rozszerz(idx, traf, na_dzien="2010-01-01")                 # rodzice i odesłania w zakresie dat wchodzą
    assert not rozszerz(idx, traf, na_dzien="2024-01-01")             # ...a poza zakresem nie (po recenzji audytu)


def test_odmowa_gdy_odpowiedz_nie_na_temat(indeks, monkeypatch):
    _mock_llm(monkeypatch, [
        {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": []},
        {"odpowiedz": "Zwrotu można żądać za okres nie dłuższy niż 3 lata.",
         "cytowane_id": ["test:art138:ust3"], "brak_pokrycia": False, "uwagi": ""},
        {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art138:ust3"}], "odpowiada_na_pytanie": False},
    ])
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odmowa" and "innej kwestii" in w.tekst
    assert any("nie na temat" in z for z in w.zdania_usuniete)


def test_weryfikacja_dopytuje_o_brakujace_werdykty(indeks, monkeypatch):
    """Mały model ocenia tylko pierwsze zdanie – reszta ma być sprawdzona osobno, nie wycięta domyślnie."""
    wywolania = []

    def fake_czat_json(wiadomosci, model=None, proby=2, **kw):
        wywolania.append((wiadomosci[-1]["content"], kw.get("schema")))
        return kolejka.pop(0)

    kolejka = [
        {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": []},
        {"odpowiedz": "Za jaki okres można żądać zwrotu? Zwrotu można żądać za okres nie dłuższy niż 3 lata. "
                      "Jeśli osoba zawiadomiła organ, okres skraca się do 12 miesięcy. Odsetki są ustawowe.",
         "cytowane_id": ["test:art138:ust3"], "brak_pokrycia": False, "uwagi": ""},
        {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art138:ust3"}], "odpowiada_na_pytanie": True},
        {"pokryte": True, "id": "test:art138:ust3"},     # zdanie 2 – osobno
        {"pokryte": False, "id": ""},                    # zdanie 3 – osobno
    ]
    monkeypatch.setattr(generate, "czat_json", fake_czat_json)
    monkeypatch.setattr(generate, "llm_dostepny", lambda: True)
    monkeypatch.setattr(generate, "KONFIG", dataclasses.replace(generate.KONFIG, pomin_analize=False, naprawa=False))  # niezależnie od LEXRAG_SKIP_ANALYSIS
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz"
    assert w.tekst == ("Zwrotu można żądać za okres nie dłuższy niż 3 lata. "
                       "Jeśli osoba zawiadomiła organ, okres skraca się do 12 miesięcy.")
    assert any("[powtórzenie pytania]" in z for z in w.zdania_usuniete)
    assert any("Odsetki" in z and "bez pokrycia" in z for z in w.zdania_usuniete)
    schemat_zbiorczy = wywolania[2][1]
    assert schemat_zbiorczy["properties"]["zdania"]["minItems"] == 3 == schemat_zbiorczy["properties"]["zdania"]["maxItems"]
    assert "Odsetki są ustawowe." in wywolania[4][0]       # trzecie zdanie poszło do osobnej weryfikacji
    assert not kolejka


def test_cytaty_odzyskane_z_tresci_gdy_model_nie_podal_id(indeks, monkeypatch):
    _mock_llm(monkeypatch, [
        {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": []},
        {"odpowiedz": "Zgodnie z art. 138 ust. 3 zwrotu można żądać za okres nie dłuższy niż 3 lata.",
         "cytowane_id": [], "brak_pokrycia": False, "uwagi": ""},
        {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art138:ust3"}], "odpowiada_na_pytanie": True},
    ])
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz"
    assert [c.id for c in w.cytaty] == ["test:art138:ust3"]


def test_odmowa_gdy_zostaje_tylko_powtorzenie_pytania(indeks, monkeypatch):
    _mock_llm(monkeypatch, [
        {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": []},
        {"odpowiedz": "Za jaki okres można żądać zwrotu? Odsetki wynoszą 7% rocznie.",
         "cytowane_id": ["test:art138:ust3"], "brak_pokrycia": False, "uwagi": ""},
    ])
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odmowa" and "Żadne zdanie" in w.tekst


def test_cykl_naprawczy_gdy_odpowiedz_nie_rozstrzyga(indeks, monkeypatch):
    """Weryfikator wskazuje błędy (odrębna podstawa jako warunek, brak rozstrzygnięcia) → jedna poprawka z listą
    błędów → ponowna kontrola → lepsza wersja zostaje."""
    prompty = []
    odpowiedzi = [
        {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": []},
        {"odpowiedz": "Zwrotu można żądać za okres nie dłuższy niż 3 lata. Aby odpowiedzieć, należy potwierdzić prawo do odsetek.",
         "cytowane_id": ["test:art138:ust3"], "brak_pokrycia": False},
        {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art138:ust3", "warunki_poprawne": True, "odrebna_podstawa_jako_warunek": False},
                    {"nr": 2, "pokryte": True, "id": "", "warunki_poprawne": True, "odrebna_podstawa_jako_warunek": True}],
         "odpowiada_na_pytanie": True, "odpowiada_wprost": False, "bledy": ["zdanie 2: warunek z innego przepisu"]},
        {"odpowiedz": "Tak, zwrotu można żądać za okres nie dłuższy niż 3 lata.", "cytowane_id": ["test:art138:ust3"], "brak_pokrycia": False},
        {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art138:ust3", "warunki_poprawne": True, "odrebna_podstawa_jako_warunek": False}],
         "odpowiada_na_pytanie": True, "odpowiada_wprost": True, "bledy": []},
    ]

    def fake(wiadomosci, model=None, proby=2, **kw):
        prompty.append((kw.get("etykieta"), wiadomosci[-1]["content"]))
        return odpowiedzi[len(prompty) - 1]

    monkeypatch.setattr(generate, "czat_json", fake)
    monkeypatch.setattr(generate, "llm_dostepny", lambda: True)
    monkeypatch.setattr(generate, "KONFIG", dataclasses.replace(generate.KONFIG, pomin_analize=False, naprawa=True))
    w = generate.odpowiedz(indeks, "czy można żądać zwrotu za 3 lata", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz" and w.naprawa and w.status_weryfikacji == "zweryfikowana"
    assert w.tekst.startswith("Tak,") and "potwierdzić" not in w.tekst
    assert [e for e, _ in prompty] == ["analiza", "odpowiedz", "weryfikacja", "naprawa", "weryfikacja"]
    assert "warunek z innego przepisu" in prompty[3][1] and "nie rozstrzyga pytania wprost" in prompty[3][1]
    assert len(w.slad["kontrola"]) == 2 and w.slad["naprawa"]["powody"]


def test_bez_naprawy_zostaje_status_ograniczona(indeks, monkeypatch):
    """Gdy cykl naprawczy jest wyłączony, odpowiedź bez rozstrzygnięcia przechodzi ze statusem „ograniczona” i uwagą."""
    _mock_llm(monkeypatch, [
        {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": []},
        {"odpowiedz": "Zwrotu można żądać za okres nie dłuższy niż 3 lata.", "cytowane_id": ["test:art138:ust3"], "brak_pokrycia": False},
        {"zdania": [{"nr": 1, "pokryte": True, "id": "test:art138:ust3", "warunki_poprawne": True, "odrebna_podstawa_jako_warunek": False}],
         "odpowiada_na_pytanie": True, "odpowiada_wprost": False, "bledy": ["brak rozstrzygnięcia"]},
    ])
    w = generate.odpowiedz(indeks, "czy można żądać zwrotu za 3 lata", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz" and not w.naprawa and w.status_weryfikacji == "ograniczona"
    assert "nie rozstrzyga pytania wprost" in w.uwagi


def test_awaria_weryfikatora_daje_status_niezweryfikowana(indeks, monkeypatch):
    def fake(wiadomosci, model=None, proby=2, **kw):
        et = kw.get("etykieta")
        if et == "analiza":
            return {"brakujace_parametry": [], "pytanie_doprecyzowujace": "", "warianty": []}
        if et == "odpowiedz":
            return {"odpowiedz": "Zwrotu można żądać za okres nie dłuższy niż 3 lata.", "cytowane_id": ["test:art138:ust3"], "brak_pokrycia": False}
        raise llm.BrakLLM("model padł")

    monkeypatch.setattr(generate, "czat_json", fake)
    monkeypatch.setattr(generate, "llm_dostepny", lambda: True)
    monkeypatch.setattr(generate, "KONFIG", dataclasses.replace(generate.KONFIG, pomin_analize=False, naprawa=True))
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz" and w.status_weryfikacji == "niezweryfikowana" and not w.naprawa
    assert "NIEZWERYFIKOWANA" in w.uwagi and w.slad["kontrola"][0]["blad"] == "model padł"
