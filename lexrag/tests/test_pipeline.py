"""Testy ścieżki pytanie → odpowiedź bez prawdziwego LLM (odpowiedzi modelu są zamockowane),
żeby sprawdzić orkiestrację: dopytanie, cytaty renderowane z indeksu, odrzucanie zdań bez
pokrycia, odmowa. Embeddingi i reranker są wyłączone (BM25 wystarcza na fixture)."""
import json
import dataclasses
import os

import pytest

from lexrag import generate, llm
from pomoce import analiza, odpowiedz_modelu, skrypt_llm, werdykt, weryfikacja
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


def test_odpowiedz_z_cytatami_i_usuwaniem_zdan(indeks, monkeypatch):
    wywolania = skrypt_llm(
        monkeypatch,
        analiza=analiza(warianty=["zwrot nienależnie pobranych świadczeń okres"]),
        odpowiedz=odpowiedz_modelu(
            "Zwrotu można żądać za okres nie dłuższy niż 3 lata. Jeśli osoba zawiadomiła organ, okres skraca się do "
            "12 miesięcy. Odsetki wynoszą 7% rocznie.", ["test:art138:ust3", "test:nieistniejacy"]),
        weryfikacja=weryfikacja([werdykt(1, "test:art138:ust3"), werdykt(2, "test:art138:ust3")]))
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz" and w.status_weryfikacji == "verified"
    assert [c.id for c in w.cytaty] == ["test:art138:ust3"]          # nieistniejące id odfiltrowane
    assert w.cytaty[0].tekst.startswith("Nie można żądać zwrotu")      # tekst z indeksu, nie z modelu
    assert len(w.zdania_usuniete) == 1 and "Odsetki" in w.zdania_usuniete[0] and "liczby spoza przepisów" in w.zdania_usuniete[0]
    assert "Odsetki" not in w.tekst
    assert [e for e, _, _ in wywolania] == ["analiza", "odpowiedz", "weryfikacja"]


def test_odmowa_gdy_brak_pokrycia(indeks, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(),
               odpowiedz=odpowiedz_modelu("Nie wiadomo.", [], brak_pokrycia=True, uwagi="brak przepisu o odsetkach"))
    w = generate.odpowiedz(indeks, "jaka jest stopa odsetek od nienależnie pobranych", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odmowa" and w.status_weryfikacji == "insufficient_sources"


def test_odmowa_gdy_odpowiedz_nie_na_temat(indeks, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(),
               odpowiedz=odpowiedz_modelu("Zwrotu można żądać za okres nie dłuższy niż 3 lata.", ["test:art138:ust3"]),
               weryfikacja=weryfikacja([werdykt(1, "test:art138:ust3")], odpowiada_na_pytanie=False))
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odmowa" and "innej kwestii" in w.tekst and w.status_weryfikacji == "insufficient_sources"
    assert any("nie na temat" in z for z in w.zdania_usuniete)


def test_weryfikacja_dopytuje_o_brakujace_werdykty(indeks, monkeypatch):
    """Mały model ocenia tylko pierwsze zdanie – reszta ma być sprawdzona osobno, nie wycięta domyślnie;
    po usunięciu zdania to, co zostało, jest weryfikowane jeszcze raz."""
    wywolania = skrypt_llm(
        monkeypatch,
        analiza=analiza(),
        odpowiedz=odpowiedz_modelu(
            "Za jaki okres można żądać zwrotu? Zwrotu można żądać za okres nie dłuższy niż 3 lata. "
            "Jeśli osoba zawiadomiła organ, okres skraca się do 12 miesięcy. Odsetki są ustawowe.", ["test:art138:ust3"]),
        weryfikacja=[weryfikacja([werdykt(1, "test:art138:ust3")]),                       # zbiorcza: tylko zdanie 1 z 3
                     weryfikacja([werdykt(1, "test:art138:ust3"), werdykt(2, "test:art138:ust3")])],   # po usunięciu
        weryfikacja_zdania=[{k: v for k, v in werdykt(0, "test:art138:ust3").items() if k != "nr"},
                            {k: v for k, v in werdykt(0, "", pokryte=False).items() if k != "nr"}])
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz" and w.status_weryfikacji == "verified"
    assert w.tekst == ("Zwrotu można żądać za okres nie dłuższy niż 3 lata. "
                       "Jeśli osoba zawiadomiła organ, okres skraca się do 12 miesięcy.")
    assert any("[powtórzenie pytania]" in z for z in w.zdania_usuniete)
    assert any("Odsetki" in z and "bez pokrycia" in z for z in w.zdania_usuniete)
    etykiety = [e for e, _, _ in wywolania]
    assert etykiety == ["analiza", "odpowiedz", "weryfikacja", "weryfikacja_zdania", "weryfikacja_zdania", "weryfikacja"]
    schemat = wywolania[2][2]["properties"]["zdania"]
    assert schemat["minItems"] == 3 == schemat["maxItems"]
    assert "Odsetki są ustawowe." in wywolania[4][1]                    # trzecie zdanie poszło do osobnej weryfikacji
    assert "Odsetki" not in wywolania[5][1]                              # ponowna weryfikacja widzi już tylko resztę


def test_cytaty_odzyskane_z_tresci_gdy_model_nie_podal_id(indeks, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(),
               odpowiedz=odpowiedz_modelu("Zgodnie z art. 138 ust. 3 zwrotu można żądać za okres nie dłuższy niż 3 lata.", []),
               weryfikacja=weryfikacja([werdykt(1, "test:art138:ust3")]))
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odpowiedz"
    assert [c.id for c in w.cytaty] == ["test:art138:ust3"]


def test_odmowa_gdy_zostaje_tylko_powtorzenie_pytania(indeks, monkeypatch):
    skrypt_llm(monkeypatch, analiza=analiza(),
               odpowiedz=odpowiedz_modelu("Za jaki okres można żądać zwrotu? Odsetki wynoszą 7% rocznie.", ["test:art138:ust3"]))
    w = generate.odpowiedz(indeks, "za jaki okres można żądać zwrotu", uzyj_emb=False, uzyj_reranker=False)
    assert w.tryb == "odmowa" and "Żadne zdanie" in w.tekst and w.status_weryfikacji == "insufficient_sources"


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


def test_modele_sa_wywolywane_pojedynczo(monkeypatch):
    """Równoległe żądania nie mogą wołać modelu naraz (Apple GPU przerywa wtedy cały proces); duże partie są dzielone,
    żeby pytanie nie czekało na cały budowany indeks."""
    import threading
    import time

    import numpy as np

    from lexrag import index

    stan = {"w": 0, "max": 0, "wywolan": 0}

    class Udawany:
        def encode(self, teksty, **kw):
            stan["w"] += 1
            stan["wywolan"] += 1
            stan["max"] = max(stan["max"], stan["w"])
            time.sleep(0.03)
            stan["w"] -= 1
            return np.ones((len(teksty), 4), dtype=np.float32)

    monkeypatch.setattr(index, "_MODEL_EMB", Udawany())
    wyniki = []
    watki = [threading.Thread(target=lambda: wyniki.append(index.embedduj(["a"] * 70, ""))) for _ in range(4)]
    for w in watki:
        w.start()
    for w in watki:
        w.join()
    assert stan["max"] == 1                                       # nigdy dwa wywołania naraz
    assert stan["wywolan"] == 4 * 3                               # 70 tekstów = 3 partie po 32, osobna blokada na partię
    assert all(w.shape == (70, 4) for w in wyniki)

