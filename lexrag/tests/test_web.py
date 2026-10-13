"""API WWW: stan, pytanie w trybie retrieval-only, wgranie ustawy i dokumentu (bez embeddingów, bez LLM)."""
import os
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from lexrag import web  # noqa: E402

TU = os.path.dirname(__file__)
DANE = os.path.join(TU, "..", "data", "samples")


@pytest.fixture()
def klient(tmp_path, monkeypatch):
    monkeypatch.setattr(web, "KATALOG_DANYCH", str(tmp_path))
    monkeypatch.setattr(web, "KATALOG_INDEKSU", str(tmp_path / "index"))
    monkeypatch.setattr(web, "Z_EMBEDDINGAMI", False)
    monkeypatch.setattr(web.STAN, "indeks", None)
    monkeypatch.setattr(web.STAN, "zadanie", {"status": "bezczynny", "komunikat": "", "kod": ""})
    monkeypatch.setattr(web, "llm_dostepny", lambda: False)            # testy API bez modelu, nawet gdy Ollama działa
    from lexrag import generate
    monkeypatch.setattr(generate, "llm_dostepny", lambda: False)
    with TestClient(web.app) as k:
        yield k


def _czekaj(klient, sekund=60):
    for _ in range(sekund * 4):
        z = klient.get("/api/stan").json()["zadanie"]
        if z["status"] in ("gotowe", "blad"):
            return z
        time.sleep(0.25)
    raise AssertionError("wgrywanie nie skończyło się w czasie")


def test_stan_bez_indeksu_i_pytanie_409(klient):
    s = klient.get("/api/stan").json()
    assert s["indeks"]["jednostek"] == 0 and s["indeks"]["akty"] == []
    assert klient.post("/api/pytanie", json={"pytanie": "cokolwiek"}).status_code == 409
    assert klient.get("/").status_code == 200 and "LexRAG" in klient.get("/").text


def test_ustawienia_serwera_modelu(klient):
    """Panel zmienia adres i model bez restartu: zapis do data/model.json i sprawdzenie połączenia."""
    from lexrag import config
    stare = (config.KONFIG.llm_base_url, config.KONFIG.llm_model, config.KONFIG.llm_api)
    try:
        r = klient.post("/api/model", json={"base_url": "http://127.0.0.1:9/v1", "model": "testowy", "api": "ollama"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["model"] == "testowy" and d["polaczenie"]["ok"] is False and d["polaczenie"]["komunikat"]
        assert config.KONFIG.llm_model == "testowy" and config.KONFIG.llm_base_url == "http://127.0.0.1:9/v1"
        assert klient.get("/api/model").json()["base_url"] == "http://127.0.0.1:9/v1"
        assert klient.get("/api/stan").json()["llm"]["polaczenie"]["ok"] is False
        assert klient.post("/api/model", json={"base_url": "ftp://x", "model": "m"}).status_code == 400
        assert klient.post("/api/model", json={"base_url": "http://x", "model": ""}).status_code == 400
        assert os.path.exists(os.path.join(web.KATALOG_DANYCH, "model.json"))
    finally:
        config.ustaw(llm_base_url=stare[0], llm_model=stare[1], llm_api=stare[2])


def test_wgranie_ustawy_i_dokumentu_oraz_pytanie(klient):
    with open(os.path.join(DANE, "ustawa_testowa.txt"), "rb") as f:
        r = klient.post("/api/wgraj", files={"plik": ("ustawa.txt", f, "text/plain")},
                        data={"kod": "test", "akt": "Ustawa testowa", "typ": "ustawa", "wersja": "tj. Dz.U. 2026 poz. 1"})
    assert r.status_code == 200, r.text
    z = _czekaj(klient)
    assert z["status"] == "gotowe", z
    with open(os.path.join(DANE, "procedura_testowa.md"), "rb") as f:
        r = klient.post("/api/wgraj", files={"plik": ("procedura.md", f, "text/markdown")},
                        data={"kod": "pnps", "akt": "Procedura NPS", "typ": "dokument", "wersja": "3.2"})
    assert r.status_code == 200, r.text
    z = _czekaj(klient)
    assert z["status"] == "gotowe", z
    s = klient.get("/api/stan").json()
    assert {a["kod"] for a in s["indeks"]["akty"]} == {"test", "pnps"}
    w = klient.post("/api/pytanie", json={"pytanie": "za jaki okres można żądać zwrotu nienależnie pobranych świadczeń"}).json()
    assert w["tryb"] == "retrieval-only"                      # bez LLM: same trafienia
    assert w["trafienia"][0]["id"] == "test:art138:ust3"
    assert klient.get("/api/jednostka/test:art138:ust3").json()["akt_kod"] == "test"
    assert klient.get("/api/jednostka/nie:ma").status_code == 404
    assert w["status_weryfikacji"] == "" and w["zrodla"] == [] and w["id_odpowiedzi"] == w["trace_id"]
    assert w["slad_url"] is None and not os.path.exists(os.path.join(web.KATALOG_DANYCH, "slady"))   # ślad domyślnie wyłączony
    assert next(a for a in s["indeks"]["akty"] if a["kod"] == "test")["wersja"] == "tj. Dz.U. 2026 poz. 1"
    assert next(a for a in s["indeks"]["akty"] if a["kod"] == "pnps")["wersja"] == "3.2"
    akt = next(a for a in s["indeks"]["akty"] if a["kod"] == "test")
    assert 0 < akt["opisy"] < akt["szukalnych"]                             # parser opisuje tylko definicje
    assert klient.post("/api/opisz/nie_ma").status_code == 404
    assert klient.post("/api/opisz/test").status_code == 409                # bez modelu opisów nie ma


def test_walidacja_wgrania(klient):
    r = klient.post("/api/wgraj", files={"plik": ("x.exe", b"abc", "application/octet-stream")}, data={"kod": "Zły kod"})
    assert r.status_code == 400
    r = klient.post("/api/wgraj", files={"plik": ("x.exe", b"abc", "application/octet-stream")}, data={"kod": "ok"})
    assert r.status_code == 400


def test_opinie_od_zgloszenia_do_eksportu(klient):
    with open(os.path.join(DANE, "ustawa_testowa.txt"), "rb") as f:
        klient.post("/api/wgraj", files={"plik": ("ustawa.txt", f, "text/plain")},
                    data={"kod": "test", "akt": "Ustawa testowa", "typ": "ustawa"})
    _czekaj(klient)
    w = klient.post("/api/pytanie", json={"pytanie": "za jaki okres można żądać zwrotu, PESEL 90010112345"}).json()
    ident = w["id_odpowiedzi"]
    assert klient.post("/api/opinia", json={"id_odpowiedzi": ident, "ocena": "zle"}).status_code == 400
    assert klient.post("/api/opinia", json={"id_odpowiedzi": "nie-ma", "ocena": "dobrze"}).status_code == 404
    assert klient.post("/api/opinia", json={"id_odpowiedzi": ident, "ocena": "zle",
                                            "wlasciwy_przepis": ["test:nie:ma"]}).status_code == 400
    r = klient.post("/api/opinia", json={"id_odpowiedzi": ident, "ocena": "zle",
                                         "wlasciwy_przepis": ["test:art138:ust3"], "komentarz": "zły ustęp"})
    assert r.status_code == 200, r.text
    kolejka = klient.get("/api/opinie").json()
    assert len(kolejka) == 1 and kolejka[0]["status"] == "nowa"
    assert "90010112345" not in kolejka[0]["pytanie"] and "***********" in kolejka[0]["pytanie"]
    oid = kolejka[0]["id"]
    r = klient.post(f"/api/opinie/{oid}/przeglad", json={"status": "zatwierdzona", "musi_zawierac": ["3 lata"],
                                                        "przegladajacy": "referent"})
    assert r.status_code == 200
    assert klient.get("/api/opinie").json() == []
    assert klient.get("/api/opinie?status=zatwierdzona").json()[0]["musi_zawierac"] == ["3 lata"]
    e = klient.get("/api/opinie/eksport").json()
    assert e["liczba"] == 1 and "test:art138:ust3" in e["jsonl"] and "3 lata" in e["jsonl"]


def _wgraj_testowa(klient):
    with open(os.path.join(DANE, "ustawa_testowa.txt"), "rb") as f:
        r = klient.post("/api/wgraj", files={"plik": ("ustawa.txt", f, "text/plain")},
                        data={"kod": "test", "akt": "Ustawa testowa", "typ": "ustawa", "wersja": "tj. testowy"})
    assert r.status_code == 200, r.text
    assert _czekaj(klient)["status"] == "gotowe"


def test_slad_domyslnie_wylaczony_opt_in_dostep_i_retencja(klient):
    """Ślad zawiera pełną treść pytań i prompty: wyłączony domyślnie, włączany na jedno pytanie albo globalnie,
    czytany tylko z adresu lokalnego albo z tokenem, a stare pliki są usuwane (retencja)."""
    from lexrag import config
    stare = (config.KONFIG.slad_wlaczony, config.KONFIG.slad_token, config.KONFIG.slad_dni)
    try:
        _wgraj_testowa(klient)
        pytanie = {"pytanie": "za jaki okres można żądać zwrotu nienależnie pobranych świadczeń"}
        w0 = klient.post("/api/pytanie", json=pytanie).json()
        assert w0["slad_url"] is None and klient.get(f"/api/slad/{w0['trace_id']}").status_code in (403, 404)
        w = klient.post("/api/pytanie", json={**pytanie, "slad": True}).json()           # opt-in na jedno pytanie
        assert w["slad_url"] == f"/api/slad/{w['trace_id']}"
        assert klient.get(w["slad_url"]).status_code == 403                                # TestClient to nie adres lokalny
        config.ustaw(slad_token="sekret")
        assert klient.get(w["slad_url"], headers={"X-Lexrag-Token": "zly"}).status_code == 403
        assert klient.get(f"{w['slad_url']}?token=zly").status_code == 403
        slad = klient.get(w["slad_url"], headers={"X-Lexrag-Token": "sekret"}).json()
        assert slad["trace_id"] == w["trace_id"] and slad["pytanie"].startswith("za jaki okres")
        slad_q = klient.get(f"{w['slad_url']}?token=sekret").json()
        assert slad_q["trace_id"] == w["trace_id"]
        assert slad["kontekst"][0] == "test:art138:ust3" and slad["trafienia"][0]["id"] == "test:art138:ust3"
        assert slad["wynik"]["tryb"] == "retrieval-only" and "konfiguracja" in slad and "llm_api_key" not in slad["konfiguracja"]
        assert klient.get("/api/slad/" + "0" * 32, headers={"X-Lexrag-Token": "sekret"}).status_code == 404
        # test bezpiecznego maskowania PESEL bez niszczenia liczb w JSON
        test_slad = {"pytanie": "mam PESEL 90010112345", "liczba_11_cyfr": 12345678901, "zagniezdzone": {"tekst": "PESEL 80010112345"}}
        web._zapisz_slad("1234567890abcdef1234567890abcdef", test_slad)
        odczyt = klient.get("/api/slad/1234567890abcdef1234567890abcdef?token=sekret").json()
        assert odczyt["liczba_11_cyfr"] == 12345678901
        assert "90010112345" not in odczyt["pytanie"] and "***********" in odczyt["pytanie"]
        assert "80010112345" not in odczyt["zagniezdzone"]["tekst"] and "***********" in odczyt["zagniezdzone"]["tekst"]
        # retencja: plik starszy niż LEXRAG_TRACE_DAYS znika przy następnym zapisie
        katalog = os.path.join(web.KATALOG_DANYCH, "slady")
        stary = os.path.join(katalog, f"{w['trace_id']}.json")
        dawno = time.time() - 30 * 86400
        os.utime(stary, (dawno, dawno))
        config.ustaw(slad_wlaczony=True)                                                    # globalne włączenie
        w2 = klient.post("/api/pytanie", json=pytanie).json()
        assert w2["slad_url"] is not None and not os.path.exists(stary)
        assert "?token=sekret" in w2["slad_url"]                                           # token w linku dla przeglądarki
        assert os.path.exists(os.path.join(katalog, f"{w2['trace_id']}.json"))
    finally:
        config.ustaw(slad_wlaczony=stare[0], slad_token=stare[1], slad_dni=stare[2])


def test_konfiguracja_w_sladzie_bez_sekretow(monkeypatch):
    import dataclasses
    import json

    from lexrag import config
    monkeypatch.setattr(config, "KONFIG", dataclasses.replace(
        config.KONFIG, llm_base_url="https://uzytkownik:haslo@serwer:11434/v1", llm_api_key="TAJNY-KLUCZ"))
    zrzut = json.dumps(config.konfiguracja_do_sladu(), ensure_ascii=False)
    assert "TAJNY-KLUCZ" not in zrzut and "haslo" not in zrzut and "uzytkownik" not in zrzut
    assert "serwer:11434" in zrzut
