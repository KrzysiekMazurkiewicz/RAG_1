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


def test_wgranie_ustawy_i_dokumentu_oraz_pytanie(klient):
    with open(os.path.join(DANE, "ustawa_testowa.txt"), "rb") as f:
        r = klient.post("/api/wgraj", files={"plik": ("ustawa.txt", f, "text/plain")},
                        data={"kod": "test", "akt": "Ustawa testowa", "typ": "ustawa"})
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


def test_walidacja_wgrania(klient):
    r = klient.post("/api/wgraj", files={"plik": ("x.exe", b"abc", "application/octet-stream")}, data={"kod": "Zły kod"})
    assert r.status_code == 400
    r = klient.post("/api/wgraj", files={"plik": ("x.exe", b"abc", "application/octet-stream")}, data={"kod": "ok"})
    assert r.status_code == 400
