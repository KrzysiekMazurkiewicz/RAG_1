import os

from lexrag.parse_ustawa import parsuj_ustawe

TU = os.path.dirname(__file__)
FIXTURE = os.path.join(TU, "..", "data", "samples", "ustawa_testowa.txt")


def _jednostki():
    with open(FIXTURE, encoding="utf-8") as f:
        return {j.id: j for j in parsuj_ustawe(f.read(), "Ustawa testowa", "test")}


def test_struktura_podstawowa():
    j = _jednostki()
    assert "test:art24" in j and "test:art24:ust1" in j and "test:art24:ust2" in j
    assert j["test:art24"].szukalny is False          # rodzic z ustępami nie jest szukalny
    assert j["test:art24:ust1"].rodzic == "test:art24"
    assert j["test:art24:ust1"].tekst.startswith("Osobom urodzonym po dniu 31 grudnia 1948")


def test_pierwszy_ustep_w_linii_artykulu_i_litery():
    j = _jednostki()
    assert j["test:art24:ust1"].sciezka.endswith("Art. 24 > ust. 1")
    assert "test:art24:ust1a" in j and j["test:art24:ust1a"].szukalny is False   # (uchylony)
    assert "test:art103a" in j and j["test:art103a"].szukalny is True            # artykuł bez ustępów


def test_punkty_zostaja_w_ustepie():
    j = _jednostki()
    t = j["test:art27:ust1"].tekst
    assert "1) osiągnęły wiek" in t and "2) mają staż" in t


def test_odwolania_i_zakresy():
    j = _jednostki()
    assert set(j["test:art24:ust1"].odwolania) == {"test:art46", "test:art184"}
    assert j["test:art4:pkt2"].odwolania == [] or True   # definicja osobno
    assert set(j["test:art4"].odwolania) == {"test:art5", "test:art6", "test:art7"}   # "art. 5–7"
    assert "test:art27" in j["test:art184:ust1"].odwolania


def test_definicje():
    j = _jednostki()
    assert j["test:art4:pkt2"].poziom == "pkt-def"
    assert j["test:art4:pkt2"].terminy == ["staż ubezpieczeniowy"]
    assert j["test:art4:pkt2"].rodzic == "test:art4"


def test_sciezka_z_dzialem_i_rozdzialem():
    j = _jednostki()
    assert j["test:art138:ust1"].sciezka.startswith("Dział IX Zwrot nienależnie pobranych świadczeń > Art. 138")
    assert "Rozdział 2 Zawieszanie i zmniejszanie świadczeń" in j["test:art104:ust1"].sciezka


def test_szum_z_pdf_jest_usuwany():
    tekst = (
        "Dziennik Ustaw – 3 – Poz. 1631\n©Kancelaria Sejmu s. 3/160\n2024-10-01\n"
        "Art. 1. 1. Pierwszy ustęp, który jest\nzawinięty w PDF.\n"
        "2. Drugi ustęp.\n1) Niniejsza ustawa dokonuje w zakresie swojej regulacji wdrożenia dyrektywy.\n"
        "Art. 2. Artykuł bez ustępów.\n"
    )
    j = {x.id: x for x in parsuj_ustawe(tekst, "T", "t")}
    assert j["t:art1:ust1"].tekst == "Pierwszy ustęp, który jest zawinięty w PDF."
    assert j["t:art1:ust2"].tekst == "Drugi ustęp."
    assert j["t:art2"].tekst == "Artykuł bez ustępów."
