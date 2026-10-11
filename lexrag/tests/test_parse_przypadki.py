"""Przypadki z prawdziwego PDF Dziennika Ustaw (tekst jednolity ustawy o FUS, Dz.U. 2024 poz. 1631)."""
from lexrag.parse_ustawa import parsuj_ustawe


def _p(tekst):
    return {j.id: j for j in parsuj_ustawe(tekst, "T", "t")}


def test_przyklejone_znaczniki_przypisow_i_przypisy_z_dolu_strony():
    t = ("Art. 24.2) 1.3) Ubezpieczonym urodzonym po dniu 31 grudnia 1948 r. przysługuje emerytura.\n"
         "2) Ze zmianą wprowadzoną przez art. 4 pkt 1 ustawy z dnia 28 lipca 2023 r.\n"
         "1a.15) Dla emerytów przyjmuje się przychód.\n"
         "2. Drugi ustęp.\n"
         "15) Utracił moc z dniem 15 marca 2005 r. na podstawie wyroku Trybunału.\n"
         "Art. 32. 1. Pierwszy.\n"
         "4a. (utracił moc)5)\n"
         "Art. 95a.13) 1. Osobie uprawnionej do renty rodzinnej przysługuje świadczenie.\n")
    j = _p(t)
    assert j["t:art24:ust1"].tekst == "Ubezpieczonym urodzonym po dniu 31 grudnia 1948 r. przysługuje emerytura."
    assert j["t:art24:ust1a"].tekst == "Dla emerytów przyjmuje się przychód."
    assert j["t:art24:ust2"].tekst == "Drugi ustęp."
    assert j["t:art32:ust4a"].tekst == "(utracił moc)" and j["t:art32:ust4a"].szukalny is False
    assert j["t:art95a:ust1"].tekst.startswith("Osobie uprawnionej")


def test_zakres_pominietych_artykulow_i_zalacznik():
    t = ("Art. 144. Ostatni merytoryczny.\n"
         "DZIAŁ X\nPrzepisy zmieniające, przejściowe i końcowe\nRozdział 1\nZmiany w przepisach obowiązujących\n"
         "Art. 145–172. (pominięte)\n"
         "Art. 173. 1. Kapitał początkowy ustala się dla osób, o których mowa w art. 144.\n"
         "Art. 196. Ustawa wchodzi w życie.\n"
         "Załącznik do ustawy z dnia 17 grudnia 1998 r.\nMężczyźni\n1 2 3 4 5 6 7 8 9\n4 5 6 7 8 9 10 11 12\n")
    j = _p(t)
    assert "t:art145-172" in j and j["t:art145-172"].szukalny is False
    assert j["t:art145-172"].sciezka.endswith("Art. 145–172")
    assert j["t:art173:ust1"].odwolania == ["t:art144"]
    assert j["t:art196"].tekst == "Ustawa wchodzi w życie."
    assert j["t:zal1"].tekst.startswith("Załącznik do ustawy") and j["t:zal1"].poziom == "zalacznik"


def test_numer_punktu_sam_w_linii_i_definicje():
    t = ("Art. 4. Użyte w ustawie określenia oznaczają:\n"
         "1)\nemeryt – osobę mającą ustalone prawo do emerytury;\n"
         "2)\n(uchylony)\n"
         "2a) okresowa emerytura kapitałowa – emeryturę, o której mowa w ustawie;\n"
         "Art. 5. 1. Ustęp pierwszy.\n")
    j = _p(t)
    assert "1) emeryt – osobę mającą ustalone prawo do emerytury;" in j["t:art4"].tekst
    assert j["t:art4:pkt1"].terminy == ["emeryt"]
    assert "t:art4:pkt2" not in j
    assert j["t:art4:pkt2a"].terminy == ["okresowa emerytura kapitałowa"]
    assert j["t:art5:ust1"].tekst == "Ustęp pierwszy."


def test_naglowek_strony_sklejony_w_jedna_linie():
    t = ("Art. 1. 1. Pierwszy ustęp, który jest\n"
         "Dziennik Ustaw – 14 – Poz. 1631\n"
         "zawinięty na kolejną stronę.\n"
         "2. Drugi.\n")
    j = _p(t)
    assert j["t:art1:ust1"].tekst == "Pierwszy ustęp, który jest zawinięty na kolejną stronę."
    assert j["t:art1:ust2"].tekst == "Drugi."


def test_dwa_brzmienia_tego_samego_ustepu():
    t = ("Art. 95. 1. W razie zbiegu wypłaca się jedno świadczenie.\n"
         "1. W razie zbiegu wypłaca się jedno świadczenie, z zastrzeżeniem art. 95a.\n"
         "2. Drugi.\n"
         "Art. 95a. Nowy.\n")
    j = _p(t)
    assert j["t:art95:ust1"].tekst.endswith("jedno świadczenie.")
    assert j["t:art95:ust1-b"].tekst.endswith("art. 95a.") and "brzmienie 2" in j["t:art95:ust1-b"].sciezka
    assert j["t:art95:ust1-b"].rodzic == "t:art95" and j["t:art95:ust2"].tekst == "Drugi."
    assert len({x.id for x in j.values()}) == len(j)


def test_preambula_obwieszczenia_nie_jest_ustawa():
    """Obwieszczenie cytuje przepisy przejściowe ustaw zmieniających („Art. 40.” …) PRZED tekstem jednolitym;
    linia „Załącznik do obwieszczenia” zaczyna właściwą ustawę i nie może włączać trybu załącznika."""
    from lexrag.parse_ustawa import parsuj_ustawe
    tekst = """OBWIESZCZENIE MARSZAŁKA SEJMU
2. Podany w załączniku tekst jednolity nie obejmuje:
„Art. 39. Do postępowań stosuje się przepisy dotychczasowe.
Art. 40. Do postępowań dotyczących odwołania stosuje się przepisy dotychczasowe.
Art. 43. Ustawa wchodzi w życie po upływie 3 miesięcy od dnia ogłoszenia.”.
Załącznik do obwieszczenia Marszałka Sejmu Rzeczypospolitej
USTAWA
z dnia 25 czerwca 1999 r.
o świadczeniach pieniężnych
Rozdział 1
Przepisy ogólne
Art. 1. 1. Świadczenia pieniężne przysługują ubezpieczonym.
2. Świadczenia z tytułu wypadków określa odrębna ustawa.
Art. 2. Świadczenia obejmują zasiłek chorobowy.
Załącznik
Tabela stawek dziennych zasiłku chorobowego według grup zawodowych
Grupa A: 80 procent podstawy wymiaru, grupa B: 100 procent podstawy wymiaru
"""
    j = parsuj_ustawe(tekst, "Ustawa testowa", "t")
    ids = {x.id for x in j}
    assert "t:art1:ust1" in ids and "t:art2" in ids
    assert not any(i.startswith("t:art40") or i.startswith("t:art43") for i in ids)
    assert any(x.poziom == "zalacznik" for x in j)           # prawdziwy załącznik ustawy nadal działa
