"""Kotwice kontekstu (art. 5 ust. 2 i kotwice domenowe), ust. 1 dla ustępów szczegółowych,
odesłania z procedur do ustaw, filtrowanie po aktach, czyszczenie ID oraz liczby pochodne z pytania."""
from lexrag.bm25 import BM25
from lexrag.generate import (
    _kontrola,
    czysc_id_w_tekscie,
    korpus_liczb,
    liczby_bez_pokrycia,
    liczby_pochodne_z_pytania,
    _normalizuj_liczby,
)
from lexrag.index import Indeks
from lexrag.models import Jednostka
from lexrag.parse_doc import parsuj_markdown
from lexrag.retrieve import Trafienie, rozszerz, szukaj
from lexrag.tokenize import tokenizuj


def _indeks_min(*jednostki: Jednostka) -> Indeks:
    szukalne = [j.id for j in jednostki if j.szukalny and j.tekst.strip()]
    teksty = [next(x for x in jednostki if x.id == i).tekst_do_indeksu() for i in szukalne]
    bm25 = BM25().dopasuj([tokenizuj(t) for t in teksty])
    return Indeks(list(jednostki), bm25, szukalne, None, {})


def test_kotwica_art5_ust2_przy_okresach_nieskladkowych_i_art87():
    j58 = Jednostka(
        id="fus:art58:ust1", akt="FUS", akt_kod="fus", poziom="ust",
        sciezka="Art. 58 > ust. 1",
        tekst="Wymagany okres składkowy i nieskładkowy wynosi 5 lat.",
    )
    j87 = Jednostka(
        id="fus:art87:ust1", akt="FUS", akt_kod="fus", poziom="ust",
        sciezka="Art. 87 > ust. 1",
        tekst="Podwyższenie emerytury do kwoty najniższej emerytury.",
    )
    j5 = Jednostka(
        id="fus:art5:ust2", akt="FUS", akt_kod="fus", poziom="ust",
        sciezka="Art. 5 > ust. 2",
        tekst="Okresy nieskładkowe uwzględnia się w wymiarze jednej trzeciej.",
    )
    idx = _indeks_min(j58, j87, j5)
    pyt = "w jakim wymiarze uwzględnia się okresy nieskładkowe przy ustalaniu emerytury"
    traf = [Trafienie("fus:art58:ust1", 1.0, "test")]
    ids = {j.id for j in rozszerz(idx, traf, pytanie=pyt, z_rodzicem=False, z_odwolaniami=False, z_definicjami=False)}
    assert "fus:art5:ust2" in ids
    assert "fus:art58:ust1" in ids

    # wyzwalacz dla art. 87 + staż / podwyższenie do najniższej emerytury
    pyt87 = "czy przy 18 latach składkowych przysługuje podwyższenie do najniższej emerytury"
    traf87 = [Trafienie("fus:art87:ust1", 1.0, "test")]
    ids87 = {j.id for j in rozszerz(idx, traf87, pytanie=pyt87, z_rodzicem=False, z_odwolaniami=False, z_definicjami=False)}
    assert "fus:art5:ust2" in ids87


def test_kotwice_domenowe_i_ust1_dla_ustepow_szczegolowych():
    j173_1 = Jednostka(id="fus:art173:ust1", akt="FUS", akt_kod="fus", poziom="ust", rodzic="fus:art173",
                       sciezka="Art. 173 > ust. 1", tekst="Kapitał początkowy ustala się dla ubezpieczonych urodzonych po 1948 r.")
    j173_1a = Jednostka(id="fus:art173:ust1a", akt="FUS", akt_kod="fus", poziom="ust", rodzic="fus:art173",
                        sciezka="Art. 173 > ust. 1a", tekst="Kapitał początkowy ustala się także dla żołnierzy.")
    j8_1 = Jednostka(id="zas:art8:ust1", akt="ZAS", akt_kod="zas", poziom="ust", rodzic="zas:art8",
                     sciezka="Art. 8 > ust. 1", tekst="Zasiłek chorobowy przysługuje przez 182 dni, a przy gruźlicy 270 dni.")
    j9_1 = Jednostka(id="zas:art9:ust1", akt="ZAS", akt_kod="zas", poziom="ust", rodzic="zas:art9",
                     sciezka="Art. 9 > ust. 1", tekst="Do okresu zasiłkowego wlicza się okresy nieprzerwanej niezdolności.")
    j116_1 = Jednostka(id="fus:art116:ust1", akt="FUS", akt_kod="fus", poziom="ust", rodzic="fus:art116",
                       sciezka="Art. 116 > ust. 1", tekst="Postępowanie w sprawach świadczeń wszczyna się na podstawie wniosku.")
    j116_1a = Jednostka(id="fus:art116:ust1a", akt="FUS", akt_kod="fus", poziom="ust", rodzic="fus:art116",
                        sciezka="Art. 116 > ust. 1a", tekst="Postępowanie o emeryturę wszczyna się z urzędu po świadczeniu przedemerytalnym.")
    j140_1 = Jednostka(id="fus:art140:ust1", akt="FUS", akt_kod="fus", poziom="ust", rodzic="fus:art140",
                       sciezka="Art. 140 > ust. 1", tekst="Potrącenia nie mogą przekraczać 60% świadczenia.")
    j140_4 = Jednostka(id="fus:art140:ust4", akt="FUS", akt_kod="fus", poziom="ust", rodzic="fus:art140",
                       sciezka="Art. 140 > ust. 4", tekst="Potrącenia z art. 139 ust. 1.")
    idx = _indeks_min(j173_1, j173_1a, j8_1, j9_1, j116_1, j116_1a, j140_1, j140_4)

    # Gdy trafia ust. 1a, ust. 1 tego samego artykułu wchodzi do kontekstu zaraz po trafieniach
    k173 = [j.id for j in rozszerz(idx, [Trafienie("fus:art173:ust1a", 1.0, "test")], pytanie="kapitał początkowy")]
    assert "fus:art173:ust1" in k173

    # Kotwica dla "okres zasiłkowy" dociąga zas:art8:ust1
    k8 = [j.id for j in rozszerz(idx, [Trafienie("zas:art9:ust1", 1.0, "test")], pytanie="okres zasiłkowy")]
    assert "zas:art8:ust1" in k8

    # Kotwica dla wszczynania postępowania o emeryturę
    k116 = [j.id for j in rozszerz(idx, [Trafienie("fus:art116:ust1a", 1.0, "test")],
                                   pytanie="czy postępowanie o emeryturę wszczyna się z urzędu czy na wniosek")]
    assert "fus:art116:ust1" in k116

    # Kotwica dla potrąceń i ust. 1 dla ust. 4
    k140 = [j.id for j in rozszerz(idx, [Trafienie("fus:art140:ust4", 1.0, "test")], pytanie="potrącenia z emerytury")]
    assert "fus:art140:ust1" in k140


def test_odwolania_z_procedury_do_ustawy_i_filtr_aktow():
    md = "## 2. Decyzja\n\nPodstawa prawna: art. 138 ust. 4 oraz art. 138 ustawy."
    sekcje = parsuj_markdown(md, "Procedura NPS", "pnps")
    assert len(sekcje) == 1
    assert "art138:ust4" in sekcje[0].odwolania
    assert "art. 138 ust. 4" in sekcje[0].odwolania_zewn

    j138_4 = Jednostka(id="fus:art138:ust4", akt="FUS", akt_kod="fus", poziom="ust",
                       sciezka="Art. 138 > ust. 4", tekst="Zwrot za okres nie dłuższy niż 3 lata.")
    j_zas = Jednostka(id="zas:art66:ust1", akt="ZAS", akt_kod="zas", poziom="ust",
                      sciezka="Art. 66 > ust. 1", tekst="Zwrot zasiłku.")
    idx = _indeks_min(sekcje[0], j138_4, j_zas)

    # Rozszerzanie procedury dociąga przepis ustawowy
    kont = [j.id for j in rozszerz(idx, [Trafienie("pnps:s1", 1.0, "test")])]
    assert "fus:art138:ust4" in kont

    # Filtrowanie po aktach w szukaj()
    traf_zas = szukaj(idx, ["zwrot"], uzyj_emb=False, akty=["zas"])
    assert [t.id for t in traf_zas] == ["zas:art66:ust1"]


def test_czysc_id_w_tekscie():
    surowy = "Prawo przysługuje [fus:art65:ust1] oraz (zas:art7 pkt 1), zob. fus:art87:ust1 i [proc1:s2]."
    czysty = czysc_id_w_tekscie(surowy)
    assert "fus:" not in czysty and "zas:" not in czysty and "proc1:" not in czysty
    assert "(art. 65 ust. 1)" in czysty
    assert "(art. 7 pkt 1)" in czysty
    assert "art. 87 ust. 1" in czysty
    assert "(sekcja 2)" in czysty


def test_liczby_z_pytania_i_pochodne_w_korpusie_kontroli():
    poch = liczby_pochodne_z_pytania("Mam 18 lat składkowych i 9 nieskładkowych oraz przypadek 14 lat i 8 lat.")
    # 18 // 3 = 6, 18 + 6 = 24, 18 + 9 = 27; 14 // 3 = 4, (14 % 3)*4 = 8, 14 + 4 = 18
    for oczekiwana in ("6", "24", "27", "4", "8", "18"):
        assert oczekiwana in poch.split()

    k = korpus_liczb([], "Mam 14 lat składkowych i 8 lat nieskładkowych.")
    assert liczby_bez_pokrycia("Okresy nieskładkowe ogranicza się do 4 lat i 8 miesięcy, co daje 18 lat i 8 miesięcy.", k) == []
    assert liczby_bez_pokrycia("Odsetki wynoszą 7% rocznie.", k) == ["7"]


def test_kontrola_dokleja_pytanie_i_pochodne_do_korpusu_liczb():
    idx = _indeks_min(Jednostka(id="fus:art1", akt="FUS", akt_kod="fus", poziom="art", sciezka="Art. 1", tekst="Ogólne."))
    pyt = "Mam 18 lat okresów składkowych i 9 lat nieskładkowych"
    k = _kontrola(idx, pyt, "Po ograniczeniu do 6 lat łączny staż wynosi 24 lata.", [], [], weryfikacja=False)
    assert k["zdania"] == ["Po ograniczeniu do 6 lat łączny staż wynosi 24 lata."]
    assert not any("liczby spoza" in u for u in k["usuniete"])
