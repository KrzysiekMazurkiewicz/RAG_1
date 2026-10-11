"""Kontrola odpowiedzi: podział na zdania, deterministyczna kontrola liczb, JSON z <think>."""
from lexrag import llm
from lexrag.generate import _zdania, korpus_liczb, liczby_bez_pokrycia
from lexrag.models import Jednostka


def test_zdania_nie_lamia_po_skrotach():
    z = _zdania("Zgodnie z art. 138 ust. 4 ustawy zwrotu żąda się za 3 lata (Dz. U. z 2024 r. poz. 986). "
                "Wyjątek: 12 miesięcy. Zob. np. ust. 5.")
    assert z == ["Zgodnie z art. 138 ust. 4 ustawy zwrotu żąda się za 3 lata (Dz. U. z 2024 r. poz. 986).",
                 "Wyjątek: 12 miesięcy.", "Zob. np. ust. 5."]


def test_kontrola_liczb():
    k = korpus_liczb([Jednostka(id="t:art138:ust4", akt="T", akt_kod="t", poziom="ust", sciezka="Art. 138 > ust. 4",
                                tekst="Nie można żądać zwrotu za okres dłuższy niż 12 miesięcy, w pozostałych wypadkach "
                                      "– za okres dłuższy niż 3 lata. Zasiłek wynosi 4 000 zł.")])
    assert liczby_bez_pokrycia("Zwrot za 3 lata albo 12 miesięcy (art. 138 ust. 4).", k) == []
    assert liczby_bez_pokrycia("Zasiłek wynosi 4000 zł, czyli trzech lat.", k) == []
    assert liczby_bez_pokrycia("Odsetki wynoszą 7% rocznie, a termin to 36 miesięcy.", k) == ["7", "36"]


def test_json_z_think_i_ogrodzeniem():
    assert llm.wyciagnij_json('<think>rozważam…</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.wyciagnij_json('[1, 2] nie jest obiektem, ale {"b": 2} jest') == {"b": 2}


def test_powtorzenie_pytania_odpada():
    from lexrag.generate import powtarza_pytanie
    pyt = "z ilu lat bierze się zarobki do podstawy wymiaru emerytury"
    assert powtarza_pytanie("Z ilu lat bierze się zarobki do podstawy wymiaru emerytury?", pyt)
    assert powtarza_pytanie("Zarobki do podstawy wymiaru emerytury bierze się z ilu lat.", pyt)
    assert not powtarza_pytanie("Podstawę wymiaru stanowi przeciętna podstawa wymiaru składki z kolejnych 10 lat "
                                "wybranych z ostatnich 20 lat (art. 15 ust. 1).", pyt)
    assert not powtarza_pytanie("Tak, emerytura przysługuje osobom urodzonym po 1948 roku.",
                                "od jakiego wieku przysługuje emerytura osobom urodzonym po 1948 roku")


def test_odzysk_cytatow_z_adresow():
    from lexrag.generate import odzyskaj_cytaty
    def j(ident):
        return Jednostka(id=ident, akt="T", akt_kod="fus", poziom="ust", sciezka=ident, tekst="x")
    kontekst = [j("fus:art58"), j("fus:art58:ust1"), j("fus:art58:ust2"), j("fus:art103a"), j("fus:art13:ust3a")]
    # poprawne id mają pierwszeństwo, śmieci odpadają
    assert odzyskaj_cytaty("cokolwiek art. 13 ust. 3a", ["fus:art58:ust2", "fus:nieistniejacy"], kontekst) == ["fus:art58:ust2"]
    # adresy w cytowane_id zamiast id
    assert odzyskaj_cytaty("", ["art. 58 ust. 2", "art. 103a"], kontekst) == ["fus:art58:ust2", "fus:art103a"]
    # brak id → adresy z treści; ustęp spoza kontekstu → tylko jednostka artykułowa, nie inny ustęp
    assert odzyskaj_cytaty("Zgodnie z art. 58 ust. 7 oraz art. 13 ust. 3a ustawy.", [], kontekst) == ["fus:art58", "fus:art13:ust3a"]
    # „§ 58” bez ustępu → jednostka artykułowa; id wpisane w tekście też się liczy
    assert set(odzyskaj_cytaty("Patrz § 58 i fus:art103a.", [], kontekst)) == {"fus:art58", "fus:art103a"}
    assert odzyskaj_cytaty("Nic konkretnego.", [], kontekst) == []
