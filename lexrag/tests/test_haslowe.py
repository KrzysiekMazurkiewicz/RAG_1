"""Pytania hasłowe: wykrywanie i alternatywy id w złotym zbiorze."""
from lexrag.evaluate import WynikPytania
from lexrag.generate import PROMPT_PRZEGLAD, pytanie_haslowe


def test_pytanie_haslowe_wykrywa_krotkie_bez_zaimka():
    assert pytanie_haslowe("wcześniejsza emerytura")
    assert pytanie_haslowe("zasiłek po ustaniu zatrudnienia")
    assert pytanie_haslowe("renta z tytułu niezdolności do pracy")
    assert not pytanie_haslowe("ile wynosi zasiłek pogrzebowy")
    assert not pytanie_haslowe("wcześniejsza emerytura?")
    assert not pytanie_haslowe("kobieta urodzona w 1948 r. z 30-letnim stażem, czy może przejść na emeryturę")
    assert not pytanie_haslowe("")


def test_prompt_przegladowy_zada_regul_i_dopytania():
    assert "PRZEGLĄDOWO" in PROMPT_PRZEGLAD and "reguły ogólnej" in PROMPT_PRZEGLAD


def test_alternatywy_id_w_oczekiwanych():
    w = WynikPytania("q", ["fus:art29:ust1|fus:art184:ust1", "fus:art1"], ["fus:art184:ust1", "fus:art2", "fus:art1"])
    assert w.pozycje() == [1, 3]
    assert WynikPytania("q", ["a|b"], ["c"]).pozycje() == [None]
