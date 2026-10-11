"""Tokenizacja i lematyzacja po polsku dla BM25.

Polska fleksja ("emerytura / emerytury / emeryturze") bez lematyzacji
zabija wyszukiwanie leksykalne, więc każdy token sprowadzamy do lematu
biblioteką `simplemma` (czysty Python, słownik w pakiecie, bez pobierania
modeli). Jeśli jej nie ma – degradacja do zwykłego lowercase.
"""
from __future__ import annotations

import re

try:
    import simplemma  # type: ignore
    _LEM = True
except ImportError:  # pragma: no cover
    _LEM = False

_SLOWO = re.compile(r"\w+", re.UNICODE)

# krótka lista stopów – celowo BEZ "nie", które w prawie zmienia sens
STOPY = set(
    "i w z na do o od oraz lub albo się jest są przez dla za po ze że ten ta to tej tego "
    "tych tym u a w z ze o oraz który która które których którym".split()
)


def lematyzuj(slowo: str) -> str:
    if _LEM and slowo.isalpha():
        try:
            return simplemma.lemmatize(slowo, lang="pl")
        except Exception:  # pragma: no cover
            return slowo
    return slowo


def tokenizuj(tekst: str) -> list[str]:
    out: list[str] = []
    for t in _SLOWO.findall(tekst.lower()):
        if t in STOPY or len(t) == 1 and not t.isdigit():
            continue
        out.append(lematyzuj(t))
    return out


def lematyzacja_dostepna() -> bool:
    return _LEM
