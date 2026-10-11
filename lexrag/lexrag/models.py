"""Model danych: jedna jednostka indeksu (ustęp ustawy, artykuł bez ustępów,
definicja ustawowa, sekcja procedury)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional


@dataclass
class Jednostka:
    id: str                      # np. "fus:art24:ust1", "fus:art4:pkt2", "proc1:s12"
    akt: str                     # pełna nazwa aktu / dokumentu
    akt_kod: str                 # krótki prefiks, np. "fus", "proc1"
    poziom: str                  # "art" | "ust" | "pkt-def" | "sekcja" | "fragment"
    sciezka: str                 # "Dział II Świadczenie podstawowe > Rozdział 1 ... > Art. 24 > ust. 1"
    tekst: str                   # pełny tekst jednostki (pkt/lit w środku, oddzielone \n)
    opis: str = ""               # jedno zdanie kontekstu (contextual retrieval) – generowane przez LLM
    rodzic: Optional[str] = None # id artykułu / sekcji nadrzędnej (do rozszerzania kontekstu)
    odwolania: list[str] = field(default_factory=list)       # id jednostek, do których odsyła tekst
    odwolania_zewn: list[str] = field(default_factory=list)  # odesłania do innych aktów / nierozwiązane
    terminy: list[str] = field(default_factory=list)         # terminy zdefiniowane (dla pkt-def)
    obowiazuje_od: Optional[str] = None  # ISO data; None = bez ograniczenia
    obowiazuje_do: Optional[str] = None
    szukalny: bool = True        # czy jednostka trafia do indeksu (rodzice z ustępami: nie)

    # --- serializacja -----------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Jednostka":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    # --- pomocnicze -------------------------------------------------------
    def tekst_do_indeksu(self) -> str:
        """Tekst, z którego liczone są embeddingi i BM25: ścieżka + opis + treść."""
        czesci = [self.sciezka]
        if self.opis:
            czesci.append(self.opis)
        czesci.append(self.tekst)
        return "\n".join(c for c in czesci if c)

    def obowiazuje(self, dzien: Optional[str]) -> bool:
        """Czy jednostka obowiązuje w danym dniu (ISO 'YYYY-MM-DD'). None = zawsze."""
        if not dzien:
            return True
        if self.obowiazuje_od and dzien < self.obowiazuje_od:
            return False
        if self.obowiazuje_do and dzien > self.obowiazuje_do:
            return False
        return True


def zapisz_jsonl(jednostki: Iterable[Jednostka], sciezka: str) -> int:
    n = 0
    with open(sciezka, "w", encoding="utf-8") as f:
        for j in jednostki:
            f.write(j.to_json() + "\n")
            n += 1
    return n


def wczytaj_jsonl(sciezka: str) -> list[Jednostka]:
    out: list[Jednostka] = []
    with open(sciezka, encoding="utf-8") as f:
        for linia in f:
            linia = linia.strip()
            if linia:
                out.append(Jednostka.from_dict(json.loads(linia)))
    return out
