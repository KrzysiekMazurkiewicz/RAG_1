import { useCallback, useEffect, useState } from 'react'
import { api, type Jednostka, type Stan } from './api'

export default function Przepisy({ stan }: { stan: Stan | null }) {
  const [q, setQ] = useState('')
  const [wybranyAkt, setWybranyAkt] = useState('')
  const [wyniki, setWyniki] = useState<Jednostka[]>([])
  const [ladowanie, setLadowanie] = useState(false)
  const [blad, setBlad] = useState('')
  const [podglad, setPodglad] = useState<Jednostka | null>(null)
  const [skopiowano, setSkopiowano] = useState('')

  const szukaj = useCallback(async (zapytanie: string, akt: string) => {
    setLadowanie(true)
    setBlad('')
    try {
      const lista = await api.przepisy(zapytanie, akt, 60)
      setWyniki(lista)
    } catch (e) {
      setBlad((e as Error).message)
    } finally {
      setLadowanie(false)
    }
  }, [])

  useEffect(() => {
    const t = setTimeout(() => {
      void szukaj(q, wybranyAkt)
    }, 180)
    return () => clearTimeout(t)
  }, [q, wybranyAkt, szukaj])

  const otworzJednostke = async (ident: string) => {
    try {
      const j = await api.jednostka(ident)
      setPodglad(j)
    } catch {
      setQ(ident.replace(/^.*:/, ''))
    }
  }

  const kopiuj = (j: Jednostka) => {
    void navigator.clipboard?.writeText(`${j.sciezka} (${j.akt})\n\n${j.tekst}`)
    setSkopiowano(j.id)
    setTimeout(() => setSkopiowano(''), 2000)
  }

  const akty = stan?.indeks.akty ?? []

  return (
    <section className="sekcja-przepisy">
      <div className="karta">
        <div className="naglowek-sekcji">
          <div>
            <h2>Baza przepisów i procedur wewnętrznych</h2>
            <p className="cichy">
              Przeglądaj pełne brzmienie artykułów ustaw i instrukcji ZUS z indeksu. Wpisz numer artykułu (np.{' '}
              <code>art. 24</code>, <code>art. 138</code>, <code>art. 5 ust. 2</code>) albo szukaną frazę.
            </p>
          </div>
          <span className="znacznik-liczba">{wyniki.length} wyników</span>
        </div>

        <div className="pasek-filtrow">
          <div className="wyszukiwarka-input">
            <input
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Szukaj po numerze artykułu (np. art. 24, art. 87, art. 138) lub słowach kluczowych…"
            />
            {q && (
              <button type="button" className="przycisk-czysty" onClick={() => setQ('')} title="Wyczyść">
                ✕
              </button>
            )}
          </div>
        </div>

        {akty.length > 0 && (
          <div className="chipy-aktow">
            <span className="etykieta-chipow">Źródło:</span>
            <button
              type="button"
              className={`chip ${wybranyAkt === '' ? 'aktywny' : ''}`}
              onClick={() => setWybranyAkt('')}
            >
              Wszystkie ({stan?.indeks.szukalnych ?? 0})
            </button>
            {akty.map((a) => (
              <button
                type="button"
                key={a.kod}
                className={`chip ${wybranyAkt === a.kod ? 'aktywny' : ''}`}
                onClick={() => setWybranyAkt(wybranyAkt === a.kod ? '' : a.kod)}
                title={a.akt}
              >
                <strong>{a.kod.toUpperCase()}</strong> · {a.akt.length > 42 ? a.akt.slice(0, 42) + '…' : a.akt} (
                {a.szukalnych})
              </button>
            ))}
          </div>
        )}

        <div className="szybkie-skroty">
          <span className="cichy">Szybki skok:</span>
          {['Art. 5', 'Art. 24', 'Art. 29', 'Art. 57', 'Art. 87', 'Art. 103a', 'Art. 116', 'Art. 138', 'Art. 173'].map(
            (skrot) => (
              <button key={skrot} type="button" className="chip maly" onClick={() => setQ(skrot)}>
                {skrot}
              </button>
            ),
          )}
        </div>
      </div>

      {podglad && (
        <div className="karta podglad-jednostki">
          <div className="wiersz-miedzy">
            <div>
              <span className="id">{podglad.id}</span> <strong>{podglad.sciezka}</strong>
              <span className="cichy"> · {podglad.akt}</span>
            </div>
            <div className="wiersz">
              <button type="button" className="maly-przycisk" onClick={() => kopiuj(podglad)}>
                {skopiowano === podglad.id ? '✓ Skopiowano' : 'Kopiuj treść'}
              </button>
              <button type="button" className="maly-przycisk" onClick={() => setPodglad(null)}>
                Zamknij podgląd
              </button>
            </div>
          </div>
          {podglad.opis && <p className="opis-kontekstowy">💡 {podglad.opis}</p>}
          <pre className="tekst-przepisu-blok">{podglad.tekst}</pre>
        </div>
      )}

      {blad && <div className="karta blad">Błąd: {blad}</div>}
      {ladowanie && <div className="karta cichy">Wyszukiwanie w indeksie…</div>}

      <div className="lista-przepisow">
        {wyniki.map((j) => (
          <article key={j.id} className="karta karta-przepisu">
            <div className="wiersz-miedzy">
              <div className="naglowek-przepisu">
                <span className="id">{j.id}</span>
                <span className="sciezka-przepisu">{j.sciezka}</span>
              </div>
              <div className="wiersz">
                {j.rodzic && (
                  <button
                    type="button"
                    className="maly-przycisk"
                    onClick={() => void otworzJednostke(j.rodzic!)}
                    title="Pokaż cały artykuł / sekcję nadrzędną"
                  >
                    Cały artykuł ({j.rodzic})
                  </button>
                )}
                <button type="button" className="maly-przycisk" onClick={() => kopiuj(j)}>
                  {skopiowano === j.id ? '✓ Skopiowano' : 'Kopiuj'}
                </button>
              </div>
            </div>

            <div className="meta-przepisu cichy">
              <span>{j.akt}</span>
              {j.wersja && <span> · {j.wersja}</span>}
              {(j.obowiazuje_od || j.obowiazuje_do) && (
                <span>
                  {' '}
                  · obowiązuje: {j.obowiazuje_od ?? '…'} – {j.obowiazuje_do ?? 'nadal'}
                </span>
              )}
            </div>

            {j.opis && <p className="opis-kontekstowy">💡 {j.opis}</p>}

            <p className="tekst-przepisu">{j.tekst}</p>

            {((j.odwolania && j.odwolania.length > 0) || (j.odwolania_zewn && j.odwolania_zewn.length > 0)) && (
              <div className="wiersz odwolania-wiersz">
                <span className="cichy">Powiązane przepisy:</span>
                {j.odwolania?.map((o) => (
                  <button
                    key={o}
                    type="button"
                    className="chip maly"
                    onClick={() => void otworzJednostke(o)}
                    title="Otwórz powiązany przepis"
                  >
                    ↪ {o}
                  </button>
                ))}
                {j.odwolania_zewn?.map((oz, idx) => (
                  <span key={idx} className="cichy">
                    [{oz}]
                  </span>
                ))}
              </div>
            )}
          </article>
        ))}

        {!ladowanie && wyniki.length === 0 && (
          <div className="karta cichy">Brak przepisów pasujących do podanego filtra.</div>
        )}
      </div>
    </section>
  )
}
