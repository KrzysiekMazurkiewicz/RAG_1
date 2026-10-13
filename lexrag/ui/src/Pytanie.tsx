import { useEffect, useState, type FormEvent } from 'react'
import { api, type Odpowiedz, type RekordHistorii, type Stan } from './api'
import OdpowiedzWidok from './Odpowiedz'

const KATEGORIE_PRZYKLADOW: { nazwa: string; pytania: { etykieta: string; tresc: string }[] }[] = [
  {
    nazwa: 'Emerytury (FUS)',
    pytania: [
      {
        etykieta: 'Wiek emerytalny po 1948 r.',
        tresc: 'od jakiego wieku przysługuje emerytura osobom urodzonym po 1948 roku',
      },
      {
        etykieta: 'Rozwiązanie umowy o pracę (art. 103a)',
        tresc:
          'Mam 65 lat i chcę przejść na emeryturę, ale nadal pracuję na umowę o pracę u tego samego pracodawcy. Czy ZUS zacznie wypłacać mi emeryturę, jeśli nie rozwiążę umowy?',
      },
      {
        etykieta: 'Zwrot nienależnie pobranych (art. 138)',
        tresc: 'za jaki okres wstecz można żądać zwrotu nienależnie pobranych świadczeń',
      },
    ],
  },
  {
    nazwa: 'Kazusy stażowe (limit 1/3 nieskładkowych)',
    pytania: [
      {
        etykieta: 'Mężczyzna: 18 lat składkowych + 9 nieskładkowych',
        tresc:
          'Mężczyzna ukończył 65 lat. Ma 18 lat okresów składkowych i 9 lat okresów nieskładkowych. Wyliczona emerytura jest niższa od kwoty najniższej. Czy spełnia warunek stażowy do podwyższenia emerytury do minimum? Ile lat okresów nieskładkowych można uwzględnić i jaki będzie łączny staż?',
      },
      {
        etykieta: 'Kobieta: 14 lat składkowych + 8 nieskładkowych',
        tresc:
          'Kobieta ukończyła 60 lat, ma 14 lat okresów składkowych i 8 lat okresów nieskładkowych. Jej wyliczona emerytura jest niższa od kwoty najniższej. Czy spełnia warunek stażowy do podwyższenia emerytury do minimum?',
      },
      {
        etykieta: 'Wcześniejsza emerytura (art. 29)',
        tresc:
          'Dzień dobry, pytam o mamę. Urodziła się w 1948 roku. Jak skończyła 55 lat, miała już 30 lat okresów składkowych i nieskładkowych. Przez cały ten czas była zatrudniona na umowie o pracę. Czy mogła wtedy przejść na wcześniejszą emeryturę, mimo że była zdrowa i nie miała orzeczenia o niezdolności do pracy?',
      },
    ],
  },
  {
    nazwa: 'Zasiłki (chorobowy, macierzyński, opiekuńczy)',
    pytania: [
      {
        etykieta: 'Choroba po ustaniu zatrudnienia',
        tresc: 'czy zasiłek chorobowy przysługuje, gdy zachorowałem po ustaniu zatrudnienia',
      },
      {
        etykieta: 'Okres zasiłkowy (182 / 270 dni)',
        tresc: 'jak długo można pobierać zasiłek chorobowy',
      },
      {
        etykieta: 'Świadczenie rehabilitacyjne',
        tresc: 'kiedy przysługuje świadczenie rehabilitacyjne i ile wynosi',
      },
    ],
  },
  {
    nazwa: 'Hasła przeglądowe',
    pytania: [
      { etykieta: 'kapitał początkowy', tresc: 'kapitał początkowy' },
      { etykieta: 'okres zasiłkowy', tresc: 'okres zasiłkowy' },
      { etykieta: 'potrącenia z emerytury', tresc: 'potrącenia z emerytury' },
      { etykieta: 'renta rodzinna', tresc: 'renta rodzinna' },
    ],
  },
]

export default function Pytanie({ stan }: { stan: Stan | null }) {
  const [tresc, setTresc] = useState('')
  const [naDzien, setNaDzien] = useState('')
  const [analiza, setAnaliza] = useState(false)
  const [slad, setSlad] = useState(false)
  const [wybraneAkty, setWybraneAkty] = useState<string[]>([])
  const [czeka, setCzeka] = useState(false)
  const [sekundy, setSekundy] = useState<number | null>(null)
  const [wynik, setWynik] = useState<Odpowiedz | null>(null)
  const [blad, setBlad] = useState('')

  // Kalkulator stażu 1/3 (art. 5 ust. 2 FUS)
  const [pokazKalkulator, setPokazKalkulator] = useState(false)
  const [plec, setPlec] = useState<'K' | 'M'>('M')
  const [lataSkladkowe, setLataSkladkowe] = useState<number>(18)
  const [lataNieskladkowe, setLataNieskladkowe] = useState<number>(9)

  // Historia ostatnich pytań
  const [historia, setHistoria] = useState<RekordHistorii[]>([])
  const [pokazHistorie, setPokazHistorie] = useState(false)

  const odswiezHistorie = async () => {
    try {
      setHistoria(await api.historia(15))
    } catch {
      /* brak historii */
    }
  }

  useEffect(() => {
    void odswiezHistorie()
  }, [])

  const przelaczAkt = (kod: string) => {
    setWybraneAkty((obecne) => (obecne.includes(kod) ? obecne.filter((x) => x !== kod) : [...obecne, kod]))
  }

  const zapytaj = async (e?: FormEvent, pytanieNadpisane?: string) => {
    e?.preventDefault()
    const p = (pytanieNadpisane ?? tresc).trim()
    if (!p || czeka) return
    if (pytanieNadpisane) setTresc(pytanieNadpisane)
    setCzeka(true)
    setBlad('')
    setWynik(null)
    const t0 = Date.now()
    try {
      const odp = await api.pytanie(
        p,
        naDzien || null,
        analiza ? true : null,
        slad ? true : null,
        wybraneAkty.length ? wybraneAkty : null,
      )
      setWynik(odp)
      void odswiezHistorie()
    } catch (err) {
      setBlad((err as Error).message)
    } finally {
      setSekundy((Date.now() - t0) / 1000)
      setCzeka(false)
    }
  }

  // Obliczenia kalkulatora 1/3 (w miesiącach, żeby obsłużyć ułamki lat jak 14 ÷ 3 = 4 lata i 8 mies.)
  const skladkoweMies = Math.max(0, Math.round(lataSkladkowe * 12))
  const nieskladkoweMies = Math.max(0, Math.round(lataNieskladkowe * 12))
  const limitNieskladkowychMies = Math.floor(skladkoweMies / 3)
  const uwzglednioneNieskladkoweMies = Math.min(nieskladkoweMies, limitNieskladkowychMies)
  const lacznyStazMies = skladkoweMies + uwzglednioneNieskladkoweMies
  const wymaganyStazLat = plec === 'K' ? 20 : 25
  const wymaganyWiek = plec === 'K' ? 60 : 65
  const spelniaProg = lacznyStazMies >= wymaganyStazLat * 12

  const formatMies = (m: number) => {
    const l = Math.floor(m / 12)
    const reszta = m % 12
    return reszta > 0 ? `${l} lat i ${reszta} mies.` : `${l} lat`
  }

  const wstawKazusKalkulatora = () => {
    const osoba = plec === 'K' ? `Kobieta ukończyła ${wymaganyWiek} lat` : `Mężczyzna ukończył ${wymaganyWiek} lat`
    const wygenerowane = `${osoba}. Ma ${lataSkladkowe} lat okresów składkowych i ${lataNieskladkowe} lat okresów nieskładkowych. Wyliczona emerytura jest niższa od kwoty najniższej. Czy spełnia warunek stażowy do podwyższenia emerytury do minimum? Ile lat okresów nieskładkowych można uwzględnić i jaki będzie łączny staż?`
    setTresc(wygenerowane)
  }

  const brakIndeksu = stan !== null && stan.indeks.jednostek === 0
  const akty = stan?.indeks.akty ?? []

  return (
    <section>
      <form className="karta formularz" onSubmit={(e) => void zapytaj(e)}>
        {akty.length > 0 && (
          <div className="chipy-aktow">
            <span className="etykieta-chipow">Szukaj w aktach:</span>
            <button
              type="button"
              className={`chip ${wybraneAkty.length === 0 ? 'aktywny' : ''}`}
              onClick={() => setWybraneAkty([])}
            >
              Wszystkie wgrane akty ({akty.length})
            </button>
            {akty.map((a) => (
              <button
                type="button"
                key={a.kod}
                className={`chip ${wybraneAkty.includes(a.kod) ? 'aktywny' : ''}`}
                onClick={() => przelaczAkt(a.kod)}
                title={a.akt}
              >
                <strong>{a.kod.toUpperCase()}</strong> · {a.akt.length > 36 ? a.akt.slice(0, 36) + '…' : a.akt}
              </button>
            ))}
          </div>
        )}

        <label className="pole">
          <div className="wiersz-miedzy">
            <span className="etykieta-pola">Pytanie pracownika lub opis sprawy (kazus)</span>
            <span className="cichy">Skrót: Ctrl/Cmd + Enter</span>
          </div>
          <textarea
            rows={4}
            value={tresc}
            onChange={(e) => setTresc(e.target.value)}
            placeholder="Opisz sytuację klienta (wiek, płeć, okresy składkowe i nieskładkowe, okoliczności sprawy) lub wpisz zagadnienie prawne…"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void zapytaj()
            }}
          />
        </label>

        <div className="wiersz pasek-opcji-pytania">
          <label className="pole male">
            <span>Stan prawny na dzień</span>
            <input type="date" value={naDzien} onChange={(e) => setNaDzien(e.target.value)} />
          </label>
          <label className="pole-wiersz" title="Model wygeneruje dodatkowe warianty zapytania prawniczego i sprawdzi, jakich danych brakuje w pytaniu">
            <input type="checkbox" checked={analiza} onChange={(e) => setAnaliza(e.target.checked)} />
            Głęboka analiza pytania (warianty + dopytanie)
          </label>
          <label
            className="pole-wiersz"
            title="Zapisuje pełny ślad diagnostyczny (prompty, surowe wyjścia modelu, werdykty zdań)"
          >
            <input
              type="checkbox"
              checked={slad || (stan?.slad.wlaczony ?? false)}
              disabled={stan?.slad.wlaczony ?? false}
              onChange={(e) => setSlad(e.target.checked)}
            />
            Zapisz ślad diagnostyczny{stan?.slad.wlaczony ? ` (aktywny, ${stan.slad.dni} dni)` : ''}
          </label>
          <div className="wiersz-prawy">
            <button
              type="button"
              className={`przycisk-narzedzie ${pokazKalkulator ? 'aktywna' : ''}`}
              onClick={() => setPokazKalkulator(!pokazKalkulator)}
            >
              🧮 Kalkulator stażu 1/3 (art. 5 ust. 2)
            </button>
            <button type="submit" className="glowny duzy" disabled={czeka || brakIndeksu || !tresc.trim()}>
              {czeka ? 'Analizuję przepisy…' : 'Wyszukaj i zweryfikuj odpowiedź'}
            </button>
            {sekundy !== null && !czeka && <span className="pigulka-czasu">{sekundy.toFixed(1)} s</span>}
          </div>
        </div>

        {brakIndeksu && (
          <p className="ostrzezenie">Brak wgranych aktów w indeksie — przejdź do zakładki „Administracja”.</p>
        )}

        {pokazKalkulator && (
          <div className="kalkulator-stazu">
            <div className="wiersz-miedzy">
              <strong>Pomocnik obliczania stażu emerytalnego (reguła 1/3 z art. 5 ust. 2 i art. 87 ust. 1 FUS)</strong>
              <span className="cichy">Okresy nieskładkowe uwzględnia się maks. do 1/3 udowodnionych składkowych</span>
            </div>
            <div className="wiersz">
              <label className="pole male">
                <span>Płeć i próg minimalny</span>
                <select value={plec} onChange={(e) => setPlec(e.target.value as 'K' | 'M')}>
                  <option value="K">Kobieta (wiek 60 lat, staż min. 20 lat)</option>
                  <option value="M">Mężczyzna (wiek 65 lat, staż min. 25 lat)</option>
                </select>
              </label>
              <label className="pole male">
                <span>Okresy składkowe (lata)</span>
                <input
                  type="number"
                  min={0}
                  max={60}
                  step={1}
                  value={lataSkladkowe}
                  onChange={(e) => setLataSkladkowe(Number(e.target.value))}
                />
              </label>
              <label className="pole male">
                <span>Okresy nieskładkowe (lata)</span>
                <input
                  type="number"
                  min={0}
                  max={40}
                  step={1}
                  value={lataNieskladkowe}
                  onChange={(e) => setLataNieskladkowe(Number(e.target.value))}
                />
              </label>
              <div className="wynik-kalkulatora">
                <div>
                  Maks. nieskładkowe (1/3 z {lataSkladkowe} lat):{' '}
                  <strong>{formatMies(limitNieskladkowychMies)}</strong> · Uwzględnione:{' '}
                  <strong>{formatMies(uwzglednioneNieskladkoweMies)}</strong>
                </div>
                <div>
                  Łączny staż po limicie: <strong>{formatMies(lacznyStazMies)}</strong> (wymagane {wymaganyStazLat} lat)
                  {' → '}
                  <span className={spelniaProg ? 'ok' : 'blad'}>
                    <strong>{spelniaProg ? 'SPEŁNIA warunek stażowy' : 'NIE SPEŁNIA warunku stażowego'}</strong>
                  </span>
                </div>
              </div>
              <button type="button" className="maly-przycisk" onClick={wstawKazusKalkulatora}>
                Wstaw kazus do pytania
              </button>
            </div>
          </div>
        )}

        <div className="sekcja-przykladow">
          {KATEGORIE_PRZYKLADOW.map((kat) => (
            <div key={kat.nazwa} className="grupa-przykladow">
              <span className="etykieta-kategorii">{kat.nazwa}:</span>
              {kat.pytania.map((p, i) => (
                <button
                  type="button"
                  key={i}
                  className="chip maly"
                  onClick={() => setTresc(p.tresc)}
                  title={p.tresc}
                >
                  {p.etykieta}
                </button>
              ))}
            </div>
          ))}
        </div>
      </form>

      {czeka && (
        <div className="karta ladowanie-karta">
          <div className="spinner" />
          <div>
            <strong>Trwa wyszukiwanie w indeksie i wieloetapowa weryfikacja odpowiedzi…</strong>
            <p className="cichy">
              Krok 1: Hybrydowe wyszukiwanie (BM25 + wektory + reranker + kotwice ustawowe) → Krok 2: Generowanie
              odpowiedzi z cytatami → Krok 3: Bramki deterministyczne i niezależna weryfikacja każdego zdania.
            </p>
          </div>
        </div>
      )}

      {blad && <div className="karta blad">Błąd: {blad}</div>}
      {wynik && <OdpowiedzWidok w={wynik} />}

      {historia.length > 0 && (
        <div className="karta">
          <div className="wiersz-miedzy">
            <h3 style={{ margin: 0 }}>Ostatnie pytania w sesji ({historia.length})</h3>
            <button type="button" className="maly-przycisk" onClick={() => setPokazHistorie(!pokazHistorie)}>
              {pokazHistorie ? 'Ukryj historię' : 'Pokaż historię zapytań'}
            </button>
          </div>
          {pokazHistorie && (
            <div className="lista-historii">
              {historia.map((h) => (
                <div key={h.id} className="wiersz-historii">
                  <div className="wiersz-miedzy">
                    <button
                      type="button"
                      className="link-historia"
                      onClick={() => void zapytaj(undefined, h.pytanie)}
                      title="Zadaj to pytanie ponownie"
                    >
                      {h.pytanie}
                    </button>
                    <div className="wiersz">
                      {h.status_weryfikacji && (
                        <span className={`znacznik ${h.status_weryfikacji}`}>{h.status_weryfikacji}</span>
                      )}
                      <span className="cichy">{h.czas.replace('T', ' ')}</span>
                    </div>
                  </div>
                  {h.tekst && <p className="cichy skrot-odpowiedzi">{h.tekst}</p>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
