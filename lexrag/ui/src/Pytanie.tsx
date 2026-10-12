import { useState, type FormEvent } from 'react'
import { api, type Odpowiedz, type Stan } from './api'
import OdpowiedzWidok from './Odpowiedz'

const PRZYKLADY = [
  'od jakiego wieku przysługuje emerytura osobom urodzonym po 1948 roku',
  'czy zasiłek chorobowy przysługuje, gdy zachorowałem po ustaniu zatrudnienia',
  'za jaki okres wstecz można żądać zwrotu nienależnie pobranych świadczeń',
  'wcześniejsza emerytura',
]

export default function Pytanie({ stan }: { stan: Stan | null }) {
  const [tresc, setTresc] = useState('')
  const [naDzien, setNaDzien] = useState('')
  const [analiza, setAnaliza] = useState(false)
  const [slad, setSlad] = useState(false)
  const [czeka, setCzeka] = useState(false)
  const [sekundy, setSekundy] = useState<number | null>(null)
  const [wynik, setWynik] = useState<Odpowiedz | null>(null)
  const [blad, setBlad] = useState('')

  const zapytaj = async (e?: FormEvent) => {
    e?.preventDefault()
    const p = tresc.trim()
    if (!p || czeka) return
    setCzeka(true)
    setBlad('')
    setWynik(null)
    const t0 = Date.now()
    try {
      setWynik(await api.pytanie(p, naDzien || null, analiza ? true : null, slad ? true : null))
    } catch (err) {
      setBlad((err as Error).message)
    } finally {
      setSekundy((Date.now() - t0) / 1000)
      setCzeka(false)
    }
  }

  const brakIndeksu = stan !== null && stan.indeks.jednostek === 0

  return (
    <section>
      <form className="karta formularz" onSubmit={zapytaj}>
        <label className="pole">
          <span>Pytanie</span>
          <textarea
            rows={3}
            value={tresc}
            onChange={(e) => setTresc(e.target.value)}
            placeholder="Opisz sytuację: kto, co, kiedy. Pełne zdanie trafia lepiej niż hasło."
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) void zapytaj()
            }}
          />
        </label>
        <div className="wiersz">
          <label className="pole male">
            <span>Stan prawny na dzień</span>
            <input type="date" value={naDzien} onChange={(e) => setNaDzien(e.target.value)} />
          </label>
          <label className="pole-wiersz">
            <input type="checkbox" checked={analiza} onChange={(e) => setAnaliza(e.target.checked)} />
            analiza pytania (warianty, dopytanie; jedno wywołanie modelu więcej)
          </label>
          <label className="pole-wiersz" title="Zapisuje pełną treść pytania, wszystkie prompty i surowe odpowiedzi modelu. Dla referenta i administratora; pliki są usuwane po ustalonym czasie.">
            <input type="checkbox" checked={slad || (stan?.slad.wlaczony ?? false)} disabled={stan?.slad.wlaczony ?? false} onChange={(e) => setSlad(e.target.checked)} />
            zapisz ślad diagnostyczny tej odpowiedzi{stan?.slad.wlaczony ? ` (włączony dla wszystkich, ${stan.slad.dni} dni)` : ''}
          </label>
          <button type="submit" className="glowny" disabled={czeka || brakIndeksu || !tresc.trim()}>
            {czeka ? 'liczę…' : 'Zapytaj'}
          </button>
          {sekundy !== null && !czeka && <span className="cichy">{sekundy.toFixed(1)} s</span>}
        </div>
        {brakIndeksu && <p className="ostrzezenie">Brak wgranych aktów: zakładka „Administracja”.</p>}
        <p className="cichy">
          Przykłady:{' '}
          {PRZYKLADY.map((p, i) => (
            <button type="button" key={i} className="link" onClick={() => setTresc(p)}>
              {p}
            </button>
          ))}
        </p>
      </form>

      {czeka && (
        <div className="karta cichy">
          Szukam przepisów i piszę odpowiedź. Model odpowiada, potem sprawdza każde zdanie; zwykle 5 do 25 s, pierwsze
          pytanie po przerwie dłużej (model ładuje się do pamięci karty).
        </div>
      )}
      {blad && <div className="karta blad">Błąd: {blad}</div>}
      {wynik && <OdpowiedzWidok w={wynik} />}
    </section>
  )
}
