import { useCallback, useEffect, useState } from 'react'
import { api, type Stan } from './api'
import Pytanie from './Pytanie'
import Opinie from './Opinie'
import Administracja from './Administracja'

type Zakladka = 'pytanie' | 'opinie' | 'administracja'

export default function App() {
  const [zakladka, setZakladka] = useState<Zakladka>('pytanie')
  const [stan, setStan] = useState<Stan | null>(null)
  const [blad, setBlad] = useState('')

  const odswiez = useCallback(async () => {
    try {
      setStan(await api.stan())
      setBlad('')
    } catch (e) {
      setBlad(`Silnik nie odpowiada: ${(e as Error).message}`)
    }
  }, [])

  useEffect(() => {
    void odswiez()
    const t = setInterval(() => void odswiez(), 15000)
    return () => clearInterval(t)
  }, [odswiez])

  const llm = stan?.llm
  return (
    <div className="app">
      <header className="naglowek">
        <div>
          <h1>LexRAG</h1>
          <p className="podtytul">pytania do ustaw z cytowaniem przepisów, model on-premise</p>
        </div>
        <div className="stan-skrot" title={llm ? llm.baza : ''}>
          {stan ? (
            <>
              <span className={llm?.dostepny && llm.polaczenie?.ok ? 'kropka ok' : 'kropka zle'} />
              {llm?.dostepny
                ? llm.polaczenie?.ok
                  ? `model ${llm.model}`
                  : `model ${llm.model} nie odpowiada (Administracja → Serwer modelu)`
                : 'bez modelu (tylko wyszukiwanie)'}{' '}
              · {stan.indeks.akty.length} akt.,{' '}
              {stan.indeks.szukalnych} przepisów
            </>
          ) : (
            blad || 'łączę…'
          )}
        </div>
      </header>

      <nav className="zakladki">
        {(
          [
            ['pytanie', 'Pytanie'],
            ['opinie', 'Opinie (referent)'],
            ['administracja', 'Administracja'],
          ] as [Zakladka, string][]
        ).map(([k, n]) => (
          <button key={k} className={zakladka === k ? 'aktywna' : ''} onClick={() => setZakladka(k)}>
            {n}
          </button>
        ))}
      </nav>

      <main>
        {zakladka === 'pytanie' && <Pytanie stan={stan} />}
        {zakladka === 'opinie' && <Opinie />}
        {zakladka === 'administracja' && <Administracja stan={stan} odswiez={odswiez} />}
      </main>

      <footer className="stopka">
        Odpowiedź czytaj razem z cytatem: tekst przepisu pochodzi z indeksu, nie od modelu. Odmowa to sygnał „sprawa dla
        referenta”, nie awaria.
      </footer>
    </div>
  )
}
