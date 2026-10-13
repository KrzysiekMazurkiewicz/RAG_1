import { useCallback, useEffect, useState } from 'react'
import { api, type Stan } from './api'
import Pytanie from './Pytanie'
import Przepisy from './Przepisy'
import Opinie from './Opinie'
import Administracja from './Administracja'

type Zakladka = 'pytanie' | 'przepisy' | 'opinie' | 'administracja'

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
    const t = setInterval(() => void odswiez(), 12000)
    return () => clearInterval(t)
  }, [odswiez])

  const llm = stan?.llm
  const zadanieWToku = stan?.zadanie.status === 'w toku'

  return (
    <div className="app">
      <header className="naglowek">
        <div className="marka">
          <div className="logo-znaczek">ZUS · LexRAG</div>
          <div>
            <h1>LexRAG — Asystent Przepisów i Procedur ZUS</h1>
            <p className="podtytul">
              Wyszukiwarka ustaw i instrukcji wewnętrznych z deterministyczną kontrolą cytatów i weryfikacją zdań
            </p>
          </div>
        </div>

        <div className="status-pasek" title={llm ? llm.baza : ''}>
          {stan ? (
            <>
              <div className="pigulka-statusu">
                <span className={llm?.dostepny && llm.polaczenie?.ok ? 'kropka ok' : 'kropka zle'} />
                {llm?.dostepny ? (
                  llm.polaczenie?.ok ? (
                    <span>
                      Model: <strong>{llm.model}</strong>
                    </span>
                  ) : (
                    <span className="blad">Model {llm.model} offline</span>
                  )
                ) : (
                  <span>Tryb wyszukiwania (bez LLM)</span>
                )}
              </div>
              <div className="pigulka-statusu">
                <strong>{stan.indeks.akty.length}</strong> aktów · <strong>{stan.indeks.szukalnych}</strong> przepisów
              </div>
            </>
          ) : (
            <div className="pigulka-statusu blad">{blad || 'Łączenie z silnikiem…'}</div>
          )}
        </div>
      </header>

      {zadanieWToku && (
        <div className="baner-zadania">
          <div className="spinner maly-spinner" />
          <span>
            <strong>Zadanie w tle ({stan?.zadanie.kod}):</strong> {stan?.zadanie.komunikat}
          </span>
        </div>
      )}

      <nav className="zakladki">
        {(
          [
            ['pytanie', '🔍 Wyszukiwarka i analiza spraw'],
            ['przepisy', `📚 Baza przepisów i procedur (${stan?.indeks.szukalnych ?? 0})`],
            ['opinie', '⚖️ Weryfikacja opinii (Referent)'],
            ['administracja', '⚙️ Administracja i wgrywanie'],
          ] as [Zakladka, string][]
        ).map(([k, n]) => (
          <button key={k} className={zakladka === k ? 'aktywna' : ''} onClick={() => setZakladka(k)}>
            {n}
          </button>
        ))}
      </nav>

      <main>
        {zakladka === 'pytanie' && <Pytanie stan={stan} />}
        {zakladka === 'przepisy' && <Przepisy stan={stan} />}
        {zakladka === 'opinie' && <Opinie />}
        {zakladka === 'administracja' && <Administracja stan={stan} odswiez={odswiez} />}
      </main>

      <footer className="stopka">
        <div>
          <strong>Zasada 99% precyzji:</strong> Tekst przepisów w sekcji „Podstawa prawna” jest wyświetlany bezpośrednio z
          indeksu (model nie może go zmienić). Każda liczba, spójnik i warunek podlegają weryfikacji.
        </div>
        <div>Wersja silnika: {stan?.wersja ?? '—'} · Przetwarzanie lokalne</div>
      </footer>
    </div>
  )
}
