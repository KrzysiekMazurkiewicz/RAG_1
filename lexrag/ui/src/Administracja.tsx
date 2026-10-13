import { useEffect, useState, type FormEvent } from 'react'
import { api, type Stan } from './api'

const PRESETY_MODELI = [
  { etykieta: 'Lokalna Ollama: Gemma 4 12B (Rekomendowany 100%)', baza: 'http://localhost:11434/v1', model: 'gemma4:12b', api: 'ollama' },
  { etykieta: 'Lokalna Ollama: Gemma 4 E4B (Laptop 16 GB / 6 GB VRAM)', baza: 'http://localhost:11434/v1', model: 'gemma4:e4b', api: 'ollama' },
  { etykieta: 'Lokalna Ollama: Qwen3 8B', baza: 'http://localhost:11434/v1', model: 'qwen3:8b', api: 'ollama' },
]

export default function Administracja({ stan, odswiez }: { stan: Stan | null; odswiez: () => Promise<void> }) {
  const [komunikat, setKomunikat] = useState('')
  const [wysylam, setWysylam] = useState(false)
  const [zOpisami, setZOpisami] = useState(false)
  const [adres, setAdres] = useState('')
  const [model, setModel] = useState('')
  const [api2, setApi2] = useState('auto')
  const [wynikModelu, setWynikModelu] = useState('')
  const [zapisuje, setZapisuje] = useState(false)

  useEffect(() => {
    if (stan && !adres && !model) {
      setAdres(stan.llm.baza)
      setModel(stan.llm.model)
    }
  }, [stan, adres, model])

  const zapiszModel = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setZapisuje(true)
    setWynikModelu('')
    try {
      const u = await api.ustawModel({ base_url: adres.trim(), model: model.trim(), api: api2 })
      setWynikModelu((u.polaczenie.ok ? 'OK: ' : 'Błąd: ') + u.polaczenie.komunikat)
      await odswiez()
    } catch (err) {
      setWynikModelu(`Błąd: ${(err as Error).message}`)
    } finally {
      setZapisuje(false)
    }
  }

  const wToku = stan?.zadanie.status === 'w toku'
  const jestModel = stan?.llm.dostepny ?? false

  useEffect(() => {
    if (!wToku) return
    const t = setInterval(() => void odswiez(), 2000)
    return () => clearInterval(t)
  }, [wToku, odswiez])

  const wyslij = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const dane = new FormData(e.currentTarget)
    setWysylam(true)
    setKomunikat('')
    try {
      await api.wgraj(dane)
      setKomunikat(
        'Przyjęto plik: parsowanie jednostek i budowa wspólnego indeksu trwają w tle (postęp widoczny na pasku zadań).',
      )
      await odswiez()
    } catch (err) {
      setKomunikat(`Błąd: ${(err as Error).message}`)
    } finally {
      setWysylam(false)
    }
  }

  const opisz = async (kod: string) => {
    setKomunikat('')
    try {
      await api.opisz(kod)
      setKomunikat(
        `Model generuje opisy kontekstowe dla aktu ${kod}; postęp widoczny w polu „Zadanie”. Po zakończeniu indeks przebuduje się automatycznie.`,
      )
      await odswiez()
    } catch (err) {
      setKomunikat(`Błąd: ${(err as Error).message}`)
    }
  }

  return (
    <section>
      <div className="siatka-kart">
        <div className="karta">
          <h3>Stan silnika i indeksu</h3>
          {stan ? (
            <table className="tabela">
              <tbody>
                <tr>
                  <th>Wersja silnika</th>
                  <td>{stan.wersja}</td>
                </tr>
                <tr>
                  <th>Serwer modelu LLM</th>
                  <td>
                    <strong>{stan.llm.model || '(brak: tryb retrieval-only)'}</strong> · <code>{stan.llm.baza}</code>
                    <br />
                    <span className={stan.llm.polaczenie?.ok ? 'ok' : 'blad'}>
                      {stan.llm.polaczenie?.ok
                        ? `✓ Połączono (${stan.llm.polaczenie.komunikat})`
                        : `✕ Brak połączenia: ${stan.llm.polaczenie?.komunikat ?? ''}`}
                    </span>
                  </td>
                </tr>
                <tr>
                  <th>Wspólny indeks</th>
                  <td>
                    <strong>{stan.indeks.szukalnych}</strong> jednostek szukalnych ({stan.indeks.jednostek} łącznie z
                    artykułami nadrzędnymi) · Embeddingi wektorowe: <strong>{stan.indeks.embeddingi ? 'TAK' : 'NIE'}</strong>
                  </td>
                </tr>
                <tr>
                  <th>Zadanie w tle</th>
                  <td>
                    <span className={`znacznik ${stan.zadanie.status === 'blad' ? 'blad' : 'odpowiedz'}`}>
                      {stan.zadanie.status}
                    </span>
                    {stan.zadanie.komunikat ? ` · ${stan.zadanie.komunikat}` : ''}
                  </td>
                </tr>
              </tbody>
            </table>
          ) : (
            <p className="cichy">Łączenie z silnikiem…</p>
          )}
          {komunikat && <p className={komunikat.startsWith('Błąd') ? 'blad' : 'ok'}>{komunikat}</p>}
        </div>

        <form className="karta formularz" onSubmit={zapiszModel}>
          <h3>Konfiguracja serwera modelu (Ollama / vLLM / Tunel Colab)</h3>
          <div className="szybkie-skroty">
            <span className="cichy">Szybkie presety:</span>
            {PRESETY_MODELI.map((p, idx) => (
              <button
                key={idx}
                type="button"
                className="chip maly"
                onClick={() => {
                  setAdres(p.baza)
                  setModel(p.model)
                  setApi2(p.api)
                }}
              >
                {p.etykieta}
              </button>
            ))}
          </div>
          <div className="wiersz">
            <label className="pole">
              <span>Adres serwera (Ollama: http://localhost:11434/v1; Colab: https://….trycloudflare.com/v1)</span>
              <input value={adres} onChange={(e) => setAdres(e.target.value)} placeholder="http://localhost:11434/v1" />
            </label>
            <label className="pole male">
              <span>Nazwa modelu</span>
              <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="gemma4:12b" />
            </label>
            <label className="pole male">
              <span>Tryb API</span>
              <select value={api2} onChange={(e) => setApi2(e.target.value)}>
                <option value="auto">auto</option>
                <option value="ollama">ollama (natywne)</option>
                <option value="openai">openai (vLLM, llama.cpp)</option>
              </select>
            </label>
            <button type="submit" className="glowny" disabled={zapisuje || !adres.trim() || !model.trim()}>
              {zapisuje ? 'Sprawdzam…' : 'Zapisz i przetestuj'}
            </button>
          </div>
          {wynikModelu && <p className={wynikModelu.startsWith('Błąd') ? 'blad' : 'ok'}>{wynikModelu}</p>}
          <p className="cichy">
            Zmiana działa natychmiast bez restartu serwera i zapisuje się w <code>data/model.json</code>. Jeśli używasz
            notatnika <code>colab/lexrag_serwer_modelu.ipynb</code>, wklej tutaj wygenerowany link{' '}
            <code>https://....trycloudflare.com/v1</code>.
          </p>
        </form>
      </div>

      <div className="karta">
        <h3>Wgrane ustawy, rozporządzenia i procedury wewnętrzne ZUS</h3>
        {stan && stan.indeks.akty.length > 0 ? (
          <table className="tabela">
            <thead>
              <tr>
                <th>Kod</th>
                <th>Pełna nazwa aktu / procedury</th>
                <th>Wersja / Dz.U.</th>
                <th>Jednostek szukalnych</th>
                <th>Opisy kontekstowe</th>
                <th>Obowiązuje od</th>
                <th>Obowiązuje do</th>
                <th>Akcje</th>
              </tr>
            </thead>
            <tbody>
              {stan.indeks.akty.map((a) => {
                const proc = a.szukalnych > 0 ? Math.round((a.opisy / a.szukalnych) * 100) : 0
                return (
                  <tr key={a.kod}>
                    <td>
                      <span className="id">{a.kod}</span>
                    </td>
                    <td>
                      <strong>{a.akt}</strong>
                    </td>
                    <td>{a.wersja || '—'}</td>
                    <td>{a.szukalnych}</td>
                    <td>
                      <div className="postep-opisow">
                        <span>
                          {a.opisy} / {a.szukalnych} ({proc}%)
                        </span>
                        <div className="pasek-postepu">
                          <div className="pasek-postepu-wypelnienie" style={{ width: `${proc}%` }} />
                        </div>
                      </div>
                    </td>
                    <td>{a.obowiazuje_od ?? '—'}</td>
                    <td>{a.obowiazuje_do ?? 'nadal'}</td>
                    <td>
                      {a.opisy < a.szukalnych && (
                        <button
                          type="button"
                          className="maly-przycisk"
                          onClick={() => void opisz(a.kod)}
                          disabled={wToku || !jestModel}
                          title={
                            jestModel
                              ? 'Model dopisze jedno zdanie potocznego opisu do każdego przepisu bez opisu'
                              : 'Wymaga połączonego modelu LLM'
                          }
                        >
                          Generuj opisy
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        ) : (
          <p className="cichy">Brak wgranych aktów w indeksie.</p>
        )}
      </div>

      <form className="karta formularz" onSubmit={wyslij}>
        <h3>Wgraj nową ustawę (PDF z ISAP) lub wewnętrzną procedurę ZUS (DOCX / Markdown / PDF)</h3>
        <div className="wiersz">
          <label className="pole">
            <span>Plik źródłowy (PDF, DOCX, MD, TXT, HTML)</span>
            <input type="file" name="plik" required accept=".pdf,.txt,.html,.htm,.docx,.md" />
          </label>
          <label className="pole male">
            <span>Krótki kod identyfikujący (np. fus, zas, proc_nps)</span>
            <input name="kod" required pattern="[a-z0-9_]{1,24}" placeholder="małe litery, cyfry, _" />
          </label>
          <label className="pole male">
            <span>Rodzaj dokumentu</span>
            <select name="typ" defaultValue="ustawa">
              <option value="ustawa">Ustawa / rozporządzenie (Art., §)</option>
              <option value="dokument">Procedura wewnętrzna / instrukcja ZUS (nagłówki)</option>
            </select>
          </label>
        </div>
        <label className="pole">
          <span>Pełna nazwa aktu prawnego lub procedury wewnętrznej</span>
          <input
            name="akt"
            placeholder="np. Ustawa o emeryturach i rentach z FUS / Procedura P-NPS rozliczania nienależnie pobranych świadczeń"
          />
        </label>
        <div className="wiersz">
          <label className="pole male">
            <span>Obowiązuje od</span>
            <input type="date" name="obowiazuje_od" />
          </label>
          <label className="pole male">
            <span>Obowiązuje do (puste = nadal)</span>
            <input type="date" name="obowiazuje_do" />
          </label>
          <label className="pole male">
            <span>Oznaczenie wersji / Dz.U.</span>
            <input name="wersja" placeholder="np. tj. Dz.U. 2024 poz. 1631 / wer. 3.2" />
          </label>
        </div>
        <div className="wiersz-miedzy">
          <label className="pole-wiersz" title={jestModel ? '' : 'Wymaga działającego modelu LLM'}>
            <input
              type="checkbox"
              name="opisy"
              value="1"
              disabled={!jestModel}
              checked={zOpisami && jestModel}
              onChange={(e) => setZOpisami(e.target.checked)}
            />
            Generuj od razu opisy kontekstowe przez LLM (ok. 4 s na przepis; można też włączyć później przyciskiem
            „Generuj opisy”)
          </label>
          <button type="submit" className="glowny duzy" disabled={wysylam || wToku}>
            {wToku ? 'Trwa zadanie w tle…' : wysylam ? 'Wysyłanie…' : 'Wgraj i przebuduj wspólny indeks'}
          </button>
        </div>
        <p className="cichy">
          <strong>Wskazówka:</strong> Procedury wewnętrzne ZUS automatycznie wykrywają powołane w tekście artykuły ustaw
          (np. <code>art. 138 ust. 4</code>) i łączą się z nimi podczas wyszukiwania odpowiedzi.
        </p>
      </form>
    </section>
  )
}
