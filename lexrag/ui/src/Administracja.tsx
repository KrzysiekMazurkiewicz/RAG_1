import { useEffect, useState, type FormEvent } from 'react'
import { api, type Stan } from './api'

export default function Administracja({ stan, odswiez }: { stan: Stan | null; odswiez: () => Promise<void> }) {
  const [komunikat, setKomunikat] = useState('')
  const [wysylam, setWysylam] = useState(false)
  const [zOpisami, setZOpisami] = useState(true)
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
      setKomunikat('Przyjęto: parsowanie i budowa indeksu trwają (embeddingi na karcie: minuta, na CPU: kilkanaście minut).')
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
      setKomunikat(`Model pisze opisy kontekstowe dla aktu ${kod}; postęp w polu „zadanie”. Po zakończeniu indeks przebuduje się sam.`)
      await odswiez()
    } catch (err) {
      setKomunikat(`Błąd: ${(err as Error).message}`)
    }
  }

  return (
    <section>
      <div className="karta">
        <h3>Stan silnika</h3>
        {stan ? (
          <table className="tabela">
            <tbody>
              <tr>
                <th>wersja</th>
                <td>{stan.wersja}</td>
              </tr>
              <tr>
                <th>model</th>
                <td>
                  {stan.llm.model || '(brak: tryb retrieval-only)'} · {stan.llm.baza} ·{' '}
                  <span className={stan.llm.polaczenie?.ok ? 'ok' : 'blad'}>
                    {stan.llm.polaczenie?.ok ? 'odpowiada' : `nie odpowiada: ${stan.llm.polaczenie?.komunikat ?? ''}`}
                  </span>
                </td>
              </tr>
              <tr>
                <th>indeks</th>
                <td>
                  {stan.indeks.jednostek} jednostek, {stan.indeks.szukalnych} szukalnych, embeddingi:{' '}
                  {stan.indeks.embeddingi ? 'tak' : 'nie'} · {stan.indeks.katalog}
                </td>
              </tr>
              <tr>
                <th>zadanie</th>
                <td>
                  {stan.zadanie.status}
                  {stan.zadanie.komunikat ? ` · ${stan.zadanie.komunikat}` : ''}
                </td>
              </tr>
            </tbody>
          </table>
        ) : (
          <p className="cichy">łączę…</p>
        )}
        {komunikat && <p className={komunikat.startsWith('Błąd') ? 'blad' : 'ok'}>{komunikat}</p>}
      </div>

      <form className="karta formularz" onSubmit={zapiszModel}>
        <h3>Serwer modelu</h3>
        <div className="wiersz">
          <label className="pole">
            <span>Adres serwera (Ollama: …:11434/v1; tunel z Colaba: https://…trycloudflare.com/v1)</span>
            <input value={adres} onChange={(e) => setAdres(e.target.value)} placeholder="http://localhost:11434/v1" />
          </label>
          <label className="pole male">
            <span>Model</span>
            <input value={model} onChange={(e) => setModel(e.target.value)} placeholder="gemma4:12b" />
          </label>
          <label className="pole male">
            <span>API</span>
            <select value={api2} onChange={(e) => setApi2(e.target.value)}>
              <option value="auto">auto</option>
              <option value="ollama">ollama</option>
              <option value="openai">openai (vLLM, llama.cpp)</option>
            </select>
          </label>
          <button type="submit" className="glowny" disabled={zapisuje || !adres.trim() || !model.trim()}>
            {zapisuje ? 'sprawdzam…' : 'Zapisz i sprawdź'}
          </button>
        </div>
        {wynikModelu && <p className={wynikModelu.startsWith('Błąd') ? 'blad' : 'ok'}>{wynikModelu}</p>}
        <p className="cichy">
          Zmiana działa od razu, bez restartu, i jest zapamiętana w data/model.json (nadpisuje zmienne środowiskowe).
          Tunel z Colaba ma inny adres w każdej sesji: wklej nowy adres z komórki „tunel” i zapisz.
        </p>
      </form>

      <div className="karta">
        <h3>Wgrane akty i dokumenty</h3>
        {stan && stan.indeks.akty.length > 0 ? (
          <table className="tabela">
            <thead>
              <tr>
                <th>kod</th>
                <th>nazwa</th>
                <th>wersja</th>
                <th>przepisów</th>
                <th>opisy kontekstowe</th>
                <th>obowiązuje od</th>
                <th>obowiązuje do</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {stan.indeks.akty.map((a) => (
                <tr key={a.kod}>
                  <td className="id">{a.kod}</td>
                  <td>{a.akt}</td>
                  <td>{a.wersja || '—'}</td>
                  <td>{a.szukalnych}</td>
                  <td>
                    {a.opisy} / {a.szukalnych}
                  </td>
                  <td>{a.obowiazuje_od ?? '—'}</td>
                  <td>{a.obowiazuje_do ?? 'nadal'}</td>
                  <td>
                    {a.opisy < a.szukalnych && (
                      <button
                        onClick={() => void opisz(a.kod)}
                        disabled={wToku || !jestModel}
                        title={jestModel ? 'model dopisze jedno zdanie opisu do każdego przepisu bez opisu (ok. 4 s na przepis na karcie)' : 'wymaga działającego modelu'}
                      >
                        Opisz przepisy
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="cichy">Nic jeszcze nie wgrano.</p>
        )}
        <p className="cichy">
          Opisy kontekstowe to jedno zdanie na przepis, pisane przez model słowami pracownika („kiedy można…”, „ile
          wynosi…”). Trafiają do wyszukiwania, nie do odpowiedzi; poprawiają trafność dla pytań hasłowych. Czas: około 4 s
          na przepis przy modelu na karcie.
        </p>
      </div>

      <form className="karta formularz" onSubmit={wyslij}>
        <h3>Wgraj akt albo dokument</h3>
        <div className="wiersz">
          <label className="pole">
            <span>Plik (PDF, TXT, HTML, DOCX, MD)</span>
            <input type="file" name="plik" required accept=".pdf,.txt,.html,.htm,.docx,.md" />
          </label>
          <label className="pole male">
            <span>Kod (np. zas_2026)</span>
            <input name="kod" required pattern="[a-z0-9_]{1,24}" placeholder="małe litery, cyfry, _" />
          </label>
          <label className="pole male">
            <span>Typ</span>
            <select name="typ" defaultValue="ustawa">
              <option value="ustawa">ustawa</option>
              <option value="dokument">dokument wewnętrzny</option>
            </select>
          </label>
        </div>
        <label className="pole">
          <span>Pełna nazwa aktu</span>
          <input name="akt" placeholder="Ustawa o świadczeniach pieniężnych z ubezpieczenia społecznego w razie choroby i macierzyństwa" />
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
            <span>Wersja (etykieta)</span>
            <input name="wersja" placeholder="tj. Dz.U. 2026 poz. 854" />
          </label>
        </div>
        <div className="wiersz">
          <label className="pole-wiersz" title={jestModel ? '' : 'wymaga działającego modelu'}>
            <input
              type="checkbox"
              name="opisy"
              value="1"
              disabled={!jestModel}
              checked={zOpisami && jestModel}
              onChange={(e) => setZOpisami(e.target.checked)}
            />
            po wgraniu model dopisze opisy kontekstowe do przepisów (ok. 4 s na przepis; dla ustawy 100 artykułów około 20 minut)
          </label>
          <button type="submit" className="glowny" disabled={wysylam || wToku}>
            {wToku ? 'trwa zadanie…' : wysylam ? 'wysyłam…' : 'Wgraj'}
          </button>
        </div>
        <p className="cichy">
          Nowa wersja tego samego aktu dostaje nowy kod (np. zas_2027), a stara datę „obowiązuje do”. Po wgraniu
          sprawdź raport w polu „zadanie” i otwórz kilka przepisów, zanim udostępnisz akt użytkownikom.
        </p>
      </form>
    </section>
  )
}
