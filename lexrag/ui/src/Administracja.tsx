import { useEffect, useState, type FormEvent } from 'react'
import { api, type Stan } from './api'

export default function Administracja({ stan, odswiez }: { stan: Stan | null; odswiez: () => Promise<void> }) {
  const [komunikat, setKomunikat] = useState('')
  const [wysylam, setWysylam] = useState(false)
  const wToku = stan?.zadanie.status === 'w toku'

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
                  <span className={stan.llm.dostepny ? 'ok' : 'blad'}>{stan.llm.dostepny ? 'odpowiada' : 'nie odpowiada'}</span>
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
      </div>

      <div className="karta">
        <h3>Wgrane akty i dokumenty</h3>
        {stan && stan.indeks.akty.length > 0 ? (
          <table className="tabela">
            <thead>
              <tr>
                <th>kod</th>
                <th>nazwa</th>
                <th>jednostek</th>
                <th>obowiązuje od</th>
                <th>obowiązuje do</th>
              </tr>
            </thead>
            <tbody>
              {stan.indeks.akty.map((a) => (
                <tr key={a.kod}>
                  <td className="id">{a.kod}</td>
                  <td>{a.akt}</td>
                  <td>{a.jednostek}</td>
                  <td>{a.obowiazuje_od ?? '—'}</td>
                  <td>{a.obowiazuje_do ?? 'nadal'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="cichy">Nic jeszcze nie wgrano.</p>
        )}
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
          <button type="submit" className="glowny" disabled={wysylam || wToku}>
            {wToku ? 'trwa wgrywanie…' : wysylam ? 'wysyłam…' : 'Wgraj'}
          </button>
        </div>
        {komunikat && <p className={komunikat.startsWith('Błąd') ? 'blad' : 'ok'}>{komunikat}</p>}
        <p className="cichy">
          Nowa wersja tego samego aktu dostaje nowy kod (np. zas_2027), a stara datę „obowiązuje do”. Po wgraniu
          sprawdź raport w polu „zadanie” i otwórz kilka przepisów, zanim udostępnisz akt użytkownikom.
        </p>
      </form>
    </section>
  )
}
