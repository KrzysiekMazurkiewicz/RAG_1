import { useCallback, useEffect, useState } from 'react'
import { api, type Opinia } from './api'

const NAZWA_OCENY = { dobrze: 'dobrze', zle: 'źle', nie_wiem: 'nie wiem' } as const

export default function Opinie() {
  const [status, setStatus] = useState<'nowa' | 'zatwierdzona' | 'odrzucona' | 'wszystkie'>('nowa')
  const [lista, setLista] = useState<Opinia[]>([])
  const [blad, setBlad] = useState('')
  const [frazy, setFrazy] = useState<Record<string, string>>({})
  const [przepisy, setPrzepisy] = useState<Record<string, string>>({})
  const [referent, setReferent] = useState(() => localStorage.getItem('lexrag.referent') ?? '')

  const wczytaj = useCallback(async () => {
    try {
      setLista(await api.opinie(status))
      setBlad('')
    } catch (e) {
      setBlad((e as Error).message)
    }
  }, [status])

  useEffect(() => {
    void wczytaj()
  }, [wczytaj])

  const zdecyduj = async (o: Opinia, decyzja: 'zatwierdzona' | 'odrzucona') => {
    const wpisane = (przepisy[o.id] ?? '').split(/[,\s]+/).filter(Boolean)
    try {
      await api.przeglad(o.id, {
        status: decyzja,
        wlasciwy_przepis: wpisane.length ? wpisane : undefined,
        musi_zawierac: (frazy[o.id] ?? '').split('|').map((f) => f.trim()).filter(Boolean),
        przegladajacy: referent,
      })
      localStorage.setItem('lexrag.referent', referent)
      await wczytaj()
    } catch (e) {
      setBlad((e as Error).message)
    }
  }

  const pobierzEksport = async () => {
    try {
      const e = await api.eksport()
      const blob = new Blob([e.jsonl + '\n'], { type: 'application/jsonl' })
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = 'z_opinii.jsonl'
      a.click()
      URL.revokeObjectURL(a.href)
    } catch (err) {
      setBlad((err as Error).message)
    }
  }

  return (
    <section>
      <div className="karta wiersz">
        <label className="pole male">
          <span>Status</span>
          <select value={status} onChange={(e) => setStatus(e.target.value as typeof status)}>
            <option value="nowa">nowe (do przeglądu)</option>
            <option value="zatwierdzona">zatwierdzone</option>
            <option value="odrzucona">odrzucone</option>
            <option value="wszystkie">wszystkie</option>
          </select>
        </label>
        <label className="pole male">
          <span>Przeglądający</span>
          <input value={referent} onChange={(e) => setReferent(e.target.value)} placeholder="imię i nazwisko" />
        </label>
        <button onClick={() => void wczytaj()}>Odśwież</button>
        <button onClick={() => void pobierzEksport()} title="zatwierdzone opinie jako wiersze złotego zbioru (JSONL)">
          Pobierz złoty zbiór z opinii
        </button>
        <span className="cichy">{lista.length} pozycji</span>
      </div>
      {blad && <div className="karta blad">Błąd: {blad}</div>}
      {lista.length === 0 && !blad && <div className="karta cichy">Brak opinii o tym statusie.</div>}

      {lista.map((o) => (
        <div key={o.id} className="karta">
          <div className="wiersz">
            <span className={`znacznik ocena-${o.ocena}`}>{NAZWA_OCENY[o.ocena]}</span>
            <span className="cichy">
              {o.czas.replace('T', ' ')} · {o.uzytkownik || 'użytkownik'} · model {o.model || '?'} · tryb {o.tryb}
            </span>
          </div>
          <p>
            <strong>Pytanie:</strong> {o.pytanie}
          </p>
          <p>
            <strong>Odpowiedź:</strong> {o.odpowiedz || <span className="cichy">(brak tekstu)</span>}
          </p>
          <p className="cichy">
            cytowano: {o.cytowane.join(', ') || '—'}
            {o.wlasciwy_przepis.length > 0 && <> · wskazany właściwy: {o.wlasciwy_przepis.join(', ')}</>}
            {o.komentarz && <> · komentarz: {o.komentarz}</>}
          </p>
          {o.status === 'nowa' ? (
            <div className="formularz-opinii">
              <input
                placeholder="poprawiony właściwy przepis (puste = zostaw wskazany), np. fus:art58:ust2"
                value={przepisy[o.id] ?? ''}
                onChange={(e) => setPrzepisy({ ...przepisy, [o.id]: e.target.value })}
              />
              <input
                placeholder="frazy, które muszą paść w poprawnej odpowiedzi, rozdzielone | (np. 5 lat|pięciu lat)"
                value={frazy[o.id] ?? ''}
                onChange={(e) => setFrazy({ ...frazy, [o.id]: e.target.value })}
              />
              <div className="wiersz">
                <button className="glowny" onClick={() => void zdecyduj(o, 'zatwierdzona')}>
                  Zatwierdź
                </button>
                <button onClick={() => void zdecyduj(o, 'odrzucona')}>Odrzuć</button>
              </div>
            </div>
          ) : (
            <p className="cichy">
              {o.status} · {o.przegladajacy || 'referent'}
              {o.musi_zawierac.length > 0 && <> · frazy: {o.musi_zawierac.join(' | ')}</>}
            </p>
          )}
        </div>
      ))}
    </section>
  )
}
