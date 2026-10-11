import { useState } from 'react'
import { api, type Jednostka, type Odpowiedz } from './api'

const NAZWA_TRYBU: Record<Odpowiedz['tryb'], string> = {
  odpowiedz: 'odpowiedź',
  odmowa: 'odmowa',
  dopytanie: 'dopytanie',
  'retrieval-only': 'tylko wyszukiwanie (bez modelu)',
  blad: 'błąd',
}

function okres(j: Pick<Jednostka, 'obowiazuje_od' | 'obowiazuje_do'>): string {
  if (!j.obowiazuje_od && !j.obowiazuje_do) return ''
  return `obowiązuje ${j.obowiazuje_od ?? '…'} – ${j.obowiazuje_do ?? 'nadal'}`
}

function Cytat({ j, otwarty }: { j: Jednostka; otwarty: boolean }) {
  return (
    <details className="cytat" open={otwarty}>
      <summary>
        <span className="id">{j.id}</span> {j.sciezka}
        <span className="cichy"> · {j.akt}</span>
      </summary>
      <p className="tekst-przepisu">{j.tekst}</p>
      {okres(j) && <p className="cichy">{okres(j)}</p>}
    </details>
  )
}

export default function OdpowiedzWidok({ w }: { w: Odpowiedz }) {
  const [ocena, setOcena] = useState<'' | 'dobrze' | 'zle' | 'nie_wiem'>('')
  const [zaznaczone, setZaznaczone] = useState<string[]>([])
  const [innyPrzepis, setInnyPrzepis] = useState('')
  const [komentarz, setKomentarz] = useState('')
  const [wyslane, setWyslane] = useState('')
  const [bladOpinii, setBladOpinii] = useState('')

  const przelacz = (id: string) =>
    setZaznaczone((z) => (z.includes(id) ? z.filter((x) => x !== id) : [...z, id]))

  const wyslij = async () => {
    setBladOpinii('')
    const przepisy = [...zaznaczone, ...innyPrzepis.split(/[,\s]+/).filter(Boolean)]
    try {
      await api.opinia({ id_odpowiedzi: w.id_odpowiedzi, ocena, wlasciwy_przepis: przepisy, komentarz })
      setWyslane(ocena === 'zle' ? 'Zgłoszenie trafiło do kolejki referenta. Dziękujemy.' : 'Dziękujemy za ocenę.')
    } catch (e) {
      setBladOpinii((e as Error).message)
    }
  }

  const czasy = Object.entries(w.czasy)
    .map(([k, v]) => `${k} ${v} s`)
    .join(', ')

  return (
    <div className="karta">
      <div className="wiersz">
        <span className={`znacznik ${w.tryb}`}>{NAZWA_TRYBU[w.tryb] ?? w.tryb}</span>
        {w.przeglad && <span className="znacznik przeglad">odpowiedź przeglądowa</span>}
        {czasy && <span className="cichy">{czasy}</span>}
      </div>

      {w.przeglad && (
        <p className="cichy">
          Pytanie jest ogólne albo brakuje w nim danych o sytuacji, więc odpowiedź wymienia wszystkie sytuacje z
          odnalezionych przepisów. Podaj szczegóły, a odpowiedź będzie konkretna.
        </p>
      )}

      {w.tryb === 'dopytanie' ? (
        <>
          <p className="odpowiedz">{w.tekst}</p>
          {w.brakujace_parametry.length > 0 && <p className="cichy">brakuje: {w.brakujace_parametry.join(', ')}</p>}
          {w.uwagi && <p className="cichy">uwagi: {w.uwagi}</p>}
        </>
      ) : (
        <>
          {w.tekst && <p className="odpowiedz">{w.tekst}</p>}
          {w.dopytanie && <p className="dopytanie">Aby odpowiedzieć dokładniej: {w.dopytanie}</p>}
          {w.brakujace_parametry.length > 0 && !w.dopytanie && (
            <p className="cichy">brakuje: {w.brakujace_parametry.join(', ')}</p>
          )}
          {w.uwagi && <p className="cichy">uwagi: {w.uwagi}</p>}
        </>
      )}

      {w.cytaty.length > 0 && (
        <>
          <h3>Podstawa prawna (tekst z ustawy, nie od modelu)</h3>
          {w.cytaty.map((j, i) => (
            <Cytat key={j.id} j={j} otwarty={i === 0} />
          ))}
        </>
      )}

      {w.tryb === 'retrieval-only' && w.trafienia.length > 0 && (
        <>
          <h3>Najlepsze trafienia</h3>
          {w.trafienia.map((t) => (
            <details key={t.id} className="cytat">
              <summary>
                <span className="id">{t.id}</span> {t.sciezka} <span className="cichy">({t.wynik})</span>
              </summary>
              <p className="tekst-przepisu">{t.tekst}</p>
            </details>
          ))}
        </>
      )}

      {w.zdania_usuniete.length > 0 && (
        <details>
          <summary className="cichy">zdania usunięte przez kontrolę ({w.zdania_usuniete.length})</summary>
          <ul className="cichy">
            {w.zdania_usuniete.map((z, i) => (
              <li key={i}>{z}</li>
            ))}
          </ul>
        </details>
      )}

      {w.tryb !== 'retrieval-only' && w.tryb !== 'blad' && (
        <div className="opinia">
          {wyslane ? (
            <p className="ok">{wyslane}</p>
          ) : (
            <>
              <span className="cichy">Czy odpowiedź jest poprawna?</span>
              {(['dobrze', 'zle', 'nie_wiem'] as const).map((o) => (
                <button key={o} className={ocena === o ? 'aktywna' : ''} onClick={() => setOcena(o)}>
                  {o === 'dobrze' ? 'dobrze' : o === 'zle' ? 'źle' : 'nie wiem'}
                </button>
              ))}
              {ocena === 'zle' && (
                <div className="formularz-opinii">
                  <p className="cichy">Który przepis jest właściwy? Zaznacz z listy przejrzanych albo wpisz adres.</p>
                  <ul className="lista-trafien">
                    {w.trafienia.slice(0, 8).map((t) => (
                      <li key={t.id}>
                        <label className="pole-wiersz">
                          <input type="checkbox" checked={zaznaczone.includes(t.id)} onChange={() => przelacz(t.id)} />
                          <span className="id">{t.id}</span> {t.sciezka}
                        </label>
                      </li>
                    ))}
                  </ul>
                  <input
                    placeholder="inny przepis, np. fus:art58:ust2"
                    value={innyPrzepis}
                    onChange={(e) => setInnyPrzepis(e.target.value)}
                  />
                  <input
                    placeholder="komentarz (np. odpowiedź o rencie, pytanie o emeryturę)"
                    value={komentarz}
                    onChange={(e) => setKomentarz(e.target.value)}
                  />
                </div>
              )}
              {ocena && (
                <button className="glowny" onClick={() => void wyslij()}>
                  Wyślij ocenę
                </button>
              )}
              {bladOpinii && <span className="blad">{bladOpinii}</span>}
            </>
          )}
        </div>
      )}
    </div>
  )
}
