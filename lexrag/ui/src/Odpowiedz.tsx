import { useState } from 'react'
import { api, type Jednostka, type Odpowiedz } from './api'

const STATUS: Record<string, { nazwa: string; opis: string; ikona: string }> = {
  verified: {
    nazwa: 'ZWERYFIKOWANA W PRZEPISACH',
    opis: 'Każde zdanie przeszło wieloetapową kontrolę: pokrycie w cytowanym przepisie, struktura warunków (i/albo), negacje, liczby oraz zakres podmiotowy.',
    ikona: '✓',
  },
  unverified: {
    nazwa: 'SZKIC DO WERYFIKACJI PRZEZ PRACOWNIKA',
    opis: 'Kontrola wykryła zastrzeżenia lub weryfikator był niedostępny. Sprawdź odpowiedź bezpośrednio z zacytowanymi przepisami poniżej.',
    ikona: '⚠',
  },
  insufficient_sources: {
    nazwa: 'SPRAWA DLA REFERENTA / BRAK PODSTAW W INDEKSIE',
    opis: 'Wgrane akty nie pozwalają rozstrzygnąć pytania jednoznacznie albo odpowiedź nie przeszła bramek bezpieczeństwa.',
    ikona: '!',
  },
}

const KRYTYCZNE = new Set([
  'weryfikacja_nieudana',
  'weryfikacja_wylaczona',
  'nie_rozstrzyga',
  'laczy_odrebne_podstawy',
  'istotne_pominiecia',
  'brak_wersji_na_date',
  'brak_wersji_historycznej',
])

const NAZWA_TRYBU: Record<Odpowiedz['tryb'], string> = {
  odpowiedz: 'Odpowiedź merytoryczna',
  odmowa: 'Odmowa (przekaż referentowi)',
  dopytanie: 'Wymagane doprecyzowanie',
  'retrieval-only': 'Wyszukiwanie przepisów (bez modelu)',
  blad: 'Błąd połączenia z modelem',
}

function okres(j: Pick<Jednostka, 'obowiazuje_od' | 'obowiazuje_do'>): string {
  if (!j.obowiazuje_od && !j.obowiazuje_do) return ''
  return `obowiązuje ${j.obowiazuje_od ?? '…'} – ${j.obowiazuje_do ?? 'nadal'}`
}

function Cytat({
  j,
  otwarty,
  onOtworzJednostke,
}: {
  j: Jednostka
  otwarty: boolean
  onOtworzJednostke: (id: string) => void
}) {
  const [skopiowano, setSkopiowano] = useState(false)

  const kopiuj = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    void navigator.clipboard?.writeText(`${j.sciezka} (${j.akt})\n\n${j.tekst}`)
    setSkopiowano(true)
    setTimeout(() => setSkopiowano(false), 1800)
  }

  return (
    <details className="cytat" open={otwarty}>
      <summary>
        <div className="cytat-naglowek">
          <div>
            <span className="id">{j.id}</span> <strong>{j.sciezka}</strong>
            <span className="cichy"> · {j.akt}</span>
          </div>
          <div className="wiersz" onClick={(e) => e.stopPropagation()}>
            {j.rodzic && (
              <button
                type="button"
                className="maly-przycisk"
                onClick={() => onOtworzJednostke(j.rodzic!)}
                title="Pokaż cały artykuł nadrzędny"
              >
                Cały artykuł
              </button>
            )}
            <button type="button" className="maly-przycisk" onClick={kopiuj}>
              {skopiowano ? '✓ Skopiowano' : 'Kopiuj'}
            </button>
          </div>
        </div>
      </summary>
      <div className="cytat-tresc">
        <p className="tekst-przepisu">{j.tekst}</p>
        <div className="wiersz-miedzy">
          {okres(j) && <span className="cichy">{okres(j)}</span>}
          {j.odwolania && j.odwolania.length > 0 && (
            <div className="wiersz odwolania-wiersz">
              <span className="cichy">Odesłania:</span>
              {j.odwolania.slice(0, 8).map((o) => (
                <button key={o} type="button" className="chip maly" onClick={() => onOtworzJednostke(o)}>
                  ↪ {o}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
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
  const [podgladJednostki, setPodgladJednostki] = useState<Jednostka | null>(null)
  const [skopiowanoOdp, setSkopiowanoOdp] = useState(false)

  const przelacz = (id: string) =>
    setZaznaczone((z) => (z.includes(id) ? z.filter((x) => x !== id) : [...z, id]))

  const otworzJednostke = async (ident: string) => {
    try {
      const j = await api.jednostka(ident)
      setPodgladJednostki(j)
    } catch {
      /* jednostka spoza indeksu */
    }
  }

  const kopiujOdpowiedz = () => {
    const podstawa = w.cytaty.map((c) => `- ${c.sciezka} (${c.akt}): ${c.tekst}`).join('\n')
    const pelnyTekst = `${w.tekst}\n\nPodstawa prawna:\n${podstawa}`
    void navigator.clipboard?.writeText(pelnyTekst)
    setSkopiowanoOdp(true)
    setTimeout(() => setSkopiowanoOdp(false), 2000)
  }

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

  const infoStatus = w.status_weryfikacji ? STATUS[w.status_weryfikacji] : null
  const sumaCzasow = Object.values(w.czasy).reduce((a, b) => a + b, 0)

  return (
    <div className="karta karta-odpowiedzi">
      {infoStatus && (
        <div className={`baner-weryfikacji ${w.status_weryfikacji}`}>
          <div className="baner-tytul">
            <span className="baner-ikona">{infoStatus.ikona}</span>
            <strong>{infoStatus.nazwa}</strong>
          </div>
          <span className="baner-opis">{infoStatus.opis}</span>
        </div>
      )}

      <div className="wiersz-miedzy pasek-meta-odpowiedzi">
        <div className="wiersz">
          <span className={`znacznik ${w.tryb}`}>{NAZWA_TRYBU[w.tryb] ?? w.tryb}</span>
          {w.przeglad && <span className="znacznik przeglad">Tryb przeglądowy (zasada ogólna + wyjątki)</span>}
          {w.naprawa && (
            <span
              className="znacznik naprawa"
              title="Pierwsza wersja nie przeszła bramek kontroli; wyświetlona odpowiedź została automatycznie poprawiona w drugim przebiegu."
            >
              Po korekcie kontrolnej
            </span>
          )}
        </div>
        <div className="wiersz">
          {Object.entries(w.czasy).map(([krok, s]) => (
            <span key={krok} className="pigulka-czasu" title={`Czas etapu: ${krok}`}>
              {krok}: <strong>{s}s</strong>
            </span>
          ))}
          {sumaCzasow > 0 && <span className="cichy">({sumaCzasow.toFixed(1)}s łącznie)</span>}
          {w.slad_url && (
            <a
              className="maly-przycisk"
              href={w.slad_url}
              target="_blank"
              rel="noreferrer"
              title="Pełny ślad diagnostyczny JSON: analiza, ranking, prompty, surowe wyjścia modelu, werdykty"
            >
              🔍 Ślad diagnostyczny
            </a>
          )}
          {w.tekst && (
            <button type="button" className="maly-przycisk" onClick={kopiujOdpowiedz}>
              {skopiowanoOdp ? '✓ Skopiowano' : 'Kopiuj z podstawą'}
            </button>
          )}
        </div>
      </div>

      {w.ostrzezenia.map((o) => (
        <div key={o.kod} className={KRYTYCZNE.has(o.kod) ? 'ramka-ostrzezenia krytyczne' : 'ramka-ostrzezenia info'}>
          <strong>Uwaga ({o.kod}):</strong> {o.tekst}
        </div>
      ))}

      {w.przeglad && (
        <p className="cichy">
          Pytanie ma formę hasłową lub prośby o przegląd, dlatego odpowiedź zestawia regułę ogólną oraz najważniejsze
          wyjątki ustawowe.
        </p>
      )}

      {w.tryb === 'dopytanie' ? (
        <div className="blok-odpowiedzi dopytanie-blok">
          <p className="odpowiedz">{w.tekst}</p>
          {w.brakujace_parametry.length > 0 && (
            <p className="cichy">Wymagane dane: {w.brakujace_parametry.join(', ')}</p>
          )}
          {w.uwagi && <p className="cichy">Uwagi: {w.uwagi}</p>}
        </div>
      ) : (
        <div className="blok-odpowiedzi">
          {w.tekst && <p className="odpowiedz">{w.tekst}</p>}
          {w.dopytanie && (
            <div className="ramka-dopytania">
              <strong>Pytanie pomocnicze do ustalenia stanu faktycznego:</strong> {w.dopytanie}
            </div>
          )}
          {w.brakujace_parametry.length > 0 && !w.dopytanie && (
            <p className="cichy">Brakujące dane: {w.brakujace_parametry.join(', ')}</p>
          )}
          {w.uwagi && <p className="cichy">Uwagi: {w.uwagi}</p>}
        </div>
      )}

      {w.dane_do_pelnej_oceny.length > 0 && (
        <div className="ramka-pelnej-oceny">
          <strong>Do indywidualnej oceny sprawy przez pracownika ZUS warto dodatkowo zweryfikować:</strong>
          <ul>
            {w.dane_do_pelnej_oceny.map((d, i) => (
              <li key={i}>
                <strong>{d.dana}</strong> — {d.dlaczego}
              </li>
            ))}
          </ul>
        </div>
      )}

      {podgladJednostki && (
        <div className="karta podglad-jednostki">
          <div className="wiersz-miedzy">
            <div>
              <span className="id">{podgladJednostki.id}</span> <strong>{podgladJednostki.sciezka}</strong>
              <span className="cichy"> · {podgladJednostki.akt}</span>
            </div>
            <button type="button" className="maly-przycisk" onClick={() => setPodgladJednostki(null)}>
              Zamknij podgląd
            </button>
          </div>
          <pre className="tekst-przepisu-blok">{podgladJednostki.tekst}</pre>
        </div>
      )}

      {w.cytaty.length > 0 && (
        <div className="sekcja-cytatow">
          <div className="wiersz-miedzy">
            <h3>Podstawa prawna (dosłowna treść z indeksu, gwarantowana bez halucynacji modelu)</h3>
            <span className="cichy">Cytowanych jednostek: {w.cytaty.length}</span>
          </div>
          {w.cytaty.map((j, i) => (
            <Cytat key={j.id} j={j} otwarty={i < 2} onOtworzJednostke={(id) => void otworzJednostke(id)} />
          ))}
          {w.zrodla.length > 0 && (
            <p className="cichy metryczka-zrodel">
              <strong>Źródła w indeksie:</strong>{' '}
              {w.zrodla
                .map(
                  (a) =>
                    `${a.akt} (wersja: ${a.wersja || 'nieoznaczona'}, obowiązuje: ${a.obowiazuje_od ?? 'brak daty'} – ${a.obowiazuje_do ?? 'nadal'})`,
                )
                .join('; ')}
              .
            </p>
          )}
        </div>
      )}

      {w.tryb === 'retrieval-only' && w.trafienia.length > 0 && (
        <div className="sekcja-cytatow">
          <h3>Najlepsze trafienia z indeksu (BM25 + wektory + reranker)</h3>
          {w.trafienia.map((t, idx) => (
            <details key={t.id} className="cytat" open={idx < 3}>
              <summary>
                <div className="cytat-naglowek">
                  <div>
                    <span className="id">{t.id}</span> <strong>{t.sciezka}</strong>
                  </div>
                  <span className="pigulka-czasu">wynik: {t.wynik}</span>
                </div>
              </summary>
              <p className="tekst-przepisu">{t.tekst}</p>
            </details>
          ))}
        </div>
      )}

      <div className="diagnostyka-dolna">
        {w.warianty.length > 0 && (
          <details>
            <summary className="cichy">Warianty zapytania wygenerowane przez analizę ({w.warianty.length})</summary>
            <ul className="cichy">
              {w.warianty.map((war, i) => (
                <li key={i}>{war}</li>
              ))}
            </ul>
          </details>
        )}

        {w.zdania_usuniete.length > 0 && (
          <details>
            <summary className="cichy">
              Zdania zatrzymane przez bramki weryfikacji ({w.zdania_usuniete.length})
            </summary>
            <ul className="cichy">
              {w.zdania_usuniete.map((z, i) => (
                <li key={i}>{z}</li>
              ))}
            </ul>
          </details>
        )}
      </div>

      {w.tryb !== 'retrieval-only' && w.tryb !== 'blad' && (
        <div className="opinia">
          {wyslane ? (
            <p className="ok">✓ {wyslane}</p>
          ) : (
            <>
              <span className="cichy">Oceń jakość odpowiedzi (buduje złoty zbiór testowy ZUS):</span>
              {(['dobrze', 'zle', 'nie_wiem'] as const).map((o) => (
                <button key={o} type="button" className={ocena === o ? 'aktywna' : ''} onClick={() => setOcena(o)}>
                  {o === 'dobrze' ? '👍 Poprawna' : o === 'zle' ? '👎 Błędna (zgłoś referentowi)' : '❓ Nie mam pewności'}
                </button>
              ))}
              {ocena === 'zle' && (
                <div className="formularz-opinii">
                  <p className="cichy">Który przepis jest właściwy? Zaznacz z listy przejrzanych albo wpisz adres:</p>
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
                    placeholder="Inny przepis (np. fus:art58:ust2, fus:art5:ust2)"
                    value={innyPrzepis}
                    onChange={(e) => setInnyPrzepis(e.target.value)}
                  />
                  <input
                    placeholder="Komentarz merytoryczny (np. pominięto limit 1/3 okresów nieskładkowych z art. 5 ust. 2)"
                    value={komentarz}
                    onChange={(e) => setKomentarz(e.target.value)}
                  />
                </div>
              )}
              {ocena && (
                <button type="button" className="glowny" onClick={() => void wyslij()}>
                  Zapisz ocenę
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
