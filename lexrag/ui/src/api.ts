// Klient REST silnika LexRAG (lexrag/web.py). Wszystkie ścieżki są względne, więc działa za proxy Vite i z ui/dist.

export interface Akt {
  kod: string
  akt: string
  jednostek: number
  obowiazuje_od: string | null
  obowiazuje_do: string | null
}

export interface Zadanie {
  status: string
  kod?: string
  komunikat?: string
}

export interface Stan {
  wersja: string
  indeks: { katalog: string; jednostek: number; szukalnych: number; embeddingi: boolean; akty: Akt[] }
  llm: { model: string; baza: string; dostepny: boolean }
  zadanie: Zadanie
}

export interface Jednostka {
  id: string
  akt: string
  akt_kod: string
  poziom: string
  sciezka: string
  tekst: string
  rodzic: string | null
  odwolania: string[]
  obowiazuje_od: string | null
  obowiazuje_do: string | null
}

export interface Trafienie {
  id: string
  wynik: number
  zrodlo: string
  sciezka: string
  tekst: string
}

export interface Odpowiedz {
  id_odpowiedzi: string
  tryb: 'odpowiedz' | 'odmowa' | 'dopytanie' | 'retrieval-only' | 'blad'
  tekst: string
  uwagi: string
  czasy: Record<string, number>
  warianty: string[]
  brakujace_parametry: string[]
  zdania_usuniete: string[]
  przeglad: boolean
  dopytanie: string
  cytaty: Jednostka[]
  trafienia: Trafienie[]
  kontekst: string[]
}

export interface Opinia {
  id: string
  czas: string
  uzytkownik: string
  id_odpowiedzi: string
  ocena: 'dobrze' | 'zle' | 'nie_wiem'
  wlasciwy_przepis: string[]
  komentarz: string
  status: 'nowa' | 'zatwierdzona' | 'odrzucona'
  musi_zawierac: string[]
  przegladajacy: string
  pytanie: string
  odpowiedz: string
  cytowane: string[]
  tryb: string
  model: string
}

async function sprawdz<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let szczegol = r.statusText
    try {
      const j = await r.json()
      szczegol = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail ?? j)
    } catch {
      /* brak JSON w błędzie */
    }
    throw new Error(`${r.status}: ${szczegol}`)
  }
  return (await r.json()) as T
}

export const api = {
  stan: () => fetch('/api/stan').then((r) => sprawdz<Stan>(r)),

  pytanie: (pytanie: string, na_dzien: string | null, analiza: boolean | null) =>
    fetch('/api/pytanie', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pytanie, na_dzien, analiza }),
    }).then((r) => sprawdz<Odpowiedz>(r)),

  jednostka: (id: string) => fetch(`/api/jednostka/${encodeURIComponent(id)}`).then((r) => sprawdz<Jednostka>(r)),

  wgraj: (dane: FormData) => fetch('/api/wgraj', { method: 'POST', body: dane }).then((r) => sprawdz<Zadanie>(r)),

  opinia: (dane: { id_odpowiedzi: string; ocena: string; wlasciwy_przepis: string[]; komentarz: string }) =>
    fetch('/api/opinia', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dane),
    }).then((r) => sprawdz<{ id: string; status: string }>(r)),

  opinie: (status: string) => fetch(`/api/opinie?status=${status}`).then((r) => sprawdz<Opinia[]>(r)),

  przeglad: (id: string, dane: { status: string; wlasciwy_przepis?: string[]; musi_zawierac: string[]; przegladajacy: string }) =>
    fetch(`/api/opinie/${id}/przeglad`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(dane),
    }).then((r) => sprawdz<{ id: string; status: string }>(r)),

  eksport: () => fetch('/api/opinie/eksport').then((r) => sprawdz<{ liczba: number; jsonl: string }>(r)),
}
