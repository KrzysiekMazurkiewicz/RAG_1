"""Klient Ollamy: gdy runner nie obsługuje schematu JSON (HTTP 501), degradacja do format=json
i dalej do samego promptu – bez błędu dla użytkownika i bez powtarzania odrzuconych żądań."""
from lexrag import llm


def test_degradacja_schematu_na_501(monkeypatch):
    monkeypatch.setattr(llm, "_OBSLUGA", {"schema": True, "json": True, "think": True})
    wywolania = []

    def fake_post(url, cialo, naglowki=None):
        wywolania.append(dict(cialo))
        if isinstance(cialo.get("format"), dict):
            raise llm._BladHTTP(501, '{"error":"structured output is unavailable"}')
        return {"message": {"content": '{"ok": true}'}, "done_reason": "stop", "prompt_eval_count": 10}

    monkeypatch.setattr(llm, "_post", fake_post)
    monkeypatch.setattr(llm, "tryb_api", lambda: "ollama")
    w = llm.czat_json([{"role": "user", "content": "x"}], model="m", schema={"type": "object"})
    assert w == {"ok": True}
    assert len(wywolania) == 2 and wywolania[1]["format"] == "json"
    # drugie wywołanie nie próbuje już schematu
    llm.czat_json([{"role": "user", "content": "y"}], model="m", schema={"type": "object"})
    assert len(wywolania) == 3 and wywolania[2]["format"] == "json"


def test_brak_think_dla_modelu_bez_myslenia(monkeypatch):
    monkeypatch.setattr(llm, "_OBSLUGA", {"schema": True, "json": True, "think": True})
    wywolania = []

    def fake_post(url, cialo, naglowki=None):
        wywolania.append(dict(cialo))
        if "think" in cialo:
            raise llm._BladHTTP(400, '{"error":"model does not support thinking"}')
        return {"message": {"content": '{"a": 1}'}}

    monkeypatch.setattr(llm, "_post", fake_post)
    monkeypatch.setattr(llm, "tryb_api", lambda: "ollama")
    w = llm.czat_json([{"role": "user", "content": "x"}], model="m", schema={"type": "object"})
    assert w == {"a": 1} and "think" not in wywolania[-1]
