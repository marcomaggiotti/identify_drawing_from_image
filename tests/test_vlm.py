import pytest

from drawing_identifier.config import VLMProfile, load_config
from drawing_identifier.vlm import MockBackend, VLMRegistry, extract_json, parse_spec


@pytest.mark.parametrize(
    "spec,provider,model,base",
    [
        ("anthropic:claude-opus-5-5", "anthropic", "claude-opus-5-5", None),
        ("ollama:qwen2.5vl:7b", "ollama", "qwen2.5vl:7b", "http://localhost:11434"),
        ("ollama:llava@http://gpu:11434", "ollama", "llava", "http://gpu:11434"),
        ("vllm:Qwen/Qwen2.5-VL-7B-Instruct", "openai_compatible", "Qwen/Qwen2.5-VL-7B-Instruct", "http://localhost:8000/v1"),
        ("hf:Qwen/Qwen2.5-VL-3B-Instruct", "hf", "Qwen/Qwen2.5-VL-3B-Instruct", None),
    ],
)
def test_parse_spec(spec, provider, model, base):
    _, prof = parse_spec(spec)
    assert prof.provider == provider and prof.model == model and prof.base_url == base


def test_parse_spec_errors():
    with pytest.raises(ValueError):
        parse_spec("nonsense")
    with pytest.raises(ValueError):
        parse_spec("foo:bar")


def test_extract_json_tolerant():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": [1, 2,], "b": "x}"} hope that helps') == {"a": [1, 2], "b": "x}"}
    assert extract_json("[1, 2]") == [1, 2]
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_registry_selection(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_config(overrides={"default_vlm": "none", "agent_vlms": {"verifier": "mock"}})
    reg = VLMRegistry(cfg)
    assert reg.for_agent("text_reader") is None
    assert isinstance(reg.for_agent("verifier"), MockBackend)
    # ad-hoc spec
    assert reg.get("mock:anything").model == "anything"
    # explicit override instance wins
    m = MockBackend(VLMProfile(provider="mock", model="m"))
    reg2 = VLMRegistry(cfg, {"*": m})
    assert reg2.for_agent("text_reader") is m


def test_mock_generate_json():
    m = MockBackend(VLMProfile(provider="mock", model="m"), responder=lambda p, ims: {"ok": len(ims)})
    import numpy as np

    assert m.generate_json("hi", [np.zeros((5, 5, 3), np.uint8)]) == {"ok": 1}
    assert m.calls == 1
