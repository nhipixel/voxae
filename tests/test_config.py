"""Settings load from env with the VOXAE_ prefix."""

from voxae.config import Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.vlm_base_url.startswith("https://openrouter.ai")
    assert s.device == "cpu"
    assert s.demo_rate_limit_per_min > 0


def test_env_override(monkeypatch):
    monkeypatch.setenv("VOXAE_VLM_MODEL", "test/model-x")
    monkeypatch.setenv("VOXAE_DEVICE", "cuda")
    s = Settings(_env_file=None)
    assert s.vlm_model == "test/model-x"
    assert s.device == "cuda"
