import json

import pytest

from latch_eval_tools.harness._cli_runner import (
    _write_pi_custom_models_json,
    pi_custom_provider_config,
)


@pytest.mark.parametrize(
    ("model", "provider", "api_key"),
    [
        ("anthropic/claude-sonnet-5-5", "anthropic", "ANTHROPIC_API_KEY"),
        (
            "openrouter/moonshotai/kimi-k3",
            "openrouter",
            "OPENROUTER_API_KEY",
        ),
        (
            "openrouter/nvidia/nemotron-3-super-120b-a12b",
            "openrouter",
            "OPENROUTER_API_KEY",
        ),
        (
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b",
            "openrouter",
            "OPENROUTER_API_KEY",
        ),
        (
            "fireworks/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
            "fireworks",
            "FIREWORKS_API_KEY",
        ),
        (
            "fireworks/accounts/fireworks/models/nemotron-3-ultra-nvfp4",
            "fireworks",
            "FIREWORKS_API_KEY",
        ),
    ],
)
def test_pi_custom_model_config_written_for_provider(
    tmp_path, model, provider, api_key
):
    _write_pi_custom_models_json(tmp_path, model)

    path = tmp_path / ".pi" / "agent" / "models.json"
    actual = json.loads(path.read_text())
    assert actual == {"providers": {provider: pi_custom_provider_config(model)}}
    assert actual["providers"][provider]["apiKey"] == f"${api_key}"
    assert actual["providers"][provider]["models"][0]["id"] == model.removeprefix(
        f"{provider}/"
    )


def test_pi_custom_model_rejects_unregistered_model(tmp_path):
    with pytest.raises(ValueError, match="No pi fireworks model config registered"):
        _write_pi_custom_models_json(tmp_path, "fireworks/unknown")


def test_sonnet_55_uses_direct_anthropic_api_with_adaptive_thinking():
    config = pi_custom_provider_config("anthropic/claude-sonnet-5-5")
    assert config["baseUrl"] == "https://api.anthropic.com"
    assert config["api"] == "anthropic-messages"
    assert config["apiKey"] == "$ANTHROPIC_API_KEY"
    model = config["models"][0]
    assert model["thinkingLevelMap"] == dict.fromkeys(
        ["minimal", "low", "medium", "high", "xhigh", "max"], "max"
    )
    assert model["compat"]["forceAdaptiveThinking"] is True
    assert model["compat"]["supportsTemperature"] is False
