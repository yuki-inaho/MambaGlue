"""Training-side configuration and registry tests (extractor, matcher, optimizer)."""

from __future__ import annotations

import torch


def test_rgbd_extractor_two_channel_forward():
    from gluefactory.models import get_model

    model = get_model("mambaglue.training.extractor_rgbd")(
        {"max_num_keypoints": 32, "force_num_keypoints": True}
    ).eval()
    assert model.backbone[0][0].conv.in_channels == 2

    image = torch.rand(1, 2, 64, 64)
    with torch.inference_mode():
        pred = model({"image": image, "image_size": torch.tensor([[64, 64]])})
    assert pred["descriptors"].shape == (1, 32, 256)
    assert pred["keypoints"].shape == (1, 32, 2)


def _tiny_matcher_conf(**overrides):
    conf = {"n_layers": 2, "num_heads": 4, "flash": False}
    conf.update(overrides)
    return conf


def test_matcher_weights_loaded_and_prefix_stripped(tmp_path):
    from mambaglue.training.matcher import MambaGlueMatcher

    source = MambaGlueMatcher(_tiny_matcher_conf())
    prefixed = {"matcher." + key: value for key, value in source.state_dict().items()}
    path = tmp_path / "mambaglue_weights.pt"
    torch.save({"model": prefixed}, path)

    target = MambaGlueMatcher(_tiny_matcher_conf(weights=str(path)))
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_matcher_weights_mismatch_fails_closed(tmp_path):
    import pytest

    from mambaglue.training.matcher import MambaGlueMatcher

    source = MambaGlueMatcher(_tiny_matcher_conf())
    incomplete = {
        key: value
        for key, value in source.state_dict().items()
        if "transformermambas.0" not in key
    }
    path = tmp_path / "incomplete.pt"
    torch.save(incomplete, path)

    with pytest.raises(ValueError):
        MambaGlueMatcher(_tiny_matcher_conf(weights=str(path)))


def test_matcher_weights_none_skips_loading(tmp_path, monkeypatch):
    from mambaglue.training import matcher as matcher_module

    def _fail(*args, **kwargs):
        raise AssertionError("weights must not be fetched when weights is None")

    monkeypatch.setattr(matcher_module.torch.hub, "load_state_dict_from_url", _fail)
    matcher_module.MambaGlueMatcher(_tiny_matcher_conf())
