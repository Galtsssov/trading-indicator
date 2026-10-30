import shutil

import pytest
import yaml

from qlab.hypothesis import Hypothesis
from qlab.registry import Registry, RegistryError


def copy_h(tmp_path, src, **changes):
    raw = yaml.safe_load(open(src, encoding="utf-8"))
    raw.update(changes)
    p = tmp_path / f"{raw['id']}.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return p


def test_duplicate_under_other_name_rejected(tmp_path):
    reg = Registry(tmp_path / "r.json")
    reg.register(Hypothesis.load("hypotheses/H002-sweep-continuation.yaml"))
    twin = copy_h(tmp_path, "hypotheses/H002-sweep-continuation.yaml", id="H900", name="Совсем другое имя",
                  description="другое описание")
    with pytest.raises(RegistryError, match="повторяет H002"):
        reg.register(Hypothesis.load(twin))
    other = copy_h(tmp_path, "hypotheses/H002-sweep-continuation.yaml", id="H901", direction="against")
    reg.register(Hypothesis.load(other))  # другое направление — другая гипотеза


def test_changes_after_registration_blocked(tmp_path):
    p = copy_h(tmp_path, "hypotheses/H002-sweep-continuation.yaml")
    reg = Registry(tmp_path / "r.json")
    reg.register(Hypothesis.load(p))
    reg.require_unchanged(Hypothesis.load(p))
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    raw["criteria"]["practical_min_delta_r"] = 0.01
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(RegistryError, match="изменены после регистрации"):
        reg.require_unchanged(Hypothesis.load(p))


def test_holdout_requires_validation_and_is_one_shot(tmp_path):
    p = copy_h(tmp_path, "hypotheses/H002-sweep-continuation.yaml")
    reg = Registry(tmp_path / "r.json")
    h = Hypothesis.load(p)
    reg.register(h)
    with pytest.raises(RegistryError, match="после пройденной валидации"):
        reg.open_holdout(h)
    reg.record(h, "validation", True, "ok", "x.md")
    reg.open_holdout(h)
    with pytest.raises(RegistryError, match="уже использован"):
        reg.open_holdout(h)
    reg.record(h, "holdout", False, "провал", "y.md")
    assert reg.items["H002"]["status"] == "Rejected"
    with pytest.raises(RegistryError, match="проверки закрыты"):
        reg.require_unchanged(h)
