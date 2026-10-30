"""Hypothesis Registry: единый список гипотез, их статусов и всех проверок (hypotheses/registry.json).

Статусы: Testing → Rejected | Verified; Archived — снята вручную.
Правила:
- нельзя зарегистрировать гипотезу с отпечатком уже существующей (та же идея под другим именем);
- нельзя запустить проверку, если YAML изменён после регистрации (замок не совпал);
- holdout — один раз на гипотезу и только после пройденной валидации.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qlab.hypothesis import Hypothesis

STATUSES = ("Testing", "Rejected", "Verified", "Archived")


class RegistryError(RuntimeError):
    pass


class Registry:
    def __init__(self, path: str | Path = "hypotheses/registry.json"):
        self.path = Path(path)
        self.data: dict[str, Any] = (json.loads(self.path.read_text(encoding="utf-8"))
                                     if self.path.exists() else {"hypotheses": {}})

    @property
    def items(self) -> dict[str, dict]:
        return self.data["hypotheses"]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def next_id(self) -> str:
        nums = [int(k[1:]) for k in self.items if k[1:].isdigit()]
        return f"H{max(nums, default=0) + 1:03d}"

    def register(self, h: Hypothesis, *, status: str = "Testing", reason: str = "") -> dict:
        if h.id in self.items:
            raise RegistryError(f"{h.id} уже зарегистрирована")
        fp = h.fingerprint()
        for other_id, other in self.items.items():
            if other["fingerprint"] == fp:
                raise RegistryError(f"{h.id} повторяет {other_id} ({other['name']}): та же идея под другим названием")
        entry = {"name": h.name, "path": str(h.path), "source": h.source, "created": str(h.created),
                 "registered_at": _now(), "fingerprint": fp, "lock": h.lock(),
                 "status": status, "reason": reason, "holdout_used": False, "checks": []}
        self.items[h.id] = entry
        self.save()
        return entry

    def require_unchanged(self, h: Hypothesis) -> None:
        entry = self.items.get(h.id)
        if entry is None:
            raise RegistryError(f"{h.id} не зарегистрирована: сначала `qlab register {h.path}`")
        if entry["lock"] != h.lock():
            raise RegistryError(f"{h.id}: условия изменены после регистрации. Это запрещено — создайте новую гипотезу")
        if entry["status"] != "Testing":
            raise RegistryError(f"{h.id}: статус {entry['status']}, проверки закрыты")

    def open_holdout(self, h: Hypothesis) -> None:
        entry = self.items[h.id]
        if entry["holdout_used"]:
            raise RegistryError(f"{h.id}: holdout уже использован")
        if not any(c["split"] == "validation" and c["passed"] for c in entry["checks"]):
            raise RegistryError(f"{h.id}: holdout доступен только после пройденной валидации")
        entry["holdout_used"] = True
        entry["holdout_opened_at"] = _now()
        self.save()

    def record(self, h: Hypothesis, split: str, passed: bool, reason: str, report: str) -> None:
        entry = self.items[h.id]
        entry["checks"].append({"time": _now(), "split": split, "passed": passed, "reason": reason, "report": report})
        if not passed:
            entry["status"], entry["reason"] = "Rejected", f"{split}: {reason}"
        elif split == "holdout":
            entry["status"], entry["reason"] = "Verified", "пройдены validation и holdout"
        self.save()

    def set_status(self, hid: str, status: str, reason: str) -> None:
        if status not in STATUSES:
            raise RegistryError(f"статус должен быть одним из {STATUSES}")
        self.items[hid]["status"], self.items[hid]["reason"] = status, reason
        self.save()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
