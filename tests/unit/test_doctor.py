from types import SimpleNamespace

from biblade_fusion.diagnostics import doctor


def test_realsense_enumeration_failure_is_a_warning(monkeypatch) -> None:
    fake_module = SimpleNamespace(context=lambda: (_ for _ in ()).throw(RuntimeError("no udev")))
    monkeypatch.setattr(doctor, "import_module", lambda _: fake_module)

    result = doctor._check_realsense()

    assert result.level is doctor.CheckLevel.WARN
    assert "enumeration unavailable" in result.message
