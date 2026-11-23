import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """G-3: nothing in the offline commands may open a network socket (Unix sockets are fine)."""
    real = socket.socket.connect

    def guarded(self, address, *args, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"network connection attempted: {address!r}")
        return real(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(AssertionError("DNS lookup attempted")))


@pytest.fixture(autouse=True)
def plenty_of_disk(monkeypatch):
    """Tests do not depend on the free space of the machine running them: the floor tests set
    `healing.clean.free_bytes` themselves."""
    for name in ("healing.clean.free_bytes", "healing.status.free_bytes"):
        monkeypatch.setattr(name, lambda *_: 1 << 50)


@pytest.fixture(autouse=True)
def no_notifications(monkeypatch):
    """The scheduled jobs' macOS notifications never show while tests run."""
    monkeypatch.setenv("HEALING_NO_NOTIFY", "1")
