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
