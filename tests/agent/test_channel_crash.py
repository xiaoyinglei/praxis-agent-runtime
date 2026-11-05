import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_runtime.harness import RuntimeStore
from agent_runtime.harness import channel_ops as ch
from tests.agent.test_channel_runtime import channel_setup


def test_killed_sender_after_real_loopback_acceptance_cannot_repeat(tmp_path: Path):
    database = tmp_path / "r.db"
    with RuntimeStore.open_sync(database) as store:
        _, start, grant = channel_setup(store, tmp_path)
        note = ch.enqueue_notification(store, grant["grant_id"], start.turn_id, {"status": "ready"}, now=100)
    accepted = threading.Event()
    release = threading.Event()
    effects = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            effects.append(self.rfile.read(int(self.headers["Content-Length"])))
            accepted.set()
            release.wait(5)
            try:
                self.send_response(201)
                self.end_headers()
                self.wfile.write(b'{"id": 99}')
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = """
import socket, sys, time
from pathlib import Path
from agent_runtime.harness import RuntimeStore, channel_ops as ch
with RuntimeStore.open_sync(Path(sys.argv[1])) as store:
    claim=ch.claim_notification(store,owner='killed',now=100,lease_seconds=1)
    ch.begin_send(store,claim,now=100)
    sock=socket.create_connection(('127.0.0.1',int(sys.argv[2])))
    sock.sendall(b'POST / HTTP/1.1\\r\\nHost: localhost\\r\\nContent-Length: 4\\r\\n\\r\\nwork')
    sock.recv(10000)
    time.sleep(30)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(database), str(server.server_port)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    try:
        assert accepted.wait(5)
        child.kill()
        child.wait(timeout=5)
        with RuntimeStore.open_sync(database) as store:
            assert ch.claim_notification(store, owner="replacement", now=102) is None
            assert ch.get(store, "outbox", note["outbox_id"])["status"] == "uncertain"
            assert effects == [b"work"]
            assert store.verify().valid
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
