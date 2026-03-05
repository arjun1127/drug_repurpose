# udp_viewer.py
import hashlib
import json
import socket
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

UDP_HOST = "0.0.0.0"
UDP_PORT = 5005
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000

history = deque(maxlen=500)
lock = threading.Lock()
latest_event = None
latest_prediction = None
last_msg_key = None
stats = {
    "packets_received": 0,
    "json_packets": 0,
    "non_json_packets": 0,
    "duplicate_packets": 0,
    "unique_packets": 0,
    "last_received_utc": None,
}


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def extract_msg_key(payload, raw_text):
    if isinstance(payload, dict):
        msg_id = payload.get("msg_id")
        event_name = payload.get("event")
        if msg_id is not None:
            return f"{event_name}:{msg_id}"
    return hashlib.sha1(raw_text.encode("utf-8", errors="replace")).hexdigest()


def udp_listener():
    global latest_event, latest_prediction, last_msg_key

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((UDP_HOST, UDP_PORT))
    except OSError as exc:
        print(
            f"[UDP] Failed to bind {UDP_HOST}:{UDP_PORT} ({exc}). "
            "Another process is likely using this port."
        )
        return

    print(f"[UDP] Listening on {UDP_HOST}:{UDP_PORT}")
    while True:
        data, addr = sock.recvfrom(65535)
        text = data.decode("utf-8", errors="replace")
        try:
            payload = json.loads(text)
            is_json = True
        except json.JSONDecodeError:
            payload = {"raw_text": text}
            is_json = False

        msg_key = extract_msg_key(payload, text)
        is_duplicate = msg_key == last_msg_key

        event = {
            "received_utc": now_utc(),
            "from": f"{addr[0]}:{addr[1]}",
            "bytes": len(data),
            "is_duplicate": is_duplicate,
            "msg_key": msg_key,
            "payload": payload,
        }

        with lock:
            history.appendleft(event)
            latest_event = event
            stats["packets_received"] += 1
            stats["last_received_utc"] = event["received_utc"]
            if is_json:
                stats["json_packets"] += 1
            else:
                stats["non_json_packets"] += 1
            if is_duplicate:
                stats["duplicate_packets"] += 1
            else:
                stats["unique_packets"] += 1
                latest_prediction = payload
            last_msg_key = msg_key

        dup_tag = " DUP" if is_duplicate else ""
        print(
            f"[UDP] {event['received_utc']} from {event['from']} "
            f"({event['bytes']} bytes){dup_tag}"
        )


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        body = json.dumps(obj, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, code=200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/latest":
            with lock:
                latest = latest_event
            self._send_json({"latest": latest})

        elif parsed.path == "/latest_prediction":
            with lock:
                payload = latest_prediction
            self._send_json({"latest_prediction": payload})

        elif parsed.path == "/stats":
            with lock:
                snapshot = dict(stats)
                snapshot["history_size"] = len(history)
            self._send_json(snapshot)

        elif parsed.path == "/history":
            query = parse_qs(parsed.query)
            limit = int(query.get("limit", ["50"])[0])
            limit = max(1, min(limit, 500))
            with lock:
                items = list(history)[:limit]
            self._send_json({"count": len(items), "items": items})

        elif parsed.path == "/":
            html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>UDP Predictions Viewer</title>
  <style>
    body { font-family: sans-serif; margin: 24px; }
    pre { background: #111; color: #eee; padding: 16px; border-radius: 8px; overflow: auto; }
  </style>
</head>
<body>
  <h2>UDP Predictions Viewer</h2>
  <p>Polling <code>/stats</code> + <code>/latest_prediction</code> every 1 second.</p>
  <h3>Stats</h3>
  <pre id="stats">Waiting...</pre>
  <h3>Latest Non-Duplicate Prediction</h3>
  <pre id="pred">Waiting for UDP packets...</pre>
  <script>
    async function refresh() {
      try {
        const statsResp = await fetch('/stats');
        const statsData = await statsResp.json();
        document.getElementById('stats').textContent = JSON.stringify(statsData, null, 2);

        const predResp = await fetch('/latest_prediction');
        const predData = await predResp.json();
        document.getElementById('pred').textContent = JSON.stringify(predData, null, 2);
      } catch (e) {
        document.getElementById('pred').textContent = 'Error: ' + e;
      }
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""
            self._send_html(html)
        else:
            self._send_json({"error": "not found"}, code=404)


def main():
    t = threading.Thread(target=udp_listener, daemon=True)
    t.start()
    print(f"[HTTP] Serving on http://{HTTP_HOST}:{HTTP_PORT}")
    ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
