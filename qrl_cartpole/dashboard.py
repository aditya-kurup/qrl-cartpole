"""Serve a local interactive dashboard for a saved CartPole experiment."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PAGE = Path(__file__).with_name("dashboard.html")


def create_handler(results_path: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/" or self.path == "/index.html":
                self.send_file(PAGE, "text/html; charset=utf-8")
            elif self.path == "/results":
                try:
                    payload = results_path.read_bytes()
                except OSError:
                    self.send_error(404, "Results JSON not found. Run the experiment first.")
                    return
                self.send_bytes(payload, "application/json")
            elif self.path == "/circuit.png":
                circuit = results_path.parent / "variational_circuit.png"
                if not circuit.is_file():
                    self.send_error(404, "Circuit figure not found. Run the experiment first.")
                    return
                self.send_file(circuit, "image/png")
            else:
                self.send_error(404)

        def send_file(self, path: Path, content_type: str) -> None:
            self.send_bytes(path.read_bytes(), content_type)

        def send_bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

    return DashboardHandler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/final/results.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    results_path = args.results.resolve()
    if not PAGE.is_file():
        parser.error(f"Dashboard page is missing: {PAGE}")
    if not results_path.is_file():
        parser.error(f"Results file not found: {results_path}. Run the experiment first.")
    with results_path.open() as handle:
        json.load(handle)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(results_path))
    print(f"Dashboard: http://{args.host}:{args.port}/")
    print(f"Results:   {results_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
