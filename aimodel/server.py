"""The mobile app's backend: a small web server (stdlib only) that serves the app and drives the model.

    python -m aimodel.server                 # this device only: http://127.0.0.1:8765
    python -m aimodel.server --host 0.0.0.0  # let your phone on the same Wi-Fi connect (uses an access token)
"""

from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import secrets
import socket
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .datasets import import_dataset
from .model import LearningModel

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
MAX_BODY = 25_000_000  # bytes: the biggest dataset upload
NET_ERRORS = (urllib.error.URLError, TimeoutError, OSError, ValueError)


class Job:
    """One background task (training / reading a dataset) whose progress the app polls."""

    def __init__(self) -> None:
        self.state = {"running": False, "kind": None, "done": 0, "total": 0, "message": "", "result": None,
                      "error": None}

    def snapshot(self) -> dict:
        return dict(self.state)


class AppState:
    def __init__(self, brain: str, online: bool = True) -> None:
        self.brain = brain
        self.online = online
        self.lock = threading.RLock()  # the model is not thread-safe: one operation at a time
        self.model = LearningModel.load(brain)
        sibling = os.path.join(os.path.dirname(os.path.abspath(brain)), "writer.npz")
        if self.model.writer is None and os.path.exists(sibling):
            try:
                self.model.load_writer(sibling)
            except (OSError, ValueError, KeyError):
                pass
        self.job = Job()

    # -- helpers
    def save(self) -> None:
        self.model.save(self.brain)

    def overview(self) -> dict:
        m = self.model
        return {"stats": m.stats(), "online": self.online, "job": self.job.snapshot(),
                "name": m.user_name() or None, "writer": m.writer is not None and m.writer_enabled}

    # -- chat
    def chat(self, text: str) -> dict:
        m = self.model
        with self.lock:
            try:
                reply, confidence = m.respond(text, use_web=self.online)
            except NET_ERRORS:
                reply, confidence = m.respond(text)
            source = m.last_source
            trace = [{"kind": s["kind"], "text": s["text"], "source": s["source"]}
                     for s in m.last_trace if s.get("source") != "my reasoning" or s["kind"] == "inference"]
            self.save()
        return {"reply": reply, "confidence": round(float(confidence or 0), 2), "source": source, "trace": trace}

    def rate(self, good: bool, better: str = "") -> dict:
        m = self.model
        with self.lock:
            if m.last_reply is None:
                return {"ok": False, "message": "Nothing to rate yet."}
            if good:
                m.feedback(True)
                msg = "Thanks! I'll remember that."
            elif better.strip():
                m.correct(m.last_query, better.strip())
                msg = "Thanks, I corrected myself."
            else:
                m.feedback(False)
                msg = "Okay, I'll trust that answer less."
            self.save()
        return {"ok": True, "message": msg}

    def teach(self, prompt: str, response: str) -> dict:
        with self.lock:
            self.model.learn(prompt, response)
            self.save()
        return {"ok": True, "message": "Got it, I learned that."}

    # -- background work
    def start_job(self, kind: str, work) -> bool:
        """Run work(progress) in a thread. Returns False if another job is already running."""
        with self.lock:
            if self.job.state["running"]:
                return False
            self.job.state.update(running=True, kind=kind, done=0, total=0, message="Starting...",
                                  result=None, error=None)

        def progress(done: int, total: int, message: str) -> None:
            self.job.state.update(done=done, total=total, message=message)

        def run() -> None:
            try:
                with self.lock:
                    result = work(progress)
                    self.save()
                total = self.job.state["total"] or 1
                self.job.state.update(result=result, message="Done", done=total, total=total)
            except NET_ERRORS as e:
                self.job.state.update(error=f"Couldn't reach it: {e}")
            except Exception as e:  # keep the server alive; show the problem in the app
                self.job.state.update(error=f"{type(e).__name__}: {e}")
            finally:
                self.job.state["running"] = False

        threading.Thread(target=run, daemon=True).start()
        return True

    def add_dataset(self, name: str, text: str) -> bool:
        return self.start_job("dataset", lambda progress: import_dataset(self.model, name, text, progress))

    def add_url(self, url: str) -> bool:
        def work(progress):
            progress(0, 1, "Reading the page")
            n = self.model.read(url)
            progress(1, 1, "Done")
            return {"name": url, "format": "web page", "answers": 0, "sentences": n}
        return self.start_job("dataset", work)

    def train(self, epochs: int) -> bool:
        def work(progress):
            loss = None
            for i in range(epochs):
                progress(i, epochs, f"Practice round {i + 1} of {epochs}")
                loss = self.model.retrain(1)
            progress(epochs, epochs, "Done")
            return {"epochs": epochs, "loss": None if loss is None else round(float(loss), 3)}
        return self.start_job("train", work)

    def study(self) -> bool:
        def work(progress):
            progress(0, 3, "Looking into open questions")
            result = self.model.study_session(use_web=self.online)
            progress(3, 3, "Done")
            return {"exam": {"score": result["exam"]["score"], "total": result["exam"]["total"]},
                    "curious": len(result["curiosity"]), "slept": result["sleep"]["kept"]}
        return self.start_job("study", work)

    # -- reports
    def quiz(self, n: int) -> dict:
        with self.lock:
            result = self.model.quiz(n)
            self.save()
        return {"score": result["score"], "total": result["total"],
                "results": [{"q": r["q"], "reply": r["reply"], "answer": r["answer"], "correct": r["correct"]}
                            for r in result["results"]]}

    def facts(self, topic: str) -> list[str]:
        from .reasoning import state, stem
        words = {stem(w) for w in topic.lower().split()}
        with self.lock:
            found = [f for f in self.model.facts
                     if not words or words & {stem(t) for t in f["subj"].lower().split() + f["obj"].lower().split()}]
            return [state(f) for f in found[-100:]][::-1]

    def memories(self) -> list[dict]:
        with self.lock:
            return [{"prompt": m["prompt"], "response": m["response"]} for m in self.model.memories[-100:]][::-1]

    def forget(self, text: str) -> int:
        with self.lock:
            n = self.model.forget(text)
            self.save()
        return n

    def evaluate(self, quick: bool) -> bool:
        def work(progress):
            from .evaluation import evaluate_all, format_report, history_path, load_history, save_run
            progress(0, 1, "Running the standard tests")
            hist = history_path(self.brain)
            report = evaluate_all(self.model, self.brain, quick=quick)
            runs = load_history(hist)
            text = format_report(report, runs[-1] if runs else None)
            save_run(hist, report)
            return {"report": text}
        return self.start_job("evaluate", work)


def make_handler(state: AppState, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Aimodel"

        def log_message(self, fmt, *args):  # quiet
            pass

        # -- plumbing
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200) -> None:
            self._send(code, json.dumps(data).encode(), "application/json", {"Cache-Control": "no-store"})

        def _authorized(self) -> bool:
            if not token:
                return True
            given = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            return hmac.compare_digest(given, token)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("That file is too big (limit 25 MB).")
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8", errors="replace") or "{}")
            if not isinstance(data, dict):
                raise ValueError("bad request")
            return data

        # -- routes
        def do_GET(self) -> None:
            path = urlparse(self.path)
            if path.path.startswith("/api/"):
                if not self._authorized():
                    return self._json({"error": "Not authorized. Open the link the server printed."}, 401)
                try:
                    return self._api_get(path.path, parse_qs(path.query))
                except Exception as e:
                    return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            self._static(path.path)

        def do_POST(self) -> None:
            if not self.path.startswith("/api/"):
                return self._json({"error": "not found"}, 404)
            if not self._authorized():
                return self._json({"error": "Not authorized. Open the link the server printed."}, 401)
            try:
                body = self._body()
                return self._api_post(self.path, body)
            except ValueError as e:
                return self._json({"error": str(e)}, 400)
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _static(self, route: str) -> None:
            name = "index.html" if route in ("/", "") else route.lstrip("/")
            full = os.path.realpath(os.path.join(APP_DIR, name))
            if not full.startswith(os.path.realpath(APP_DIR) + os.sep) or not os.path.isfile(full):
                return self._send(404, b"Not found", "text/plain")
            with open(full, "rb") as f:
                data = f.read()
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
                ctype += "; charset=utf-8"
            self._send(200, data, ctype, {"Cache-Control": "no-cache"})

        def _api_get(self, route: str, q: dict) -> None:
            if route == "/api/state":
                with state.lock:
                    return self._json(state.overview())
            if route == "/api/job":
                return self._json(state.job.snapshot())
            if route == "/api/facts":
                return self._json({"facts": state.facts((q.get("topic") or [""])[0])})
            if route == "/api/memories":
                return self._json({"memories": state.memories()})
            self._json({"error": "not found"}, 404)

        def _api_post(self, route: str, b: dict) -> None:
            text = lambda key: str(b.get(key, "")).strip()  # noqa: E731
            if route == "/api/chat":
                if not text("text"):
                    raise ValueError("Say something first.")
                return self._json(state.chat(text("text")))
            if route == "/api/rate":
                return self._json(state.rate(bool(b.get("good")), text("better")))
            if route == "/api/teach":
                if not text("prompt") or not text("response"):
                    raise ValueError("Both a question and an answer are needed.")
                return self._json(state.teach(text("prompt"), text("response")))
            if route == "/api/dataset":
                if text("url"):
                    if not text("url").startswith(("http://", "https://")):
                        raise ValueError("The link must start with http:// or https://")
                    started = state.add_url(text("url"))
                elif text("text"):
                    started = state.add_dataset(text("name") or "pasted text", str(b["text"]))
                else:
                    raise ValueError("Choose a file, paste text, or give a link.")
                return self._json({"started": started, "message": "" if started else "Another task is running."},
                                  200 if started else 409)
            if route == "/api/train":
                epochs = max(1, min(int(b.get("epochs") or 5), 50))
                started = state.train(epochs)
                return self._json({"started": started}, 200 if started else 409)
            if route == "/api/study":
                started = state.study()
                return self._json({"started": started}, 200 if started else 409)
            if route == "/api/evaluate":
                started = state.evaluate(bool(b.get("quick", True)))
                return self._json({"started": started}, 200 if started else 409)
            if route == "/api/quiz":
                return self._json(state.quiz(max(1, min(int(b.get("n") or 5), 20))))
            if route == "/api/forget":
                if not text("text"):
                    raise ValueError("Say what to forget.")
                return self._json({"forgot": state.forget(text("text"))})
            if route == "/api/online":
                state.online = bool(b.get("on"))
                return self._json({"online": state.online})
            if route == "/api/writer":
                with state.lock:
                    state.model.writer_enabled = bool(b.get("on"))
                    state.save()
                return self._json({"ok": True})
            self._json({"error": "not found"}, 404)

    return Handler


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; it just picks the Wi-Fi interface
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def serve(brain: str, host: str, port: int, online: bool, token: str | None) -> ThreadingHTTPServer:
    state = AppState(brain, online)
    httpd = ThreadingHTTPServer((host, port), make_handler(state, token))
    httpd.state = state  # type: ignore[attr-defined]
    return httpd


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the Aimodel app (open it in your phone's browser).")
    ap.add_argument("--brain", default="brain.json", help="where to save what it learns")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 = this device only; 0.0.0.0 = other devices on your Wi-Fi too")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--offline", action="store_true", help="never look things up on the web")
    ap.add_argument("--token", help="access token for other devices (made up for you if omitted)")
    args = ap.parse_args(argv)

    local = args.host in ("127.0.0.1", "localhost", "::1")
    token = None if local else (args.token or secrets.token_urlsafe(9))
    httpd = serve(args.brain, args.host, args.port, not args.offline, token)
    where = "127.0.0.1" if local else lan_address()
    suffix = f"/?token={token}" if token else "/"
    print(f"Aimodel app is running. Open this on your phone's browser:\n\n    http://{where}:{args.port}{suffix}\n")
    if not local:
        print("Anyone with that link can use and teach your model, so keep it to your own Wi-Fi.")
    print("Then use the browser menu: 'Add to Home screen' to install it. Ctrl+C stops it.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.state.save()  # type: ignore[attr-defined]
        httpd.server_close()
        print("\nSaved. Bye!")


if __name__ == "__main__":
    main()
