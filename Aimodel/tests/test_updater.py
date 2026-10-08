import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aimodel.server import AppState
from aimodel.updater import VERSION_FILE, UpdateError, Updater

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@t", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}


def git(cwd, *args):
    out = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True, env=ENV)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


class GitRepos:
    """A "GitHub" (bare repo), the copy the app runs from, and a second copy to push new code from."""

    def __init__(self):
        self.dir = tempfile.mkdtemp()
        self.remote = os.path.join(self.dir, "remote.git")
        subprocess.run(["git", "init", "--bare", "-b", "main", self.remote], capture_output=True, env=ENV, check=True)
        self.dev = os.path.join(self.dir, "dev")
        subprocess.run(["git", "clone", self.remote, self.dev], capture_output=True, env=ENV, check=True)
        git(self.dev, "checkout", "-b", "main")
        self.write(self.dev, "app.txt", "one")
        self.push("first version")
        self.app = os.path.join(self.dir, "app")
        subprocess.run(["git", "clone", "-b", "main", self.remote, self.app], capture_output=True, env=ENV, check=True)

    def write(self, where, name, text):
        with open(os.path.join(where, name), "w") as f:
            f.write(text)

    def push(self, message):
        git(self.dev, "add", "-A")
        git(self.dev, "commit", "-q", "-m", message)
        git(self.dev, "push", "-q", "origin", "main")

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestGitUpdates(unittest.TestCase):
    def setUp(self):
        self.repos = GitRepos()
        os.environ.update({k: v for k, v in ENV.items() if k.startswith("GIT_")})
        self.updater = Updater(self.repos.app)

    def tearDown(self):
        self.repos.close()

    def test_up_to_date(self):
        st = self.updater.check()
        self.assertEqual((st["mode"], st["available"], st["error"]), ("git", False, None))
        self.assertEqual(st["current"]["subject"], "first version")

    def test_new_code_pushed_is_found_and_brought_in(self):
        self.repos.write(self.repos.dev, "app.txt", "two")
        self.repos.push("second version")
        self.repos.write(self.repos.dev, "app.txt", "three")
        self.repos.push("third version")
        st = self.updater.check()
        self.assertTrue(st["available"] and st["can_update"])
        self.assertEqual((st["behind"], st["changes"]), (2, ["third version", "second version"]))
        old = st["current"]["commit"]
        result = self.updater.apply()
        self.assertTrue(result["updated"])
        self.assertNotEqual(result["current"]["commit"], old)
        with open(os.path.join(self.repos.app, "app.txt")) as f:
            self.assertEqual(f.read(), "three")
        self.assertFalse(self.updater.check()["available"])

    def test_local_changes_are_never_overwritten(self):
        self.repos.write(self.repos.dev, "app.txt", "two")
        self.repos.push("second version")
        self.repos.write(self.repos.app, "app.txt", "my own edit")
        st = self.updater.check()
        self.assertTrue(st["available"])
        self.assertFalse(st["can_update"])
        self.assertIn("changed files", st["why_not"])
        with self.assertRaises(UpdateError):
            self.updater.apply()
        with open(os.path.join(self.repos.app, "app.txt")) as f:
            self.assertEqual(f.read(), "my own edit")

    def test_a_copy_with_its_own_commits_is_not_forced(self):
        self.repos.write(self.repos.app, "mine.txt", "x")
        git(self.repos.app, "add", "-A")
        git(self.repos.app, "commit", "-q", "-m", "mine")
        self.repos.write(self.repos.dev, "app.txt", "two")
        self.repos.push("second version")
        st = self.updater.check()
        self.assertFalse(st["can_update"])
        self.assertIn("not on GitHub", st["why_not"])

    def test_untracked_files_like_the_brain_do_not_block_an_update(self):
        self.repos.write(self.repos.app, "brain.json", "{}")
        self.repos.write(self.repos.dev, "app.txt", "two")
        self.repos.push("second version")
        self.assertTrue(self.updater.check()["can_update"])
        self.updater.apply()
        self.assertTrue(os.path.exists(os.path.join(self.repos.app, "brain.json")))

    def test_no_connection_is_reported_not_raised(self):
        git(self.repos.app, "remote", "set-url", "origin", os.path.join(self.repos.dir, "gone.git"))
        st = self.updater.check()
        self.assertTrue(st["error"])
        self.assertFalse(st["available"])


class TestAppUpdates(unittest.TestCase):
    def setUp(self):
        self.repos = GitRepos()
        os.environ.update({k: v for k, v in ENV.items() if k.startswith("GIT_")})
        self.dir = tempfile.mkdtemp()
        self.state = AppState(os.path.join(self.dir, "brain.json"), online=False)
        self.state.updater = Updater(self.repos.app)
        self.restarted = threading.Event()
        self.state.restart = self.restarted.set

    def tearDown(self):
        self.repos.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def push_new(self):
        self.repos.write(self.repos.dev, "app.txt", "two")
        self.repos.push("a change")

    def test_apply_pulls_then_restarts(self):
        self.push_new()
        self.assertTrue(self.state.update_status(check=True)["available"])
        result = self.state.apply_update()
        self.assertEqual((result["updated"], result["restarting"]), (True, True))
        self.assertTrue(self.restarted.wait(5))

    def test_nothing_new_means_no_restart(self):
        result = self.state.apply_update()
        self.assertEqual((result["updated"], result["restarting"]), (False, False))
        self.assertFalse(self.restarted.wait(1.5))

    def test_a_running_task_blocks_the_update(self):
        self.push_new()
        self.state.job.state["running"] = True
        with self.assertRaises(UpdateError):
            self.state.apply_update()
        self.assertFalse(self.restarted.is_set())

    def test_automatic_updates_follow_the_switch(self):
        self.push_new()
        self.state.set_auto_update(False)
        off = threading.Event()
        threading.Thread(target=self.state.auto_update_loop, args=(off, 0.05), daemon=True).start()
        self.assertFalse(self.restarted.wait(1.0))  # switched off: new code is left alone
        off.set()
        self.state.set_auto_update(True)
        self.assertEqual(self.state._read_settings(), {"auto_update": True})
        threading.Thread(target=self.state.auto_update_loop, args=(threading.Event(), 0.05), daemon=True).start()
        self.assertTrue(self.restarted.wait(8))  # switched on: it pulls and restarts

    def test_automatic_updates_wait_for_a_running_task(self):
        self.push_new()
        self.state.job.state["running"] = True
        threading.Thread(target=self.state.auto_update_loop, args=(threading.Event(), 0.05), daemon=True).start()
        self.assertFalse(self.restarted.wait(1.0))

    def test_the_setting_is_remembered(self):
        self.state.set_auto_update(False)
        again = AppState(os.path.join(self.dir, "brain.json"), online=False)
        self.assertFalse(again.auto_update)


class FakeGitHub(BaseHTTPRequestHandler):
    sha = "abc1234def5678"
    files = {"app.txt": b"new", "aimodel/x.py": b"print(1)"}

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/sha":
            body = self.sha.encode()
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for name, data in self.files.items():
                    z.writestr("Aimodel-main/" + name, data)
                z.writestr("Aimodel-main/../evil.txt", b"escape")
                z.writestr("Aimodel-main/.git/config", b"nope")
            body = buf.getvalue()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestDownloadedCopy(unittest.TestCase):
    def setUp(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHub)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.root = tempfile.mkdtemp()
        self.outer = os.path.dirname(self.root)
        with open(os.path.join(self.root, "brain.json"), "w") as f:
            f.write('{"mine": true}')
        self.updater = Updater(self.root, api_url=base + "/sha", zip_url=base + "/zip")

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_it_is_a_download_install_without_git(self):
        self.assertEqual(self.updater.mode, "download")
        self.assertEqual(self.updater.current()["commit"], "unknown")

    def test_update_copies_the_new_files_and_keeps_yours(self):
        st = self.updater.check()
        self.assertTrue(st["available"] and st["can_update"])
        self.assertTrue(self.updater.apply()["updated"])
        with open(os.path.join(self.root, "app.txt"), "rb") as f:
            self.assertEqual(f.read(), b"new")
        self.assertTrue(os.path.exists(os.path.join(self.root, "aimodel", "x.py")))
        with open(os.path.join(self.root, "brain.json")) as f:
            self.assertEqual(f.read(), '{"mine": true}')  # your brain is not in the ZIP
        self.assertEqual(self.updater.current()["commit"], "abc1234")
        self.assertFalse(os.path.exists(os.path.join(self.root, ".git")))
        self.assertFalse(os.path.exists(os.path.join(self.outer, "evil.txt")))  # a ../ path in the ZIP is ignored
        self.assertFalse(self.updater.check()["available"])  # and now it knows it is current

    def test_a_new_version_on_github_is_noticed_later(self):
        self.updater.check()
        self.updater.apply()
        FakeGitHub.sha = "9999999aaaaaaa"
        try:
            self.assertTrue(self.updater.check()["available"])
        finally:
            FakeGitHub.sha = "abc1234def5678"

    def test_unreachable_github_is_an_error_message(self):
        broken = Updater(self.root, api_url="http://127.0.0.1:9/sha", zip_url="http://127.0.0.1:9/zip")
        self.assertTrue(broken.check()["error"])


if __name__ == "__main__":
    unittest.main()


class TestCheckThrottle(unittest.TestCase):
    def test_automatic_checks_are_spaced_but_the_button_always_looks(self):
        repos = GitRepos()
        try:
            os.environ.update({k: v for k, v in ENV.items() if k.startswith("GIT_")})
            d = tempfile.mkdtemp()
            state = AppState(os.path.join(d, "brain.json"), online=False)
            state.updater = Updater(repos.app)
            self.assertFalse(state.update_status(check=True)["available"])
            repos.write(repos.dev, "app.txt", "two")
            repos.push("new")
            self.assertFalse(state.update_status(check=True)["available"])   # too soon: the cached answer
            self.assertTrue(state.update_status(check=True, force=True)["available"])  # the button looks anyway
        finally:
            repos.close()
