import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from aimodel.datasets import import_dataset, parse
from aimodel.model import LearningModel
from aimodel.server import serve


class TestDatasets(unittest.TestCase):
    def test_csv_question_answer(self):
        data = parse("qa.csv", "Question,Answer\nWhat colour is grass?,Green\nWho wrote Hamlet?,Shakespeare\n")
        self.assertEqual(data["pairs"], [("What colour is grass?", "Green"), ("Who wrote Hamlet?", "Shakespeare")])

    def test_jsonl_and_json(self):
        rows = [{"instruction": "Say hi", "output": "Hi!"}, {"text": "Cats are mammals."}]
        a = parse("d.jsonl", "\n".join(json.dumps(r) for r in rows))
        b = parse("d.json", json.dumps(rows))
        for d in (a, b):
            self.assertEqual(d["pairs"], [("Say hi", "Hi!")])
            self.assertIn("Cats are mammals", d["text"])

    def test_plain_text_and_unknown_columns(self):
        self.assertEqual(parse("n.txt", "Cats purr.")["pairs"], [])
        self.assertEqual(parse("x.csv", "a,b\n1,2\n")["text"], "a,b\n1,2\n")

    def test_import_teaches_the_model(self):
        m = LearningModel()
        seen = []
        out = import_dataset(m, "qa.csv", "question,answer\nWhat colour is grass?,Green\n",
                             lambda *a: seen.append(a))
        self.assertEqual(out["answers"], 1)
        reply, _ = m.respond("What colour is grass?")
        self.assertIn("Green", reply)
        self.assertTrue(seen)


class TestServer(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.httpd = serve(os.path.join(self.dir.name, "brain.json"), "127.0.0.1", 0, False, "secret")
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, path, body=None, token="secret"):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {token}"} if token else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait_job(self):
        for _ in range(300):
            _, job = self.call("/api/job")
            if not job["running"]:
                return job
            time.sleep(0.1)
        self.fail("job never finished")

    def test_app_files_and_auth(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/") as r:
            self.assertIn(b"Aimodel", r.read())
        for p in ("/manifest.webmanifest", "/sw.js", "/icon.svg"):
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{p}") as r:
                self.assertEqual(r.status, 200)
        self.assertEqual(self.call("/api/state", token=None)[0], 401)
        self.assertEqual(self.call("/api/state", token="wrong")[0], 401)
        self.assertEqual(self.call("/api/state")[0], 200)

    def test_no_path_traversal(self):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/../server.py")
            self.fail("served a file outside the app folder")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)

    def test_train_with_dataset_then_chat(self):
        code, r = self.call("/api/dataset", {"name": "qa.csv",
                            "text": "question,answer\nWhat is the capital of Zubrowka?,Lutz\n"})
        self.assertEqual(code, 200)
        job = self.wait_job()
        self.assertIsNone(job["error"])
        self.assertEqual(job["result"]["answers"], 1)
        _, chat = self.call("/api/chat", {"text": "What is the capital of Zubrowka?"})
        self.assertIn("Lutz", chat["reply"])
        self.assertEqual(self.call("/api/rate", {"good": True})[1]["ok"], True)
        self.call("/api/train", {"epochs": 1})
        self.assertIsNone(self.wait_job()["error"])
        self.assertTrue(os.path.exists(os.path.join(self.dir.name, "brain.json")))  # it saved

    def test_teach_facts_quiz_and_errors(self):
        self.assertEqual(self.call("/api/teach", {"prompt": "ping", "response": "pong"})[0], 200)
        self.assertEqual(self.call("/api/teach", {"prompt": "ping"})[0], 400)
        self.assertEqual(self.call("/api/chat", {"text": ""})[0], 400)
        self.assertEqual(self.call("/api/dataset", {})[0], 400)
        self.call("/api/chat", {"text": "My sister lives in Delhi"})
        _, facts = self.call("/api/facts?topic=sister")
        self.assertTrue(any("Delhi" in f for f in facts["facts"]))
        self.assertEqual(self.call("/api/quiz", {"n": 2})[0], 200)
        self.assertEqual(self.call("/api/memories")[1]["memories"][0]["prompt"], "ping")
        self.assertEqual(self.call("/api/forget", {"text": "ping"})[1]["forgot"], 1)


if __name__ == "__main__":
    unittest.main()
