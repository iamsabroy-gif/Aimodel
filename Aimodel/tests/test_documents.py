import base64
import importlib.util
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from unittest import mock

from aimodel import documents
from aimodel.documents import DocumentError, markdown_to_text, pdf_to_text, unwrap
from aimodel.model import LearningModel
from aimodel.server import serve

HAVE_PYPDF = importlib.util.find_spec("pypdf") is not None


def make_pdf(pages: list[list[str]]) -> bytes:
    """A small valid PDF: each page is a list of text lines."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        content = "BT /F1 12 Tf 14 TL 50 750 Td " + " T* ".join(
            "(" + ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj" for ln in lines) + " ET"
        objs.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
        c = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {c} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


MD = """---
title: Penguins
---
# Penguins

Penguins are **flightless** birds that live in the [Southern
Hemisphere](https://example.com/sh). See ![a picture](p.png).

## Diet
- They eat krill and fish.
- [x] They `swim` fast.
- Hi

| Species | Height | Weight |
|---|---|---|
| Emperor | 1.1 m | 35 kg |

```python
print("not knowledge")
```

> Emperor penguins are the largest species.[^1]
"""


class TestMarkdown(unittest.TestCase):
    def test_markup_goes_and_the_words_stay(self):
        text = markdown_to_text(MD)
        self.assertIn("Penguins are flightless birds that live in the Southern Hemisphere.", text)
        self.assertIn("They eat krill and fish.", text.splitlines())
        self.assertIn("They swim fast.", text.splitlines())
        self.assertIn("Emperor: Height 1.1 m, Weight 35 kg.", text)
        self.assertIn("Emperor penguins are the largest species.", text)
        for junk in ("**", "](", "```", "print(", "title:", "[^1]", "http", "|", "# "):
            self.assertNotIn(junk, text)
        self.assertNotIn("Hi.", text)  # a one-word bullet is not a sentence

    def test_it_can_answer_from_markdown(self):
        m = LearningModel(seed=0)
        m.add_document(markdown_to_text(MD), "notes.md")
        self.assertIn("flightless", m.respond("What are penguins?", learn=False)[0])
        self.assertIn("krill", m.respond("What do penguins eat?", learn=False)[0])
        self.assertIn("1.1 m", m.respond("How tall is the emperor?", learn=False)[0])

    def test_reading_a_markdown_file_by_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "notes.md")
            with open(path, "w") as f:
                f.write(MD)
            m = LearningModel(seed=0)
            self.assertGreater(m.read(path), 2)
            self.assertTrue(all("**" not in k["text"] for k in m.knowledge))

    def test_unwrap_joins_lines_and_hyphenation(self):
        self.assertEqual(unwrap("A long exam-\nple that wraps\nacross lines.\n\nNext."),
                         "A long example that wraps across lines.\nNext.")


@unittest.skipUnless(HAVE_PYPDF, "pypdf is not installed")
class TestPdf(unittest.TestCase):
    def test_text_comes_out_unwrapped_without_page_furniture(self):
        pages = [["Annual Report", "Penguins are flightless birds that live in the", "Southern Hemisphere.", "1"],
                 ["Annual Report", "They eat krill and fish.", "2"],
                 ["Annual Report", "Emperors are the largest species.", "3"]]
        text = pdf_to_text(make_pdf(pages))
        self.assertIn("Penguins are flightless birds that live in the Southern Hemisphere.", text)
        self.assertIn("They eat krill and fish.", text)
        self.assertNotIn("Annual Report", text)  # a header on every page
        self.assertNotIn("\n1\n", "\n" + text + "\n")  # page numbers

    def test_it_can_answer_from_a_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.pdf")
            with open(path, "wb") as f:
                f.write(make_pdf([["Penguins are flightless birds that live in the", "Southern Hemisphere."],
                                  ["They eat krill and fish."]]))
            m = LearningModel(seed=0)
            self.assertGreater(m.read(path), 1)
            self.assertIn("krill", m.respond("What do penguins eat?", learn=False)[0])

    def test_bad_files_give_advice_not_a_crash(self):
        with self.assertRaises(DocumentError) as e:
            pdf_to_text(b"this is not a pdf")
        self.assertIn("couldn't open that PDF", str(e.exception))
        with self.assertRaises(DocumentError) as e:
            pdf_to_text(make_pdf([[""]]))
        self.assertIn("no text", str(e.exception))


class TestWithoutPypdf(unittest.TestCase):
    def test_a_missing_pypdf_says_how_to_install_it(self):
        with mock.patch.dict("sys.modules", {"pypdf": None}):
            with self.assertRaises(DocumentError) as e:
                documents.pdf_to_text(b"%PDF-1.4")
        self.assertIn("pip install pypdf", str(e.exception))


class TestUpload(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.httpd = serve(os.path.join(self.dir.name, "brain.json"), "127.0.0.1", 0, False, None)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def post(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.httpd.server_address[1]}/api/dataset",
                                     data=json.dumps(body).encode())
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def job(self):
        import time
        for _ in range(100):
            with urllib.request.urlopen(f"http://127.0.0.1:{self.httpd.server_address[1]}/api/job") as r:
                job = json.loads(r.read())
            if not job["running"]:
                return job
            time.sleep(0.1)

    def test_markdown_upload(self):
        self.post({"name": "notes.md", "text": MD})
        job = self.job()
        self.assertEqual((job["error"], job["result"]["format"]), (None, "markdown"))
        self.assertGreater(job["result"]["sentences"], 2)

    @unittest.skipUnless(HAVE_PYPDF, "pypdf is not installed")
    def test_pdf_upload_as_base64(self):
        pdf = make_pdf([["Penguins eat krill and fish every day."]])
        self.post({"name": "r.pdf", "base64": base64.b64encode(pdf).decode()})
        job = self.job()
        self.assertEqual((job["error"], job["result"]["format"]), (None, "PDF"))

    def test_a_broken_pdf_reports_the_problem(self):
        self.post({"name": "r.pdf", "base64": base64.b64encode(b"nope").decode()})
        job = self.job()
        self.assertIsNotNone(job["error"])
        self.assertNotIn("Couldn't reach", job["error"])


if __name__ == "__main__":
    unittest.main()
