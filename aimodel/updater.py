"""Keep this copy of Aimodel up to date with what was pushed to GitHub.

Two ways, chosen by how it was installed:
  * git      a clone (git clone ...): fetch, then fast-forward only. Never merges or overwrites your changes.
  * download a ZIP or Pydroid copy with no .git: download the latest ZIP and copy its files over this folder.
             Your brain, your own vectors and writer files are never in the ZIP, so they are never touched.
The app asks for a check, shows what changed, and a button applies it (then the server restarts itself).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

REPO = os.environ.get("AIMODEL_REPO", "iamsabroy-gif/Aimodel")
BRANCH = os.environ.get("AIMODEL_BRANCH", "main")
API_URL = os.environ.get("AIMODEL_API_URL", f"https://api.github.com/repos/{REPO}/commits/{BRANCH}")
ZIP_URL = os.environ.get("AIMODEL_ZIP_URL", f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}")
MAX_ZIP = 120_000_000
VERSION_FILE = ".aimodel_version"
UA = {"User-Agent": "Aimodel-updater"}


class UpdateError(Exception):
    """Something the person can act on; the message says what."""


def _git(root: str, *args: str, timeout: float = 40.0) -> str:
    try:
        done = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise UpdateError("git is not installed here.") from None
    except subprocess.TimeoutExpired:
        raise UpdateError("Talking to GitHub took too long. Check the connection and try again.") from None
    if done.returncode != 0:
        raise UpdateError((done.stderr or done.stdout).strip().splitlines()[-1] if (done.stderr or done.stdout).strip()
                          else f"git {args[0]} failed")
    return done.stdout.strip()


class Updater:
    def __init__(self, root: str, api_url: str | None = None, zip_url: str | None = None) -> None:
        self.root = os.path.abspath(root)
        self.api_url, self.zip_url = api_url or API_URL, zip_url or ZIP_URL
        self.mode = self._detect()
        self.last = {"checked": None, "error": None, "behind": 0, "latest": None, "changes": []}

    # -- what kind of install is this
    def _detect(self) -> str:
        if os.path.exists(os.path.join(self.root, ".git")):
            try:
                _git(self.root, "rev-parse", "--is-inside-work-tree")
                return "git"
            except UpdateError:
                pass
        return "download"

    def _branch(self) -> tuple[str, str]:
        """(local branch, the remote branch it follows)."""
        branch = _git(self.root, "symbolic-ref", "--short", "HEAD")
        try:
            upstream = _git(self.root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        except UpdateError:
            upstream = f"origin/{branch}"
        return branch, upstream

    # -- version of what is running
    def current(self) -> dict:
        if self.mode == "git":
            try:
                full = _git(self.root, "rev-parse", "HEAD")
                when = _git(self.root, "log", "-1", "--format=%cs")
                subject = _git(self.root, "log", "-1", "--format=%s")
                return {"commit": full[:7], "date": when, "subject": subject, "mode": "git"}
            except UpdateError:
                pass
        path = os.path.join(self.root, VERSION_FILE)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {"commit": str(data.get("commit", ""))[:7] or "unknown", "date": data.get("date", ""),
                    "subject": "", "mode": "download"}
        except (OSError, ValueError):
            return {"commit": "unknown", "date": "", "subject": "", "mode": "download"}

    # -- is there something new?
    def check(self) -> dict:
        """Look at GitHub. Returns the same dict as status() with fresh information."""
        self.last["error"] = None
        try:
            if self.mode == "git":
                self._check_git()
            else:
                self._check_download()
        except UpdateError as e:
            self.last["error"] = str(e)
        self.last["checked"] = time.time()
        return self.status()

    def _check_git(self) -> None:
        branch, upstream = self._branch()
        remote, _, remote_branch = upstream.partition("/")
        _git(self.root, "fetch", "--quiet", remote, remote_branch or branch)
        behind = int(_git(self.root, "rev-list", "--count", f"HEAD..{upstream}") or 0)
        ahead = int(_git(self.root, "rev-list", "--count", f"{upstream}..HEAD") or 0)
        changes = _git(self.root, "log", "-n", "8", "--format=%s", f"HEAD..{upstream}").splitlines() if behind else []
        dirty = bool(_git(self.root, "status", "--porcelain", "--untracked-files=no"))
        self.last.update(behind=behind, ahead=ahead, dirty=dirty, changes=changes,
                         latest=_git(self.root, "rev-parse", "--short=7", upstream))

    def _check_download(self) -> None:
        try:
            req = urllib.request.Request(self.api_url, headers={**UA, "Accept": "application/vnd.github.sha"})
            with urllib.request.urlopen(req, timeout=20) as r:
                latest = r.read(200).decode().strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise UpdateError(f"Couldn't reach GitHub ({e}).") from None
        if not latest or len(latest) < 7 or not all(c in "0123456789abcdef" for c in latest.lower()[:40]):
            raise UpdateError("GitHub gave an answer I couldn't read.")
        mine = self.current()["commit"]
        newer = mine == "unknown" or not latest.startswith(mine)
        self.last.update(behind=1 if newer else 0, ahead=0, dirty=False, latest=latest[:7],
                         changes=["There is a newer version (I can't list what changed in a downloaded copy)."]
                         if newer else [])

    def status(self) -> dict:
        info = self.current()
        behind = self.last.get("behind", 0)
        problem = self.last.get("error")
        can = behind > 0 and not problem
        why = ""
        if self.mode == "git" and behind > 0:
            if self.last.get("dirty"):
                can, why = False, "You have changed files in this copy, so I won't overwrite them."
            elif self.last.get("ahead"):
                can, why = False, "This copy has changes that are not on GitHub, so it can't be fast-forwarded."
        return {"mode": self.mode, "current": info, "latest": self.last.get("latest"), "behind": behind,
                "changes": self.last.get("changes", []), "available": behind > 0, "can_update": can,
                "why_not": why, "error": problem, "checked": self.last.get("checked")}

    # -- do it
    def apply(self) -> dict:
        """Bring this copy up to date. Raises UpdateError with advice if it can't."""
        self.check()
        st = self.status()
        if st["error"]:
            raise UpdateError(st["error"])
        if not st["available"]:
            return {"updated": False, "message": "Already up to date.", "current": st["current"]}
        if not st["can_update"]:
            raise UpdateError(st["why_not"] or "I can't update this copy.")
        if self.mode == "git":
            _, upstream = self._branch()
            _git(self.root, "merge", "--ff-only", "--quiet", upstream)
        else:
            self._apply_download()
        self.last.update(behind=0, changes=[])
        return {"updated": True, "message": "Updated.", "current": self.current()}

    def _apply_download(self) -> None:
        try:
            req = urllib.request.Request(self.zip_url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read(MAX_ZIP + 1)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise UpdateError(f"Couldn't download the update ({e}).") from None
        if len(data) > MAX_ZIP:
            raise UpdateError("The update is bigger than I expected, so I stopped.")
        try:
            archive = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise UpdateError("The download wasn't a valid ZIP file.") from None
        names = [n for n in archive.namelist() if n and not n.endswith("/")]
        top = names[0].split("/")[0] + "/" if names else ""
        staged = tempfile.mkdtemp(prefix="aimodel-update-")
        try:
            for name in names:  # unpack to a scratch folder first: nothing changes unless it all unpacks
                rel = name[len(top):] if name.startswith(top) else name
                target = os.path.realpath(os.path.join(staged, rel))
                if not rel or not target.startswith(os.path.realpath(staged) + os.sep):
                    continue  # a path that would escape the folder: skip it
                if rel.split("/")[0] == ".git":
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with archive.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            for folder, _, files in os.walk(staged):
                for file in files:
                    src = os.path.join(folder, file)
                    dest = os.path.join(self.root, os.path.relpath(src, staged))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    tmp = dest + ".update"
                    shutil.copyfile(src, tmp)
                    os.replace(tmp, dest)  # one file at a time, each replaced in one step
        finally:
            shutil.rmtree(staged, ignore_errors=True)
        with open(os.path.join(self.root, VERSION_FILE), "w", encoding="utf-8") as f:
            json.dump({"commit": self.last.get("latest") or "", "date": time.strftime("%Y-%m-%d")}, f)
