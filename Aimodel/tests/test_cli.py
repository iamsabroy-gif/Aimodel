import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every chat command, on an empty brain and with a little knowledge. None may crash.
COMMANDS = [
    "/help", "/stats", "/progress", "/facts", "/facts cat", "/why", "/good", "/bad", "/gen", "/gen the",
    "/forget nothing", "/similar cat", "/train 1", "/quiz", "/quiz 2", "/sleep", "/curious", "/define",
    "/greet", "/greet forget nothing", "/greet kem cho => Majama {name}!", "kem cho", "/greet forget kem cho",
    "/export {dir}/export", "/writer", "/writer off", "/writer on", "/online off", "/online on",
    "/evaluate quick", "/evaluate history", "/teach hi there friend => Hello!", "/read /nonexistent.txt",
    "/vectors /nonexistent.txt", "/nonsense", "hello", "My name is Sam", "What is my name?", "/quit",
]


class TestEveryCommand(unittest.TestCase):
    def run_chat(self, lines, brain):
        return subprocess.run([sys.executable, "-m", "aimodel", "--brain", brain, "--offline"],
                              input="\n".join(lines) + "\n", text=True, capture_output=True, cwd=ROOT, timeout=600)

    def test_no_command_crashes(self):
        with tempfile.TemporaryDirectory() as d:
            lines = [c.replace("{dir}", d) for c in COMMANDS]
            result = self.run_chat(lines, os.path.join(d, "brain.json"))
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("Saved to", result.stdout)

    def test_again_on_the_saved_brain_with_some_knowledge(self):
        with tempfile.TemporaryDirectory() as d:
            notes = os.path.join(d, "notes.txt")
            with open(notes, "w") as f:
                f.write("Cats are mammals. Mammals are warm-blooded animals. Animals are living things. "
                        "My dog is called Rex.")
            brain = os.path.join(d, "brain.json")
            first = self.run_chat([f"/read {notes}", "Is a cat an animal?", "/good", "/quit"], brain)
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            lines = [c.replace("{dir}", d) for c in COMMANDS if c != "/quit"] + ["/quit"]
            second = self.run_chat(lines, brain)
            self.assertEqual(second.returncode, 0, second.stderr[-2000:])
            self.assertNotIn("Traceback", second.stderr)


if __name__ == "__main__":
    unittest.main()
