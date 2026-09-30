"""One-tap launcher (handy on phones, e.g. Pydroid 3): run this file to start the chat.

The brain is saved next to this file, so it's found again on the next run.
"""

import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, here)

from aimodel.cli import main  # noqa: E402

main(sys.argv[1:])
