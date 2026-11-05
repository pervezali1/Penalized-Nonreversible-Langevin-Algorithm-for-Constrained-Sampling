"""Voice the adversarial reel's script with Kokoro (see ../voicekit.py).    python narration.py <workdir>"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import voicekit  # noqa: E402
from scene import SCRIPT, TIMING  # noqa: E402

if __name__ == "__main__":
    voicekit.synth(SCRIPT, os.path.abspath(sys.argv[1]), TIMING)
