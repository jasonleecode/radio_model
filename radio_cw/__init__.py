"""radio_cw: neural CW (Morse) transceiver experiment.

Three loosely-coupled layers communicate through plain text + metadata:
  perception (audio -> CW text)  ->  decision (state machine)  ->  execution (text -> audio)
"""

from .morse import morse_to_text, text_to_morse
from .synth import SynthConfig, synthesize

__all__ = ["text_to_morse", "morse_to_text", "synthesize", "SynthConfig"]
