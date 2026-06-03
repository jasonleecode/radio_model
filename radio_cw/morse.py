"""Morse code table and text<->symbol conversions.

This module is the shared vocabulary for all three layers:
  - execution layer encodes text -> dot/dash patterns -> audio
  - perception layer decodes audio -> dot/dash patterns -> text
  - decision layer works purely on decoded text

Timing follows the PARIS standard (the word "PARIS" = 50 dit units):
  dit              = 1 unit
  dah              = 3 units
  intra-char gap   = 1 unit  (between elements of one character)
  inter-char gap   = 3 units (between characters)
  word gap         = 7 units (between words)

The dit duration in milliseconds for a given speed:
  dit_ms = 1200 / WPM
"""

from __future__ import annotations

# Canonical International Morse table. '.' = dit, '-' = dah.
CHAR_TO_MORSE: dict[str, str] = {
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",   "E": ".",
    "F": "..-.",  "G": "--.",   "H": "....",  "I": "..",    "J": ".---",
    "K": "-.-",   "L": ".-..",  "M": "--",    "N": "-.",    "O": "---",
    "P": ".--.",  "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",  "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "/": "-..-.",
    "=": "-...-",  # BT / break, also used as separator in exchanges
    "+": ".-.-.",  # AR / end of message
    "-": "-....-",
    "(": "-.--.",  ")": "-.--.-", ":": "---...", ";": "-.-.-.",
    "'": ".----.", '"': ".-..-.", "@": ".--.-.", "!": "-.-.--",
}

# Common CW prosigns (procedural signals) sent as a single run with no
# intra-character gaps. Represented in text with angle brackets, e.g. <AR>.
# These matter for the decision-layer state machine.
PROSIGN_TO_MORSE: dict[str, str] = {
    "<AR>": ".-.-.",    # end of message
    "<AS>": ".-...",    # wait / stand by
    "<BT>": "-...-",    # break / new paragraph
    "<SK>": "...-.-",   # end of contact (silent key)
    "<KN>": "-.--.",    # go ahead, named station only
    "<BK>": "-...-.-",  # break-in
    "<SOS>": "...---...",
}

MORSE_TO_CHAR: dict[str, str] = {v: k for k, v in CHAR_TO_MORSE.items()}
MORSE_TO_PROSIGN: dict[str, str] = {v: k for k, v in PROSIGN_TO_MORSE.items()}


class MorseError(ValueError):
    """Raised when text or a symbol pattern cannot be mapped."""


def text_to_morse(text: str, *, strict: bool = False) -> list[str]:
    """Convert text to a list of per-character morse patterns.

    Words are separated by a single space ' ' token in the returned list,
    so callers can apply the 7-unit word gap. Unknown characters are dropped
    unless ``strict`` is set.

    >>> text_to_morse("OK")
    ['---', '-.-']
    >>> text_to_morse("A B")
    ['.-', ' ', '-...']
    """
    out: list[str] = []
    words = text.upper().split(" ")
    for wi, word in enumerate(words):
        if wi > 0:
            out.append(" ")  # word gap marker
        i = 0
        while i < len(word):
            # Try to match a prosign like <AR> first.
            if word[i] == "<":
                end = word.find(">", i)
                if end != -1:
                    token = word[i : end + 1]
                    if token in PROSIGN_TO_MORSE:
                        out.append(PROSIGN_TO_MORSE[token])
                        i = end + 1
                        continue
            ch = word[i]
            if ch in CHAR_TO_MORSE:
                out.append(CHAR_TO_MORSE[ch])
            elif strict:
                raise MorseError(f"unmappable character: {ch!r}")
            i += 1
    return out


def morse_to_text(patterns: list[str]) -> str:
    """Convert a list of morse patterns (with ' ' word markers) back to text.

    Tries prosign match before single-character match.
    """
    chars: list[str] = []
    for p in patterns:
        if p == " ":
            chars.append(" ")
        elif p in MORSE_TO_PROSIGN:
            chars.append(MORSE_TO_PROSIGN[p])
        elif p in MORSE_TO_CHAR:
            chars.append(MORSE_TO_CHAR[p])
        else:
            chars.append("?")  # undecodable element
    return "".join(chars)


def dit_seconds(wpm: float) -> float:
    """Dit duration in seconds for a PARIS-standard speed."""
    if wpm <= 0:
        raise MorseError("wpm must be positive")
    return 1.2 / wpm
