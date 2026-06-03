"""CTC vocabulary and tokenizer for the perception model.

The model emits a sequence of *text tokens*; CTC needs a blank symbol plus a
fixed token set. Two CW-specific subtleties drive the design:

1. **Morse-code collisions.** Several text symbols share one code and are thus
   acoustically identical -- the model literally cannot tell them apart:
       -...-  = '='  and prosign <BT>
       .-.-.  = '+'  and prosign <AR>
       -.--.  = '('  and prosign <KN>
   Colliding symbols must map to a single token or the targets are unlearnable.
   We pick one canonical token per code and ``canonicalize`` the aliases.

2. **Word space is a real token**, distinct from the CTC blank. CW word gaps
   (7 units) carry meaning, so ' ' is in the vocabulary; blank means "emit
   nothing this frame".

Token id 0 is reserved for the CTC blank.
"""

from __future__ import annotations

from .morse import CHAR_TO_MORSE, PROSIGN_TO_MORSE

BLANK = "<blank>"
SPACE = " "

# Canonical token set, chosen so no two entries share a Morse code.
# Letters + digits, a little punctuation, the BT separator as '=', word space,
# and the prosigns that don't collide with included punctuation.
_LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
_DIGITS = [str(d) for d in range(10)]
_PUNCT = [".", ",", "?", "/", "="]  # '=' is the BT separator
_PROSIGNS = ["<AR>", "<SK>", "<KN>", "<AS>", "<BK>"]

# Aliases: text forms that share a Morse code with a canonical token.
_ALIASES = {
    "<BT>": "=",   # -...-
    "+": "<AR>",   # .-.-.
    "(": "<KN>",   # -.--.
}


def _verify_no_collisions() -> None:
    """Guard: every canonical token must have a unique Morse code."""
    code_of = {**CHAR_TO_MORSE, **PROSIGN_TO_MORSE}
    seen: dict[str, str] = {}
    for tok in _PUNCT + _PROSIGNS:
        code = code_of.get(tok)
        if code is None:
            continue
        if code in seen:
            raise ValueError(
                f"vocab collision: {tok!r} and {seen[code]!r} share code {code!r}"
            )
        seen[code] = tok


class Vocabulary:
    """Maps between text and CTC token ids."""

    def __init__(self) -> None:
        _verify_no_collisions()
        self.tokens: list[str] = [BLANK, SPACE] + _LETTERS + _DIGITS + _PUNCT + _PROSIGNS
        self.tok2id: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.blank_id = 0

    def __len__(self) -> int:
        return len(self.tokens)

    @staticmethod
    def canonicalize(token: str) -> str:
        return _ALIASES.get(token, token)

    def tokenize(self, text: str) -> list[str]:
        """Split text into canonical tokens (prosigns kept whole)."""
        text = text.upper()
        out: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "<":
                end = text.find(">", i)
                if end != -1:
                    tok = self.canonicalize(text[i : end + 1])
                    if tok in self.tok2id:
                        out.append(tok)
                        i = end + 1
                        continue
            tok = self.canonicalize(ch)
            if tok in self.tok2id:
                out.append(tok)
            i += 1
        return out

    def encode(self, text: str) -> list[int]:
        """Text -> list of token ids (no blanks)."""
        return [self.tok2id[t] for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        """Token ids -> text (ids should already be CTC-collapsed)."""
        return "".join(self.tokens[i] for i in ids if i != self.blank_id)

    def ctc_collapse(self, ids: list[int]) -> list[int]:
        """Collapse a raw per-frame argmax path: merge repeats, drop blanks."""
        out: list[int] = []
        prev = None
        for i in ids:
            if i != prev and i != self.blank_id:
                out.append(i)
            prev = i
        return out

    def greedy_decode(self, ids: list[int]) -> str:
        return self.decode(self.ctc_collapse(ids))
