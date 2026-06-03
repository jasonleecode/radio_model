"""Realistic CW text generator for training the perception model.

Random *uniform* letters would teach the model nothing about CW's strong
priors -- in real copy "599", "DE", "73", "CQ", "TU" dominate. This sampler
produces QSO-flavoured text (calls, RST exchanges, sign-offs, ragchew with
ham abbreviations) so the model learns those priors, plus some near-random
groups so it doesn't overfit to templates.

All output stays within the model vocabulary (A-Z, 0-9, . , ? / = , prosigns).
"""

from __future__ import annotations

import numpy as np

# Common CW abbreviations / Q-codes / prosigns used in ragchew.
ABBREV = [
    "CQ", "DE", "TU", "TNX", "FB", "OM", "YL", "ES", "HR", "HW", "UR", "RST",
    "QTH", "QRZ", "QRM", "QRN", "QSB", "QSL", "QSO", "QRP", "RIG", "ANT", "WX",
    "PWR", "GM", "GA", "GE", "GN", "RR", "CPY", "GL", "GUD", "AGN", "PSE", "NW",
    "NAME", "OP", "WID", "VY", "73", "88", "K", "KN",
]
NAMES = ["JOHN", "BOB", "JIM", "TOM", "BILL", "DAVE", "MIKE", "STEVE", "PAUL",
         "FRANK", "LI", "WANG", "KEN", "RON", "ED", "AL", "MAX", "SAM"]
QTHS = ["BOSTON", "NYC", "LONDON", "BERLIN", "TOKYO", "BEIJING", "PARIS",
        "ROME", "OSAKA", "SEOUL", "MOSCOW", "MADRID", "DENVER", "MIAMI"]
RIGS = ["FT857", "K3", "IC7300", "TS590", "FT991", "KX3", "QRP"]

# Callsign prefix shapes that match the decision-layer CALLSIGN_RE.
_PREFIX_LETTERS = "ABCDEFGHJKLMNPRSTVWXYZ"


def _rand_callsign(rng: np.random.Generator) -> str:
    shape = rng.choice(["LD", "LLD", "DLD"], p=[0.45, 0.4, 0.15])
    pre = ""
    for c in shape:
        if c == "L":
            pre += rng.choice(list(_PREFIX_LETTERS))
        else:  # D
            pre += str(rng.integers(0, 10))
    suffix_len = int(rng.integers(1, 4))
    suffix = "".join(rng.choice(list(_PREFIX_LETTERS)) for _ in range(suffix_len))
    call = pre + suffix
    if rng.random() < 0.05:  # occasional portable
        call += "/P"
    return call


def _rand_rst(rng: np.random.Generator) -> str:
    r = rng.integers(3, 6)
    s = rng.integers(5, 10)
    return f"{r}{s}9"  # tone almost always 9 for CW


def cq_call(rng: np.random.Generator) -> str:
    me = _rand_callsign(rng)
    n = rng.choice([1, 2, 3], p=[0.2, 0.4, 0.4])
    return "CQ " * n + f"DE {me} {me} K"


def answer(rng: np.random.Generator) -> str:
    me, them = _rand_callsign(rng), _rand_callsign(rng)
    return f"{them} DE {me} {me} K"


def exchange(rng: np.random.Generator) -> str:
    me, them = _rand_callsign(rng), _rand_callsign(rng)
    rst = _rand_rst(rng)
    name = rng.choice(NAMES)
    qth = rng.choice(QTHS)
    parts = [f"{them} DE {me}", "="]
    if rng.random() < 0.6:
        parts += [rng.choice(["GE", "GM", "GA"]), "ES", "TNX", "CALL", "="]
    parts += ["UR", "RST", rst, rst, "=", "NAME", name, name, "=",
              "QTH", qth, "="]
    if rng.random() < 0.4:
        parts += ["RIG", rng.choice(RIGS), "PWR", f"{rng.integers(5, 100)}W", "="]
    parts += ["HW?", f"{them} DE {me} K"]
    return " ".join(parts)


def closing(rng: np.random.Generator) -> str:
    me, them = _rand_callsign(rng), _rand_callsign(rng)
    end = rng.choice(["<SK>", "K", "<AR>"])
    return (f"{them} DE {me} = RR FB = TNX FER QSO = "
            f"73 ES GL = {them} DE {me} {end}")


def ragchew(rng: np.random.Generator) -> str:
    """A loose run of abbreviations / numbers, less templated."""
    n = int(rng.integers(3, 10))
    toks = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.6:
            toks.append(rng.choice(ABBREV))
        elif roll < 0.8:
            toks.append(_rand_rst(rng))
        elif roll < 0.9:
            toks.append(rng.choice(NAMES))
        else:
            toks.append(str(rng.integers(0, 1000)))
    return " ".join(toks)


# Sampling weights across categories.
_GENERATORS = [
    (cq_call, 0.20),
    (answer, 0.15),
    (exchange, 0.25),
    (closing, 0.15),
    (ragchew, 0.20),
    (lambda r: _rand_callsign(r), 0.05),
]


class CorpusSampler:
    """Draw realistic CW text strings."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self._gens = [g for g, _ in _GENERATORS]
        self._weights = np.array([w for _, w in _GENERATORS])
        self._weights /= self._weights.sum()

    def sample(self) -> str:
        gen = self.rng.choice(self._gens, p=self._weights)
        return gen(self.rng)

    def samples(self, n: int) -> list[str]:
        return [self.sample() for _ in range(n)]
