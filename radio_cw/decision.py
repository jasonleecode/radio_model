"""Decision layer: decoded CW text + metadata -> reply text.

A finite state machine over a CW QSO (contact). Deliberately *not* a language
model: callsigns and RST reports must be exact, so we parse a few structured
tokens out of the incoming copy and fill templates. The SNR from the metadata
sidecar drives the signal report (RST) we send back.

Two entry roles that converge after the handshake:
  - responder: we hear a CQ and answer it
  - caller:    we call CQ (``call_cq``) and someone answers

QSO flow (states):
  IDLE ──hear CQ──▶ ANSWERED_CQ ──they come back──▶ SENT_EXCHANGE
  IDLE ──call_cq──▶ CALLING ─────answer──────────▶ SENT_EXCHANGE
  SENT_EXCHANGE ──their exchange──▶ CLOSING ──their 73──▶ DONE

The parser is intentionally lenient: real perception output has errors, so we
match on presence of key tokens rather than rigid grammar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

# Amateur callsign: 1-3 char prefix, a digit, 1-4 letter suffix. Pragmatic, not
# exhaustive (no portable "/P" handling yet).
CALLSIGN_RE = re.compile(r"\b[A-Z0-9]{1,3}[0-9][A-Z]{1,4}\b")


class QSOState(Enum):
    IDLE = auto()
    CALLING = auto()       # we called CQ, waiting for an answer
    ANSWERED_CQ = auto()   # we answered someone's CQ, waiting for their come-back
    SENT_EXCHANGE = auto()  # we sent our report, waiting for theirs
    CLOSING = auto()       # we sent 73, waiting for their sign-off
    DONE = auto()


@dataclass
class StationInfo:
    """Our own station identity, used to fill reply templates."""

    callsign: str
    name: str = "OP"
    qth: str = "NIL"


@dataclass
class Incoming:
    """Structured view of one received transmission."""

    raw: str
    has_cq: bool
    peer_call: str | None      # sender's callsign (token after DE)
    addressed_to_us: bool      # our callsign appears in the copy
    has_rst: bool
    asks_report: bool          # contains HW / HW?
    has_73: bool
    has_sk: bool


def snr_to_rst(snr_db: float) -> str:
    """Map an SNR estimate to a conventional CW RST report (tone always 9)."""
    if snr_db >= 12:
        return "599"
    if snr_db >= 6:
        return "579"
    if snr_db >= 2:
        return "559"
    if snr_db >= -2:
        return "449"
    return "339"


def parse_incoming(text: str, my_call: str) -> Incoming:
    up = text.upper()
    # '=' and the <BT> prosign share one code (-...-); the decoder emits <BT>,
    # the synth uses '='. Treat both as separators.
    tokens = up.replace("=", " ").replace("<BT>", " ").split()
    my = my_call.upper()

    peer = None
    if "DE" in tokens:
        i = tokens.index("DE")
        for tok in tokens[i + 1 :]:
            if CALLSIGN_RE.fullmatch(tok) and tok != my:
                peer = tok
                break
    if peer is None:  # fall back to any callsign that isn't ours
        for tok in tokens:
            if CALLSIGN_RE.fullmatch(tok) and tok != my:
                peer = tok
                break

    # RST: 3 digits, or with cut numbers N(9)/T(0), e.g. 5NN.
    has_rst = bool(re.search(r"\b[0-9TN]{3}\b", up)) and any(
        c.isdigit() for c in up
    )

    return Incoming(
        raw=text,
        has_cq="CQ" in tokens,
        peer_call=peer,
        addressed_to_us=my in tokens,
        has_rst=has_rst,
        asks_report="HW" in tokens or "HW?" in up,
        has_73="73" in tokens or "88" in tokens,
        has_sk="<SK>" in up or "SK" in tokens,
    )


class QSOMachine:
    """Stateful CW QSO responder/caller."""

    def __init__(self, station: StationInfo):
        self.me = station
        self.state = QSOState.IDLE
        self.peer: str | None = None
        self.last_rx_rst: bool = False

    # --- driving the machine -------------------------------------------------

    def call_cq(self) -> str:
        """Initiate a contact by calling CQ. Returns the text to transmit."""
        self.state = QSOState.CALLING
        c = self.me.callsign
        return f"CQ CQ CQ DE {c} {c} K"

    def process(self, text: str, snr_db: float = 12.0) -> str | None:
        """Consume a received transmission, return reply text or None.

        ``snr_db`` (from the metadata sidecar) sets the RST we send.
        Returns None when there is nothing to say (e.g. QSO finished, or copy
        we can't act on in the current state).
        """
        rx = parse_incoming(text, self.me.callsign)
        handler = {
            QSOState.IDLE: self._on_idle,
            QSOState.CALLING: self._on_calling,
            QSOState.ANSWERED_CQ: self._on_answered_cq,
            QSOState.SENT_EXCHANGE: self._on_sent_exchange,
            QSOState.CLOSING: self._on_closing,
            QSOState.DONE: lambda rx, snr: None,
        }[self.state]
        return handler(rx, snr_db)

    # --- per-state handlers --------------------------------------------------

    def _on_idle(self, rx: Incoming, snr_db: float) -> str | None:
        # Someone is calling CQ, or calling us directly.
        if rx.peer_call and (rx.has_cq or rx.addressed_to_us):
            self.peer = rx.peer_call
            self.state = QSOState.ANSWERED_CQ
            c = self.me.callsign
            return f"{self.peer} DE {c} {c} K"
        return None

    def _on_calling(self, rx: Incoming, snr_db: float) -> str | None:
        # We called CQ; someone answered with their callsign.
        if rx.peer_call and rx.addressed_to_us:
            self.peer = rx.peer_call
            return self._send_exchange(snr_db, greet=True)
        return None

    def _on_answered_cq(self, rx: Incoming, snr_db: float) -> str | None:
        # The station we answered comes back to us; send our exchange.
        if rx.addressed_to_us or rx.peer_call == self.peer:
            return self._send_exchange(snr_db, greet=True)
        return None

    def _on_sent_exchange(self, rx: Incoming, snr_db: float) -> str | None:
        # We're waiting for their report; on receipt, acknowledge and start 73.
        if rx.has_rst or rx.asks_report or rx.addressed_to_us:
            self.state = QSOState.CLOSING
            c, p = self.me.callsign, self.peer
            return (f"{p} DE {c} = RR FB = TNX FER NICE QSO = "
                    f"73 ES GL = {p} DE {c} <SK>")
        return None

    def _on_closing(self, rx: Incoming, snr_db: float) -> str | None:
        # Their sign-off; send a final 73 and finish.
        if rx.has_73 or rx.has_sk or rx.addressed_to_us:
            self.state = QSOState.DONE
            c, p = self.me.callsign, self.peer
            return f"{p} DE {c} = 73 <SK>"
        return None

    # --- templates -----------------------------------------------------------

    def _send_exchange(self, snr_db: float, *, greet: bool) -> str:
        self.state = QSOState.SENT_EXCHANGE
        rst = snr_to_rst(snr_db)
        c, p = self.me.callsign, self.peer
        g = "GE ES TNX CALL = " if greet else ""
        return (f"{p} DE {c} = {g}UR RST {rst} {rst} = "
                f"NAME {self.me.name} {self.me.name} = "
                f"QTH {self.me.qth} = HW? {p} DE {c} K")
