"""Tests for the decision-layer state machine and the end-to-end loop."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radio_cw.channel import add_white_noise  # noqa: E402
from radio_cw.decision import (  # noqa: E402
    QSOMachine,
    QSOState,
    StationInfo,
    parse_incoming,
    snr_to_rst,
)
from radio_cw.decode_dsp import decode_audio  # noqa: E402
from radio_cw.synth import SynthConfig, synthesize  # noqa: E402


def test_snr_to_rst_monotonic():
    assert snr_to_rst(20) == "599"
    assert snr_to_rst(-10) == "339"
    # readability/strength should not increase as SNR drops
    reports = [snr_to_rst(s) for s in (20, 10, 4, 0, -5)]
    assert reports == sorted(reports, reverse=True)


def test_parse_callsign_after_de():
    rx = parse_incoming("CQ CQ DE W1AW W1AW K", my_call="BG1ABC")
    assert rx.has_cq
    assert rx.peer_call == "W1AW"
    assert not rx.addressed_to_us


def test_parse_addressed_to_us_and_bt():
    rx = parse_incoming("BG1ABC DE W1AW <BT> UR RST 599 <BT> HW?", my_call="BG1ABC")
    assert rx.addressed_to_us
    assert rx.peer_call == "W1AW"
    assert rx.has_rst
    assert rx.asks_report


def test_answer_cq_transitions():
    m = QSOMachine(StationInfo("BG1ABC"))
    reply = m.process("CQ CQ DE W1AW W1AW K", snr_db=15)
    assert m.state == QSOState.ANSWERED_CQ
    assert m.peer == "W1AW"
    assert "W1AW" in reply and "BG1ABC" in reply


def test_rst_reflects_snr():
    m = QSOMachine(StationInfo("BG1ABC", name="LI", qth="BJ"))
    m.process("CQ DE W1AW K", snr_db=15)
    exch = m.process("BG1ABC DE W1AW K", snr_db=3)  # weak signal
    assert "559" in exch  # snr=3 -> 559
    assert m.state == QSOState.SENT_EXCHANGE


def test_caller_role():
    m = QSOMachine(StationInfo("BG1ABC"))
    cq = m.call_cq()
    assert "CQ" in cq and m.state == QSOState.CALLING
    reply = m.process("BG1ABC DE W1AW W1AW K", snr_db=12)
    assert m.state == QSOState.SENT_EXCHANGE
    assert "UR RST" in reply


def _run_qso(snr_db=None, seed=0):
    rng = np.random.default_rng(seed)
    a = QSOMachine(StationInfo("BG1ABC", name="LI", qth="BJ"))
    b = QSOMachine(StationInfo("W1AW", name="BOB", qth="NYC"))
    freqs = {"BG1ABC": 600.0, "W1AW": 720.0}
    sender, listener = a, b
    msg = a.call_cq()
    for _ in range(14):
        cfg = SynthConfig(wpm=20, sidetone_hz=freqs[sender.me.callsign])
        audio, _ = synthesize(msg, cfg)
        if snr_db is not None:
            audio = add_white_noise(audio, snr_db, rng)
        decoded, meta = decode_audio(audio, 8000)
        reply = listener.process(decoded, meta.snr_db)
        if reply is None:
            break
        msg, sender, listener = reply, listener, sender
    return a, b


def test_full_loop_clean():
    a, b = _run_qso(snr_db=None)
    assert a.state == QSOState.DONE
    assert b.state == QSOState.DONE


def test_full_loop_noisy():
    a, b = _run_qso(snr_db=5)
    # Both should at least reach the closing handshake under moderate noise.
    assert a.state in (QSOState.CLOSING, QSOState.DONE)
    assert b.state in (QSOState.CLOSING, QSOState.DONE)
