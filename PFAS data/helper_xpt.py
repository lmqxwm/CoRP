"""Minimal reader of SAS XPORT v5 files (one member, as distributed by NHANES): returns {name: array, '_labels': {...}}."""
import struct
import numpy as np


def _ibm(b):
    """IBM 370 floats as big-endian uint64 (short fields right-padded with zero bytes) -> float64; SAS missing values -> nan."""
    sign, exp, frac = b >> 63, (b >> 56) & 0x7f, b & 0x00ffffffffffffff
    v = (frac / 2.0 ** 56) * 16.0 ** (exp.astype(float) - 64)
    v = np.where(sign == 1, -v, v)
    return np.where((frac == 0) & (b >> 56 != 0), np.nan, v)


def read_xpt(path):
    raw = open(path, 'rb').read()
    i = raw.index(b'HEADER RECORD*******NAMESTR HEADER RECORD')
    nvar = int(raw[i + 54:i + 58])
    ns = raw[i + 80:i + 80 + 140 * nvar]
    cols = []
    for k in range(nvar):
        r = ns[140 * k:140 * (k + 1)]
        typ, _, lng = struct.unpack('>hhh', r[:6])
        cols.append((r[8:16].decode().strip(), r[16:56].decode('latin1').strip(), typ, lng, struct.unpack('>i', r[84:88])[0]))
    j = raw.index(b'HEADER RECORD*******OBS     HEADER RECORD', i) + 80
    width = sum(c[3] for c in cols)
    body = raw[j:]
    nobs = len(body) // width
    while nobs and body[(nobs - 1) * width:nobs * width].strip(b' ') == b'':
        nobs -= 1
    rows = np.frombuffer(body[:nobs * width], 'u1').reshape(nobs, width)
    out, lab = {}, {}
    for name, label, typ, lng, pos in cols:
        cell = rows[:, pos:pos + lng]
        if typ == 1:
            full = np.zeros((nobs, 8), 'u1'); full[:, :lng] = cell
            out[name] = _ibm(full.view('>u8').ravel())
        else:
            out[name] = np.array([bytes(c).decode('latin1').strip() for c in cell])
        lab[name] = label
    out['_labels'] = lab
    return out
