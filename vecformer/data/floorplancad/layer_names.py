"""
CAD layer names -> hashed word tokens, for the layer-name embedding.

US sheets carry AIA-style layer names ("A-Door", "A-WALL-ABOVE", "P-FIXT",
"xref-plan|A-Millwork (Dashed)"); the words say what the geometry on the layer is.
Each name becomes up to MAX_TOKENS lower-cased words (XREF binding prefix removed,
split on anything not a letter or digit), hashed into a fixed vocabulary so unseen
firms' names need no vocabulary file. Token 0 is padding; a layer without a name
is all padding.
"""
import re
import zlib

import torch

VOCAB = 4096
MAX_TOKENS = 4
_XREF = re.compile(r"^.*[|$]0?\$?")
_WORD = re.compile(r"[a-z0-9]+")


def name_tokens(name, vocab=VOCAB, max_tokens=MAX_TOKENS):
    if not name:
        return [0] * max_tokens
    words = _WORD.findall(_XREF.sub("", str(name)).lower())
    ids = [zlib.crc32(w.encode()) % (vocab - 1) + 1 for w in words[:max_tokens]]
    return ids + [0] * (max_tokens - len(ids))


def layer_token_table(layer_names, num_layers, vocab=VOCAB, max_tokens=MAX_TOKENS):
    """(num_layers, max_tokens) long; row i = tokens of layer id i (padding where unnamed)."""
    rows = [name_tokens(layer_names[i] if i < len(layer_names) else None, vocab, max_tokens)
            for i in range(num_layers)]
    return torch.tensor(rows, dtype=torch.long).reshape(num_layers, max_tokens)
