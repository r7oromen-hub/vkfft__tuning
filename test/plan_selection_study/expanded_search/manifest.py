"""Durable, replayable workload candidate manifest."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import enumerate_candidates
import selection


def encode(row):
    return (json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf8')


def make(path, n, batch, identity_request, seed=20261030, limit=64):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rows, selected = selection.select(enumerate_candidates.candidates(n,batch), identity_request, seed, limit)
    protocol = dict(record_type='manifest_protocol', n=n, batch=batch, identity_request=identity_request,
                    selection_seed=seed, limit=limit)
    digest = hashlib.sha256(); counts = Counter()
    with path.open('xb') as out:
        out.write(encode(protocol))
        for row in rows:
            line = encode(dict(record_type='candidate_manifest', **row))
            out.write(line); digest.update(line); counts[row['selection']] += 1
        complete = dict(record_type='manifest_complete', total=len(rows), selected=len(selected),
                        candidate_lines_sha256=digest.hexdigest(), counts=dict(counts))
        out.write(encode(complete)); out.flush(); os.fsync(out.fileno())
    return selected, complete
