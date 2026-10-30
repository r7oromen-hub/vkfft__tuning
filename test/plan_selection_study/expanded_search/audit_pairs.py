"""Independent paired-log calculations for ten-round confirmation sessions.

The caller must independently verify guards, numeric validity, source/binary
identity, seeds, paired order, workload and session independence. These
functions do not grant a performance claim or correct selection bias.
"""
import math
import statistics

T_975_DF9 = 2.262157162798205


def ten_round_session(pairs):
    """pairs contains (candidate_ms, original_default_ms), one per block."""
    if len(pairs) != 10:
        raise ValueError('confirmation session requires exactly 10 paired rounds')
    logs = []
    for candidate, original in pairs:
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in (candidate, original)):
            raise ValueError('paired times must be finite and positive')
        logs.append(math.log(candidate / original))
    center = statistics.mean(logs)
    margin = T_975_DF9 * statistics.stdev(logs) / math.sqrt(10)
    return {'rounds': 10, 'geometric_mean_ratio': math.exp(center),
            'ci95_low': math.exp(center-margin), 'ci95_high': math.exp(center+margin),
            'candidate_over_original_by_round': [math.exp(x) for x in logs],
            'latency_reduction': 1 - math.exp(center),
            'caveat': 'within-session t interval; independent-block assumption; does not correct screening or multiple comparison'}


def passes_session_criterion(result, practical_reduction=0.02):
    if not 0 < practical_reduction < 1:
        raise ValueError('practical threshold must lie in (0,1)')
    return result['latency_reduction'] >= practical_reduction and result['ci95_high'] < 1.0
