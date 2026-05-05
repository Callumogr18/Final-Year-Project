"""
Hybrid scorer test - one invariant check plus a chart that shows how the
formula HybridEval = Sim^alpha * Quality^beta * H^gamma reacts to each
input across a handful of representative scenarios.

Embeddings are mocked so no API key is needed. The chart is written to
tests/LLM_tests/output/hybrid_scorer.png.
"""
import os
import math
import pytest
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from metrics.hybrid.scorer import compute_hybrid_score


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'LLM_tests', 'output')


def embeddings_with_sim(sim_target):
    """
    Return two 2D vectors whose cosine similarity equals sim_target
    """
    theta = math.acos(max(-1.0, min(1.0, sim_target)))
    return lambda texts: [[1.0, 0.0], [math.cos(theta), math.sin(theta)]]


def judge(hall, fluency=1.0, coherence=1.0, consistency=1.0, reasoning=1.0, factual=1.0):
    return {
        'hallucination':    hall,
        'fluency':          fluency,
        'coherence':        coherence,
        'consistency':      consistency,
        'reasoning':        reasoning,
        'factual_accuracy': factual,
    }


SCENARIOS = [
    ('Perfect',            1.0,  judge(1.0)),
    ('Strong overall',     0.85, judge(0.9, fluency=0.9, coherence=0.9, factual=0.9)),
    ('Mediocre quality',   0.75, judge(0.7, fluency=0.5, coherence=0.5,
                                        consistency=0.5, reasoning=0.4, factual=0.6)),
    ('Low similarity',     0.20, judge(1.0)),
    ('Hallucinated',       0.95, judge(0.0)),  # gate kills score
]


def test_hybrid_scorer_chart():
    """
    Run each scenario through compute_hybrid_score and plot the breakdown
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    keys = ['similarity', 'quality', 'hallucination_gate', 'hybrid_score']
    rows = []

    for label, sim, judge in SCENARIOS:
        with patch('metrics.hybrid.scorer.get_embeddings', side_effect=embeddings_with_sim(sim)):
            result = compute_hybrid_score('response', 'reference', judge, 'QA')
        rows.append((label, [result[k] for k in keys]))

        assert 0.0 <= result['hybrid_score'] <= 1.0
        if judge['hallucination'] == 0.0:
            assert result['hybrid_score'] == 0.0

    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows])  # shape: (n_scenarios, 4)

    x = np.arange(len(labels))
    width = 0.20
    colours = ['steelblue', 'darkorange', 'seagreen', 'firebrick']

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, key in enumerate(keys):
        offsets = (i - 1.5) * width
        ax.bar(x + offsets, values[:, i], width, label=key, color=colours[i])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title('HybridEval components per scenario (QA weights)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'hybrid_scorer.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)

    assert os.path.exists(path) and os.path.getsize(path) > 0
