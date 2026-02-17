"""Tests for quality metric visualization functions."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from kvshuttle.visualization.quality import (
    plot_cosine_vs_perplexity,
    plot_perplexity_delta,
    plot_token_agreement,
)


@pytest.fixture
def sample_bar_data():
    """Synthetic data for bar-chart plots."""
    return {
        "uniform_int8": {"model_a": [0.5, 0.6, 0.4], "model_b": [0.3, 0.2]},
        "kivi_2bit": {"model_a": [1.2, 1.1], "model_b": [0.9]},
    }


class TestPerplexityDeltaPlot:
    """Tests for plot_perplexity_delta()."""

    def test_returns_figure(self, sample_bar_data):
        fig = plot_perplexity_delta(sample_bar_data)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, sample_bar_data, tmp_path):
        out = tmp_path / "ppl_delta.pdf"
        fig = plot_perplexity_delta(sample_bar_data, output_path=str(out))
        assert isinstance(fig, plt.Figure)
        assert out.exists()
        assert out.stat().st_size > 0


class TestTokenAgreementPlot:
    """Tests for plot_token_agreement()."""

    def test_returns_figure(self, sample_bar_data):
        fig = plot_token_agreement(sample_bar_data)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, sample_bar_data, tmp_path):
        out = tmp_path / "token_agreement.pdf"
        fig = plot_token_agreement(sample_bar_data, output_path=str(out))
        assert isinstance(fig, plt.Figure)
        assert out.exists()
        assert out.stat().st_size > 0


class TestCosineVsPerplexity:
    """Tests for plot_cosine_vs_perplexity()."""

    def test_returns_figure(self):
        cosine = {"uniform_int8": 0.98, "kivi_2bit": 0.85}
        ppl = {"uniform_int8": 0.5, "kivi_2bit": 1.2}
        fig = plot_cosine_vs_perplexity(cosine, ppl)
        assert isinstance(fig, plt.Figure)

    def test_saves_to_file(self, tmp_path):
        cosine = {"uniform_int8": 0.98}
        ppl = {"uniform_int8": 0.5}
        out = tmp_path / "cosine_ppl.pdf"
        fig = plot_cosine_vs_perplexity(cosine, ppl, output_path=str(out))
        assert isinstance(fig, plt.Figure)
        assert out.exists()

    def test_handles_empty_common_set(self):
        cosine = {"compressor_a": 0.9}
        ppl = {"compressor_b": 1.0}
        fig = plot_cosine_vs_perplexity(cosine, ppl)
        assert isinstance(fig, plt.Figure)
