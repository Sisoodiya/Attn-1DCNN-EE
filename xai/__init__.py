"""
Attn-1DCNN-EE — Explainable AI (XAI) Package
==============================================

Component 5: Dual-layer explainability framework.

Layer 1 (Inherent)  — Attention heatmaps from the Soft Attention mechanism.
Layer 2 (Post-hoc)  — SHAP feature attribution (contributors vs offsets).
Layer 3 (Validation) — MDMC perturbation faithfulness evaluation.
Layer 4 (Reporting)  — LLM-enhanced diagnostic report generation.
Layer 5 (Reliability) — Integrated-gradients risk attribution and
                        reliability-centric visual/reporting helpers.
"""

from xai.attention_viz import plot_attention_heatmap, plot_attention_top_channels
from xai.shap_explainer import SHAPExplainer
from xai.faithfulness import FaithfulnessEvaluator
from xai.report import DiagnosticReporter
from xai.reliability_xai import ReliabilityAttributor
from xai.reliability_viz import (
    plot_reliability_heatmap,
    plot_reliability_contrast,
    plot_sensor_importance,
)

__all__ = [
    "plot_attention_heatmap",
    "plot_attention_top_channels",
    "SHAPExplainer",
    "FaithfulnessEvaluator",
    "DiagnosticReporter",
    "ReliabilityAttributor",
    "plot_reliability_heatmap",
    "plot_reliability_contrast",
    "plot_sensor_importance",
]
