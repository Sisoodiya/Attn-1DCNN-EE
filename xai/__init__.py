"""
Attn-1DCNN-EE — Explainable AI (XAI) Package
==============================================

Component 5: Dual-layer explainability framework.

Layer 1 (Inherent)  — Attention heatmaps from the Soft Attention mechanism.
Layer 2 (Post-hoc)  — SHAP feature attribution (contributors vs offsets).
Layer 3 (Validation) — MDMC perturbation faithfulness evaluation.
Layer 4 (Reporting)  — LLM-enhanced diagnostic report generation.
"""

from xai.attention_viz import plot_attention_heatmap, plot_attention_top_channels
from xai.shap_explainer import SHAPExplainer
from xai.faithfulness import FaithfulnessEvaluator
from xai.report import DiagnosticReporter

__all__ = [
    "plot_attention_heatmap",
    "plot_attention_top_channels",
    "SHAPExplainer",
    "FaithfulnessEvaluator",
    "DiagnosticReporter",
]
