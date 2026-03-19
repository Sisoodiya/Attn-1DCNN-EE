"""
Diagnostic Reporter — XAI Layer 4: LLM-Enhanced Reporting
==========================================================

Translates SHAP attributions and attention summaries into structured,
human-readable diagnostic reports via an LLM.  The reporter builds an
**impact-based system prompt** containing:

* The predicted fault class (or "Unknown Fault" flag).
* Top contributing sensors with their SHAP magnitude and deviation
  direction.
* Top offset sensors counteracting the diagnosis.
* A temporal attention summary (which time window was most critical).

The prompt is sent to a user-supplied LLM callable.  If no LLM is
configured, the fully formatted prompt is returned so the operator can
paste it into any LLM interface manually.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


_SYSTEM_PROMPT = """\
You are an expert nuclear power plant diagnostic AI assistant.
You analyse sensor attribution data from the Attn-1DCNN-EE fault
detection model deployed on a pressurised water reactor (PWR).

Your task:
1. Identify the most likely accident type based on the contributing
   sensor deviations provided.
2. Explain the physical chain of events that connects the deviating
   sensors (e.g. a pressure drop causing flow rate changes).
3. Highlight the top areas of concern for the plant operators.
4. Recommend immediate actions or further monitoring steps.

Be concise, precise, and safety-oriented.  Always err on the side of
caution when the fault type is flagged as "Unknown".\
"""

_RELIABILITY_SYSTEM_PROMPT = """\
You are an expert nuclear reliability analysis assistant.
You analyze reliability degradation events from an Attn-1DCNN-EE
monitoring pipeline for a pressurized water reactor.

Your task:
1. Explain why reliability dropped at the reported operating timestamp.
2. Link the drop to sensor-time attribution evidence.
3. Distinguish likely true degradation from potential false alarms.
4. Recommend operator actions and additional verification checks.

Be concise, evidence-based, and safety-oriented.\
"""


class DiagnosticReporter:
    """Build diagnostic prompts and optionally invoke an LLM.

    Parameters
    ----------
    llm_fn : callable | None
        ``(system_prompt: str, user_prompt: str) → str``.
        A function that calls an LLM and returns the generated text.
        If ``None``, :meth:`generate_report` returns the raw prompt
        for manual use.
    system_prompt : str | None
        Override the default nuclear-diagnostic system prompt.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable[[str, str], str]] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm_fn = llm_fn
        self.system_prompt = system_prompt or _SYSTEM_PROMPT
        self.reliability_system_prompt = _RELIABILITY_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        prediction: str,
        contributors: List[Tuple[str, float]],
        offsets: List[Tuple[str, float]],
        attention_peak_timesteps: Optional[List[int]] = None,
        sample_id: Optional[int] = None,
    ) -> str:
        """Assemble the impact-based diagnostic user prompt.

        Parameters
        ----------
        prediction : str
            Predicted class name or ``"Unknown Fault"``.
        contributors : list[(name, shap_value)]
            Top sensors pushing toward the diagnosed fault.
        offsets : list[(name, shap_value)]
            Top sensors counteracting the diagnosis.
        attention_peak_timesteps : list[int] | None
            Time-step indices where attention was highest.
        sample_id : int | None
            Optional sample identifier for traceability.

        Returns
        -------
        str
            The fully formatted user prompt.
        """
        lines: list[str] = []

        if sample_id is not None:
            lines.append(f"Sample ID: {sample_id}")
        lines.append(f"Model Diagnosis: {prediction}")
        lines.append("")

        # Contributors
        lines.append("## Top Contributing Sensors (pushing toward diagnosis)")
        for name, val in contributors:
            direction = "INCREASE" if val > 0 else "DECREASE"
            lines.append(f"  - {name}: SHAP = {val:+.4f} ({direction})")
        lines.append("")

        # Offsets
        if offsets:
            lines.append("## Top Offsetting Sensors (counteracting diagnosis)")
            for name, val in offsets:
                direction = "INCREASE" if val > 0 else "DECREASE"
                lines.append(f"  - {name}: SHAP = {val:+.4f} ({direction})")
            lines.append("")

        # Temporal focus
        if attention_peak_timesteps:
            lines.append("## Critical Time Window")
            lines.append(
                f"  Highest attention at time steps: "
                f"{attention_peak_timesteps}"
            )
            lines.append("")

        lines.append(
            "Based on the sensor deviations above, provide a diagnostic "
            "analysis explaining the likely physical root cause, areas of "
            "concern, and recommended operator actions."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        prediction: str,
        contributors: List[Tuple[str, float]],
        offsets: List[Tuple[str, float]],
        attention_peak_timesteps: Optional[List[int]] = None,
        sample_id: Optional[int] = None,
    ) -> Dict[str, str]:
        """Generate a full diagnostic report.

        Parameters
        ----------
        prediction, contributors, offsets, attention_peak_timesteps,
        sample_id
            See :meth:`build_prompt`.

        Returns
        -------
        dict
            ``{"system_prompt": str, "user_prompt": str,
              "report": str | None}``.
            ``report`` is the LLM-generated text, or ``None`` if no
            LLM was configured.
        """
        user_prompt = self.build_prompt(
            prediction=prediction,
            contributors=contributors,
            offsets=offsets,
            attention_peak_timesteps=attention_peak_timesteps,
            sample_id=sample_id,
        )

        report = None
        if self.llm_fn is not None:
            report = self.llm_fn(self.system_prompt, user_prompt)

        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "report": report,
        }

    # ------------------------------------------------------------------
    # Reliability-mode reporting
    # ------------------------------------------------------------------

    def build_reliability_prompt(
        self,
        reliability_summary: Dict[str, Any],
        event: Optional[Dict[str, Any]],
        contributors: List[Tuple[str, float]],
        offsets: List[Tuple[str, float]],
        attention_peak_timesteps: Optional[List[int]] = None,
        sample_id: Optional[int] = None,
    ) -> str:
        """Build a prompt focused on reliability degradation events."""
        lines: List[str] = []

        if sample_id is not None:
            lines.append(f"Sample ID: {sample_id}")

        if reliability_summary:
            lines.append("## Reliability Metrics")
            if "failure_rate" in reliability_summary:
                lines.append(
                    f"  - Failure rate (lambda): "
                    f"{float(reliability_summary['failure_rate']):.6f}"
                )
            if "mttf" in reliability_summary:
                lines.append(
                    f"  - MTTF: {float(reliability_summary['mttf']):.6f}"
                )
            if "operating_time" in reliability_summary:
                lines.append(
                    f"  - Operating time: "
                    f"{float(reliability_summary['operating_time']):.6f}"
                )
            if "failure_count" in reliability_summary:
                lines.append(
                    f"  - Failure events: "
                    f"{int(reliability_summary['failure_count'])}"
                )
            lines.append("")

        if event is not None:
            lines.append("## Reliability Drop Event")
            lines.append(f"  - Timestamp: {event.get('timestamp')}")
            lines.append(f"  - Risk score: {event.get('risk_score', 'n/a')}")
            lines.append(
                f"  - Accepted envelopes: {event.get('accept_count', 'n/a')}"
            )
            lines.append(
                f"  - Nearest class context: "
                f"{event.get('nearest_class_name', 'n/a')}"
            )
            lines.append("")

        lines.append("## Top Contributing Sensors (degradation drivers)")
        for name, val in contributors:
            direction = "INCREASE" if val > 0 else "DECREASE"
            lines.append(f"  - {name}: attribution = {val:+.4f} ({direction})")
        lines.append("")

        if offsets:
            lines.append("## Offsetting Sensors")
            for name, val in offsets:
                direction = "INCREASE" if val > 0 else "DECREASE"
                lines.append(f"  - {name}: attribution = {val:+.4f} ({direction})")
            lines.append("")

        if attention_peak_timesteps:
            lines.append("## Critical Time Steps")
            lines.append(f"  - Peaks: {attention_peak_timesteps}")
            lines.append("")

        lines.append(
            "Provide a reliability-centric explanation describing why "
            "reliability dropped, the likely physical mechanism, whether "
            "this appears to be a true event or potential false alarm, and "
            "recommended operator actions."
        )

        return "\n".join(lines)

    def generate_reliability_report(
        self,
        reliability_summary: Dict[str, Any],
        event: Optional[Dict[str, Any]],
        contributors: List[Tuple[str, float]],
        offsets: List[Tuple[str, float]],
        attention_peak_timesteps: Optional[List[int]] = None,
        sample_id: Optional[int] = None,
    ) -> Dict[str, Optional[str]]:
        """Generate reliability-focused narrative report."""
        user_prompt = self.build_reliability_prompt(
            reliability_summary=reliability_summary,
            event=event,
            contributors=contributors,
            offsets=offsets,
            attention_peak_timesteps=attention_peak_timesteps,
            sample_id=sample_id,
        )

        report = None
        if self.llm_fn is not None:
            report = self.llm_fn(self.reliability_system_prompt, user_prompt)

        return {
            "system_prompt": self.reliability_system_prompt,
            "user_prompt": user_prompt,
            "report": report,
        }
