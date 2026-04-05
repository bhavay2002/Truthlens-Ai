from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.analysis.argument_mining import ArgumentMiningAnalyzer
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.analysis.context_omission_detector import ContextOmissionDetector
from src.analysis.discourse_coherence_analyzer import DiscourseCoherenceAnalyzer
from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
from src.analysis.framing_analysis import FramingAnalyzer
from src.analysis.ideological_language_detector import IdeologicalLanguageDetector
from src.analysis.information_density_analyzer import InformationDensityAnalyzer
from src.analysis.information_omission_detector import InformationOmissionDetector
from src.analysis.narrative_conflict import NarrativeConflictAnalyzer
from src.analysis.narrative_propagation import NarrativePropagationAnalyzer
from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
from src.analysis.narrative_temporal_analyzer import NarrativeTemporalAnalyzer
from src.analysis.propaganda_pattern_detector import PropagandaPatternDetector
from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
from src.analysis.source_attribution_analyzer import SourceAttributionAnalyzer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisIntegrationRunner:
    """
    Executes analysis modules and returns a unified dictionary.
    """

    argument_mining: Optional[ArgumentMiningAnalyzer] = None
    context_omission: Optional[ContextOmissionDetector] = None
    discourse_coherence: Optional[DiscourseCoherenceAnalyzer] = None
    emotion_target: Optional[EmotionTargetAnalyzer] = None
    framing: Optional[FramingAnalyzer] = None
    ideological_language: Optional[IdeologicalLanguageDetector] = None
    information_density: Optional[InformationDensityAnalyzer] = None
    information_omission: Optional[InformationOmissionDetector] = None
    narrative_conflict: Optional[NarrativeConflictAnalyzer] = None
    narrative_propagation: Optional[NarrativePropagationAnalyzer] = None
    narrative_role: Optional[NarrativeRoleExtractor] = None
    narrative_temporal: Optional[NarrativeTemporalAnalyzer] = None
    propaganda_pattern: Optional[PropagandaPatternDetector] = None
    rhetorical_device: Optional[RhetoricalDeviceDetector] = None
    source_attribution: Optional[SourceAttributionAnalyzer] = None
    bias_profile_builder: Optional[BiasProfileBuilder] = None

    def __post_init__(self) -> None:
        self.argument_mining = self.argument_mining or self._safe_init(
            "argument_mining", ArgumentMiningAnalyzer
        )
        self.context_omission = self.context_omission or self._safe_init(
            "context_omission", ContextOmissionDetector
        )
        self.discourse_coherence = self.discourse_coherence or self._safe_init(
            "discourse_coherence", DiscourseCoherenceAnalyzer
        )
        self.emotion_target = self.emotion_target or self._safe_init(
            "emotion_target", EmotionTargetAnalyzer
        )
        self.framing = self.framing or self._safe_init("framing", FramingAnalyzer)
        self.ideological_language = self.ideological_language or self._safe_init(
            "ideological_language", IdeologicalLanguageDetector
        )
        self.information_density = self.information_density or self._safe_init(
            "information_density", InformationDensityAnalyzer
        )
        self.information_omission = self.information_omission or self._safe_init(
            "information_omission", InformationOmissionDetector
        )
        self.narrative_conflict = self.narrative_conflict or self._safe_init(
            "narrative_conflict", NarrativeConflictAnalyzer
        )
        self.narrative_propagation = self.narrative_propagation or self._safe_init(
            "narrative_propagation", NarrativePropagationAnalyzer
        )
        self.narrative_role = self.narrative_role or self._safe_init(
            "narrative_role", NarrativeRoleExtractor
        )
        self.narrative_temporal = self.narrative_temporal or self._safe_init(
            "narrative_temporal", NarrativeTemporalAnalyzer
        )
        self.propaganda_pattern = self.propaganda_pattern or self._safe_init(
            "propaganda_pattern", PropagandaPatternDetector
        )
        self.rhetorical_device = self.rhetorical_device or self._safe_init(
            "rhetorical_device", RhetoricalDeviceDetector
        )
        self.source_attribution = self.source_attribution or self._safe_init(
            "source_attribution", SourceAttributionAnalyzer
        )
        self.bias_profile_builder = self.bias_profile_builder or self._safe_init(
            "bias_profile_builder", BiasProfileBuilder
        )

    def _safe_init(self, name: str, cls: Any) -> Any | None:
        try:
            return cls()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Analysis module unavailable: %s (%s)", name, exc)
            return None

    def _safe_analyze(self, name: str, analyzer: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if analyzer is None:
            return {}
        try:
            output = analyzer.analyze(*args, **kwargs)
            return output if isinstance(output, dict) else {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Analysis module failed at runtime: %s (%s)", name, exc)
            return {}

    def analyze_text(self, text: str) -> Dict[str, Dict[str, Any]]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        outputs: Dict[str, Dict[str, Any]] = {}

        outputs["argument_mining"] = self._safe_analyze(
            "argument_mining", self.argument_mining, text
        )
        outputs["context_omission"] = self._safe_analyze(
            "context_omission", self.context_omission, text
        )
        outputs["discourse_coherence"] = self._safe_analyze(
            "discourse_coherence", self.discourse_coherence, text
        )
        outputs["emotion_target"] = self._safe_analyze(
            "emotion_target", self.emotion_target, text
        )
        outputs["framing"] = self._safe_analyze("framing", self.framing, text)
        outputs["ideological_language"] = self._safe_analyze(
            "ideological_language", self.ideological_language, text
        )
        outputs["information_density"] = self._safe_analyze(
            "information_density", self.information_density, text
        )
        outputs["information_omission"] = self._safe_analyze(
            "information_omission", self.information_omission, text
        )
        outputs["narrative_temporal"] = self._safe_analyze(
            "narrative_temporal", self.narrative_temporal, text
        )
        outputs["rhetorical_device"] = self._safe_analyze(
            "rhetorical_device", self.rhetorical_device, text
        )
        outputs["source_attribution"] = self._safe_analyze(
            "source_attribution", self.source_attribution, text
        )
        outputs["narrative_role"] = self._safe_analyze(
            "narrative_role", self.narrative_role, text
        )

        roles = outputs["narrative_role"]
        hero_entities = roles.get("hero_entities", []) if isinstance(roles, dict) else []
        villain_entities = (
            roles.get("villain_entities", []) if isinstance(roles, dict) else []
        )
        victim_entities = roles.get("victim_entities", []) if isinstance(roles, dict) else []

        outputs["narrative_conflict"] = self._safe_analyze(
            "narrative_conflict",
            self.narrative_conflict,
            text,
            hero_entities=hero_entities,
            villain_entities=villain_entities,
            victim_entities=victim_entities,
        )
        outputs["narrative_propagation"] = self._safe_analyze(
            "narrative_propagation",
            self.narrative_propagation,
            text,
            hero_entities=hero_entities,
            villain_entities=villain_entities,
            victim_entities=victim_entities,
        )

        outputs["propaganda_pattern"] = (
            self.propaganda_pattern.analyze(
                emotion_features=outputs.get("emotion_target", {}),
                narrative_features=outputs.get("narrative_conflict", {}),
                rhetorical_features=outputs.get("rhetorical_device", {}),
                argument_features=outputs.get("argument_mining", {}),
                information_features=outputs.get("information_density", {}),
            )
            if self.propaganda_pattern is not None
            else {}
        )

        if self.bias_profile_builder is not None:
            try:
                profile = self.bias_profile_builder.build_profile(
                    bias_features={
                        **outputs.get("framing", {}),
                        **outputs.get("ideological_language", {}),
                        **outputs.get("context_omission", {}),
                    },
                    emotion_features=outputs.get("emotion_target", {}),
                    narrative_features={
                        **outputs.get("narrative_conflict", {}),
                        **outputs.get("narrative_propagation", {}),
                        **outputs.get("narrative_temporal", {}),
                    },
                    discourse_features={
                        **outputs.get("argument_mining", {}),
                        **outputs.get("discourse_coherence", {}),
                        **outputs.get("source_attribution", {}),
                    },
                    ideology_predictions={},
                )
                outputs["bias_profile"] = profile if isinstance(profile, dict) else {}
            except Exception as exc:  # noqa: BLE001
                logger.warning("Analysis module failed at runtime: bias_profile (%s)", exc)
                outputs["bias_profile"] = {}
        else:
            outputs["bias_profile"] = {}

        return outputs

