from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_output(prefix: str, raw: Dict[str, Any]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in raw.items():
        feature_key = f"{prefix}{key}"
        if _is_number(value):
            output[feature_key] = float(value)
        elif isinstance(value, (list, tuple, set)):
            output[f"{feature_key}_count"] = float(len(value))
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if _is_number(sub_value):
                    output[f"{feature_key}_{sub_key}"] = float(sub_value)
    return output


@dataclass
class _BaseAnalysisFeature(BaseFeature):
    module_path: str = ""
    analyzer_class: str = ""
    key_prefix: str = ""
    cache_key: str = ""
    _analyzer: Any = field(default=None, init=False, repr=False)
    _load_failed: bool = field(default=False, init=False, repr=False)

    def initialize(self) -> None:
        if self._analyzer is not None or self._load_failed:
            return
        try:
            module = importlib.import_module(self.module_path)
            analyzer_type = getattr(module, self.analyzer_class)
            self._analyzer = analyzer_type()
        except Exception as exc:  # noqa: BLE001
            self._load_failed = True
            logger.exception("Analyzer init failed for %s: %s", self.name, exc)

    def _analyze(self, context: FeatureContext) -> Dict[str, Any]:
        if self._analyzer is None:
            return {}
        return self._analyzer.analyze(context.text)

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not isinstance(context.text, str):
            logger.warning("Skipping %s: context.text is not a string", self.name)
            return {}
        if not context.text.strip():
            return {}
        if self._analyzer is None:
            self.initialize()
        if self._analyzer is None:
            return {}

        try:
            raw = self._analyze(context)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Runtime failure in %s: %s", self.name, exc)
            return {}

        if isinstance(raw, dict):
            context.cache[self.cache_key] = raw
            return _numeric_output(self.key_prefix, raw)
        return {}


@dataclass
@register_feature
class AnalysisArgumentMiningFeature(_BaseAnalysisFeature):
    name: str = "analysis_argument_mining_feature"
    description: str = "Adapter for src.analysis.argument_mining"
    module_path: str = "src.analysis.argument_mining"
    analyzer_class: str = "ArgumentMiningAnalyzer"
    key_prefix: str = "analysis_argument_"
    cache_key: str = "analysis_argument"


@dataclass
@register_feature
class AnalysisContextOmissionFeature(_BaseAnalysisFeature):
    name: str = "analysis_context_omission_feature"
    description: str = "Adapter for src.analysis.context_omission_detector"
    module_path: str = "src.analysis.context_omission_detector"
    analyzer_class: str = "ContextOmissionDetector"
    key_prefix: str = "analysis_context_omission_"
    cache_key: str = "analysis_context_omission"


@dataclass
@register_feature
class AnalysisDiscourseCoherenceFeature(_BaseAnalysisFeature):
    name: str = "analysis_discourse_coherence_feature"
    description: str = "Adapter for src.analysis.discourse_coherence_analyzer"
    module_path: str = "src.analysis.discourse_coherence_analyzer"
    analyzer_class: str = "DiscourseCoherenceAnalyzer"
    key_prefix: str = "analysis_discourse_"
    cache_key: str = "analysis_discourse"


@dataclass
@register_feature
class AnalysisEmotionTargetFeature(_BaseAnalysisFeature):
    name: str = "analysis_emotion_target_feature"
    description: str = "Adapter for src.analysis.emotion_target_analysis"
    module_path: str = "src.analysis.emotion_target_analysis"
    analyzer_class: str = "EmotionTargetAnalyzer"
    key_prefix: str = "analysis_emotion_target_"
    cache_key: str = "analysis_emotion_target"


@dataclass
@register_feature
class AnalysisFramingFeature(_BaseAnalysisFeature):
    name: str = "analysis_framing_feature"
    description: str = "Adapter for src.analysis.framing_analysis"
    module_path: str = "src.analysis.framing_analysis"
    analyzer_class: str = "FramingAnalyzer"
    key_prefix: str = "analysis_framing_"
    cache_key: str = "analysis_framing"


@dataclass
@register_feature
class AnalysisIdeologicalLanguageFeature(_BaseAnalysisFeature):
    name: str = "analysis_ideological_language_feature"
    description: str = "Adapter for src.analysis.ideological_language_detector"
    module_path: str = "src.analysis.ideological_language_detector"
    analyzer_class: str = "IdeologicalLanguageDetector"
    key_prefix: str = "analysis_ideological_"
    cache_key: str = "analysis_ideological"


@dataclass
@register_feature
class AnalysisInformationDensityFeature(_BaseAnalysisFeature):
    name: str = "analysis_information_density_feature"
    description: str = "Adapter for src.analysis.information_density_analyzer"
    module_path: str = "src.analysis.information_density_analyzer"
    analyzer_class: str = "InformationDensityAnalyzer"
    key_prefix: str = "analysis_information_density_"
    cache_key: str = "analysis_information_density"


@dataclass
@register_feature
class AnalysisInformationOmissionFeature(_BaseAnalysisFeature):
    name: str = "analysis_information_omission_feature"
    description: str = "Adapter for src.analysis.information_omission_detector"
    module_path: str = "src.analysis.information_omission_detector"
    analyzer_class: str = "InformationOmissionDetector"
    key_prefix: str = "analysis_information_omission_"
    cache_key: str = "analysis_information_omission"


@dataclass
@register_feature
class AnalysisNarrativeRoleFeature(_BaseAnalysisFeature):
    name: str = "analysis_narrative_role_feature"
    description: str = "Adapter for src.analysis.narrative_role_extractor"
    module_path: str = "src.analysis.narrative_role_extractor"
    analyzer_class: str = "NarrativeRoleExtractor"
    key_prefix: str = "analysis_narrative_role_"
    cache_key: str = "analysis_narrative_role"


@dataclass
@register_feature
class AnalysisNarrativeConflictFeature(_BaseAnalysisFeature):
    name: str = "analysis_narrative_conflict_feature"
    description: str = "Adapter for src.analysis.narrative_conflict"
    module_path: str = "src.analysis.narrative_conflict"
    analyzer_class: str = "NarrativeConflictAnalyzer"
    key_prefix: str = "analysis_narrative_conflict_"
    cache_key: str = "analysis_narrative_conflict"


@dataclass
@register_feature
class AnalysisNarrativePropagationFeature(_BaseAnalysisFeature):
    name: str = "analysis_narrative_propagation_feature"
    description: str = "Adapter for src.analysis.narrative_propagation"
    module_path: str = "src.analysis.narrative_propagation"
    analyzer_class: str = "NarrativePropagationAnalyzer"
    key_prefix: str = "analysis_narrative_propagation_"
    cache_key: str = "analysis_narrative_propagation"


@dataclass
@register_feature
class AnalysisNarrativeTemporalFeature(_BaseAnalysisFeature):
    name: str = "analysis_narrative_temporal_feature"
    description: str = "Adapter for src.analysis.narrative_temporal_analyzer"
    module_path: str = "src.analysis.narrative_temporal_analyzer"
    analyzer_class: str = "NarrativeTemporalAnalyzer"
    key_prefix: str = "analysis_narrative_temporal_"
    cache_key: str = "analysis_narrative_temporal"


@dataclass
@register_feature
class AnalysisRhetoricalDeviceFeature(_BaseAnalysisFeature):
    name: str = "analysis_rhetorical_device_feature"
    description: str = "Adapter for src.analysis.rhetorical_device_detector"
    module_path: str = "src.analysis.rhetorical_device_detector"
    analyzer_class: str = "RhetoricalDeviceDetector"
    key_prefix: str = "analysis_rhetorical_"
    cache_key: str = "analysis_rhetorical"


@dataclass
@register_feature
class AnalysisSourceAttributionFeature(_BaseAnalysisFeature):
    name: str = "analysis_source_attribution_feature"
    description: str = "Adapter for src.analysis.source_attribution_analyzer"
    module_path: str = "src.analysis.source_attribution_analyzer"
    analyzer_class: str = "SourceAttributionAnalyzer"
    key_prefix: str = "analysis_source_attribution_"
    cache_key: str = "analysis_source_attribution"


@dataclass
@register_feature
class AnalysisPropagandaPatternFeature(_BaseAnalysisFeature):
    name: str = "analysis_propaganda_pattern_feature"
    description: str = "Adapter for src.analysis.propaganda_pattern_detector"
    module_path: str = "src.analysis.propaganda_pattern_detector"
    analyzer_class: str = "PropagandaPatternDetector"
    key_prefix: str = "analysis_propaganda_pattern_"
    cache_key: str = "analysis_propaganda_pattern"

    def _analyze(self, context: FeatureContext) -> Dict[str, Any]:
        if self._analyzer is None:
            return {}
        return self._analyzer.analyze(
            emotion_features=context.cache.get("analysis_emotion_target", {}),
            narrative_features=context.cache.get("analysis_narrative_conflict", {}),
            rhetorical_features=context.cache.get("analysis_rhetorical", {}),
            argument_features=context.cache.get("analysis_argument", {}),
            information_features=context.cache.get("analysis_information_density", {}),
        )


@dataclass
@register_feature
class AnalysisBiasProfileFeature(_BaseAnalysisFeature):
    name: str = "analysis_bias_profile_feature"
    description: str = "Adapter for src.analysis.bias_profile_builder"
    module_path: str = "src.analysis.bias_profile_builder"
    analyzer_class: str = "BiasProfileBuilder"
    key_prefix: str = "analysis_bias_profile_"
    cache_key: str = "analysis_bias_profile"

    def _analyze(self, context: FeatureContext) -> Dict[str, Any]:
        if self._analyzer is None:
            return {}

        profile = self._analyzer.build_profile(
            bias_features={
                **context.cache.get("analysis_framing", {}),
                **context.cache.get("analysis_ideological", {}),
                **context.cache.get("analysis_context_omission", {}),
            },
            emotion_features=context.cache.get("analysis_emotion_target", {}),
            narrative_features={
                **context.cache.get("analysis_narrative_conflict", {}),
                **context.cache.get("analysis_narrative_temporal", {}),
                **context.cache.get("analysis_narrative_propagation", {}),
            },
            discourse_features={
                **context.cache.get("analysis_discourse", {}),
                **context.cache.get("analysis_argument", {}),
                **context.cache.get("analysis_source_attribution", {}),
            },
            ideology_predictions={},
        )

        metrics = profile.get("metrics", {})
        output: Dict[str, Any] = {
            "bias_score": profile.get("bias_score", 0.0),
        }
        if isinstance(metrics, dict):
            output.update(metrics)
        return output

