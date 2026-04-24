# src/analysis/base_analyzer.py

from abc import ABC, abstractmethod
from src.analysis.feature_context import FeatureContext


class BaseAnalyzer(ABC):

    @abstractmethod
    def analyze(self, ctx: FeatureContext) -> dict:
        pass