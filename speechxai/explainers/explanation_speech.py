from dataclasses import dataclass

import numpy as np
from typing import Optional


@dataclass
class ExplanationSpeech:
    features: list
    scores: np.array
    explainer: str
    target: list
    audio_path: Optional[str] = None


@dataclass
class EvaluationSpeech:
    """Generic class to represent an Evaluation"""

    name: str
    score: list
    target: list
