from defenses.victim.reversesigmoid import ReverseSigmoid as LegacyReverseSigmoid

from flowguard.defenses.prediction.base import LegacyPredictionDefense


class ReverseSigmoidDefense(LegacyPredictionDefense):
    name = "reverse_sigmoid"
    legacy_class = LegacyReverseSigmoid
