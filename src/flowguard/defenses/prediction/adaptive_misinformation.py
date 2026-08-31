from defenses.victim.am import AM as LegacyAdaptiveMisinformation

from flowguard.defenses.prediction.base import LegacyPredictionDefense


class AdaptiveMisinformationDefense(LegacyPredictionDefense):
    name = "adaptive_misinformation"
    legacy_class = LegacyAdaptiveMisinformation
