from defenses.victim.mld import MLD as LegacyMld

from flowguard.defenses.prediction.base import LegacyPredictionDefense


class ModelGuardDefense(LegacyPredictionDefense):
    name = "modelguard"
    legacy_class = LegacyMld
