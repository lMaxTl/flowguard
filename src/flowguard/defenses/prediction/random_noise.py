from defenses.victim.randnoise import RandomNoise as LegacyRandomNoise

from flowguard.defenses.prediction.base import LegacyPredictionDefense


class RandomNoiseDefense(LegacyPredictionDefense):
    name = "random_noise"
    legacy_class = LegacyRandomNoise
