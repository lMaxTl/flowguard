import torch
from defenses.victim.mad import MAD as LegacyMad

from flowguard.defenses.prediction.base import LegacyPredictionDefense


class MadDefense(LegacyPredictionDefense):
    name = "mad"
    legacy_class = LegacyMad

    def transform(
        self,
        probabilities: torch.Tensor,
        *,
        logits: torch.Tensor | None = None,
        inputs: torch.Tensor | None = None,
        auxiliary: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.legacy is None:
            raise RuntimeError("prepare() must be called before transform()")
            
        if probabilities.dim() > 1 and probabilities.size(0) > 1:
            results = []
            batch_size = probabilities.size(0)
            num_classes = probabilities.size(1)
            for i in range(probabilities.size(0)):
                prob = probabilities[i:i+1]
                if auxiliary is not None:
                    if auxiliary.dim() == 3 and auxiliary.size(0) == batch_size:
                        info = auxiliary[i]
                    elif auxiliary.dim() == 2 and auxiliary.size(0) == batch_size * num_classes:
                        info = auxiliary[i * num_classes : (i + 1) * num_classes]
                    elif auxiliary.dim() == 2 and auxiliary.size(0) == num_classes:
                        info = auxiliary
                    else:
                        info = auxiliary[i:i+1]
                else:
                    info = None
                results.append(self.legacy.get_yprime(prob, x_info=info))
            return torch.cat(results, dim=0)
        else:
            return self.legacy.get_yprime(probabilities, x_info=auxiliary)
