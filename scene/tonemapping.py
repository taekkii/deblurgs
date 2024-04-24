
import torch
import torch.nn as nn

class ToneMapping(nn.Module):

    # TODO 
    # somehow make this code smarter (than if elses...)
    def __init__(self, tone_mapping_type:str, eps=1e-8, bound=1/25):
        """
        Tone Mapping (CRF).
        currently only support: x^(1/2.2)
        """
        self.tone_mapping_type = tone_mapping_type
        self.eps = eps
        self.bound = bound
        super().__init__()
    
    def forward(self, x):
        if self.tone_mapping_type == "gamma":
            return ((x-self.bound) / (1.0-2.0*self.bound)).clamp_min(self.eps)  ** (1/2.2)
        elif self.tone_mapping_type == "reverse_gamma":
            return x.clamp_min(self.eps) ** (2.2) * (1.0-2.0*self.bound) + self.bound
        elif self.tone_mapping_type in ["identity", "reverse_identity"]:
            return x
        else:
            raise NotImplementedError("Unknown tone mapping type.")
    
    def inverse(self):
        if "reverse" in self.tone_mapping_type:
            return ToneMapping(self.tone_mapping_type[:8])
        else:
            return ToneMapping("reverse_"+self.tone_mapping_type)
        