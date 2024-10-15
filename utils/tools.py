import numpy as np
import torch 

def aplatir(l):
    # Cas de base: si l n'est pas une liste, retourne une liste contenant l
    if not isinstance(l, list):
        return [l]
    # Cas récursif: si l est une liste, applique aplatir à chaque élément et combine les résultats
    else:
        return [item for sublist in l for item in aplatir(sublist)]
    
def make_gradient_clip_hook(max_norm):
    def gradient_clip_hook(grad):
        norm = torch.norm(grad)
        if norm > max_norm:
            return grad * (max_norm / norm)
        return grad
    return gradient_clip_hook

def gradient_clip_hook(grad, max_norm=1.0):
    norm = torch.norm(grad)
    if norm > max_norm:
        print(norm)
        return grad * (max_norm / norm)
    return grad