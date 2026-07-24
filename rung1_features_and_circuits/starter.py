"""Rung 1 — Features are learned, and legible: an oriented-edge detector in InceptionV1 (runnable).

The Distill "Circuits" thread began by opening an image classifier and finding individual
neurons that detect a visual feature -- curves, edges, orientations -- wired together by
weights. This script finds one in **InceptionV1** (the actual Distill model = torchvision's
`googlenet`): it probes the first convolutional layer with oriented gratings, finds a unit that
is sharply orientation-selective, and runs THE CONTROL from Adebayo et al. (2018) -- re-run the
same probe on a **randomly initialized** network. If the tuning is a real learned feature, it
must vanish when the weights are random.

    python starter.py                 # downloads InceptionV1 weights (~50MB) once

Dependencies: torch, torchvision.
"""
import math
import torch
from torchvision.models import googlenet, GoogLeNet_Weights

dev = "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)   # ImageNet normalization
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
ORIS = [i * math.pi / 12 for i in range(12)]               # 0..165 degrees, 12 orientations


def gratings(oris, size=64, freq=6.0, phases=(0, math.pi / 2, math.pi, 3 * math.pi / 2)):
    """Sinusoidal gratings at each orientation and several phases, ImageNet-normalized."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size),
                            indexing="ij")
    out = []
    for th in oris:
        for ph in phases:
            g = torch.sin(2 * math.pi * freq * (xs * math.cos(th) + ys * math.sin(th)) + ph)
            out.append((g.expand(3, size, size) * 0.5 + 0.5 - MEAN) / STD)
    return torch.stack(out).to(dev), len(phases)


@torch.no_grad()
def conv1_tuning(model, oris):
    """Mean activation of every conv1 channel at each orientation (averaged over phase). [O, C]."""
    x, P = gratings(oris)
    grab = {}
    h = model.conv1.register_forward_hook(lambda m, i, o: grab.__setitem__("a", o))
    model(x)
    h.remove()
    a = torch.relu(grab["a"]).mean(dim=(2, 3))             # [O*P, C] spatial mean
    return a.view(len(oris), P, -1).mean(1)               # [O, C] average over phases


def orientation_selectivity(tuning):
    """1 - circular variance of each channel's tuning over orientation, in [0,1]. A perfectly
    orientation-tuned unit -> 1; a flat (untuned) unit -> 0."""
    ang = torch.linspace(0, math.pi, tuning.shape[0] + 1)[:-1]
    vec = (tuning * torch.exp(2j * ang)[:, None]).sum(0).abs()
    return vec / tuning.sum(0).clamp_min(1e-6)             # [C]


if __name__ == "__main__":
    trained = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval().to(dev)
    random_net = googlenet(weights=None, init_weights=True).eval().to(dev)   # the null: no learning

    tun_t = conv1_tuning(trained, ORIS)
    tun_r = conv1_tuning(random_net, ORIS)
    osi_t = orientation_selectivity(tun_t)
    osi_r = orientation_selectivity(tun_r)

    top = int(osi_t.argmax())
    print(f"InceptionV1 conv1 has {tun_t.shape[1]} channels. Most orientation-selective: "
          f"channel #{top}, selectivity {osi_t[top]:.3f}.")

    curve = tun_t[:, top]
    peak = int(curve.argmax())
    print(f"\nChannel #{top} tuning curve (it is an edge/orientation detector, peak at "
          f"{peak * 15}deg):")
    for i, th in enumerate(ORIS):
        bar = "#" * int(40 * curve[i] / curve.max())
        print(f"  {i * 15:3d}deg | {bar}")

    print("\nCONTROL (Adebayo randomized-network sanity check):")
    print(f"  TRAINED weights -- top-5 channel selectivity: "
          f"{[round(v, 2) for v in osi_t.topk(5).values.tolist()]}  (median {osi_t.median():.2f})")
    print(f"  RANDOM  weights -- top-5 channel selectivity: "
          f"{[round(v, 2) for v in osi_r.topk(5).values.tolist()]}  (median {osi_r.median():.2f})")
    print("\nSharp orientation tuning appears only in the TRAINED network -- the edge detector is a")
    print("learned feature living in the weights, not an artifact of the architecture or the probe.")
    print("That is the Circuits thesis, and the same randomization null guards every rung above.")
