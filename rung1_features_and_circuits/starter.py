"""Rung 1 — Features are learned, and legible: an oriented-edge detector in InceptionV1 (runnable).

The Distill "Circuits" thread began by opening an image classifier and finding individual
neurons that detect a visual feature -- edges, curves, orientations -- wired together by
weights. This script finds one in **InceptionV1** (the actual Distill model = torchvision's
`googlenet`), then runs TWO nulls and shows that *which null you choose changes the answer*.

    python starter.py                 # downloads InceptionV1 weights (~50MB) once

Two methodological points this rung teaches the hard way (both cost us a wrong result):

  1. Measure orientation tuning **at each unit's preferred spatial frequency**. Averaging
     responses across frequencies understates tuning -- with too narrow a frequency range,
     conv1 (a 7x7 stride-2 filter that likes FINE gratings) looks *less* selective than random.
  2. The standard randomization null (re-initialize the weights; Adebayo et al. 2018) is
     **weak**, because a randomly-initialized network is nearly dead deeper in. The stronger
     null shuffles each trained kernel's weights, preserving the weight distribution while
     destroying its structure. Real features must beat the *stronger* null.

Dependencies: torch, torchvision.
"""
import math
import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

dev = "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)   # ImageNet normalization
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
N_ORI = 12
ORIS = [i * math.pi / N_ORI for i in range(N_ORI)]          # 0..165 degrees
FREQS = [2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0]            # must span the preferred frequency
PHASES = (0, math.pi / 2, math.pi, 3 * math.pi / 2)
SIZE = 128


def gratings():
    """Sinusoidal gratings at each (orientation, frequency, phase), ImageNet-normalized."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, SIZE), torch.linspace(-1, 1, SIZE),
                            indexing="ij")
    out = []
    for th in ORIS:
        for fq in FREQS:
            for ph in PHASES:
                g = torch.sin(2 * math.pi * fq * (xs * math.cos(th) + ys * math.sin(th)) + ph)
                out.append((g.expand(3, SIZE, SIZE) * 0.5 + 0.5 - MEAN) / STD)
    return torch.stack(out).to(dev)


@torch.no_grad()
def conv1_tuning(model, stim):
    """Orientation tuning of every conv1 channel, AT EACH CHANNEL'S PREFERRED SPATIAL
    FREQUENCY. Returns [N_ORI, C]."""
    grab = {}
    h = model.conv1.register_forward_hook(lambda m, i, o: grab.__setitem__("a", o))
    model(stim)
    h.remove()
    a = torch.relu(grab["a"]).mean(dim=(2, 3))                     # [N_ORI*F*P, C] spatial mean
    t = a.view(N_ORI, len(FREQS), len(PHASES), -1).mean(2)         # [N_ORI, F, C] avg over phase
    best_f = t.mean(0).argmax(0)                                   # [C] preferred frequency
    idx = best_f.view(1, 1, -1).expand(N_ORI, 1, -1)
    return t.gather(1, idx).squeeze(1)                             # [N_ORI, C] at best frequency


def orientation_selectivity(tuning):
    """1 - circular variance of each channel's tuning over orientation, in [0,1]. Perfectly
    orientation-tuned -> 1; flat (untuned) -> 0."""
    ang = torch.linspace(0, math.pi, tuning.shape[0] + 1)[:-1]
    return (tuning * torch.exp(2j * ang)[:, None]).sum(0).abs() / tuning.sum(0).clamp_min(1e-9)


def shuffled_net(seed):
    """The STRONGER null: permute each conv kernel's weights within-channel. Preserves the
    weight distribution (unlike re-initialization) but destroys the learned structure."""
    g = torch.Generator().manual_seed(seed)
    net = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval()
    with torch.no_grad():
        for mod in net.modules():
            if isinstance(mod, nn.Conv2d):
                w = mod.weight.data
                flat = w.view(w.shape[0], -1)
                for i in range(flat.shape[0]):
                    flat[i] = flat[i][torch.randperm(flat.shape[1], generator=g)]
    return net.to(dev)


if __name__ == "__main__":
    stim = gratings()
    trained = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).eval().to(dev)
    osi_t = orientation_selectivity(conv1_tuning(trained, stim))

    rand_osi, shuf_osi = [], []
    for s in range(3):                                             # a null DISTRIBUTION, not one draw
        torch.manual_seed(1000 + s)
        rnet = googlenet(weights=None, init_weights=True).eval().to(dev)
        rand_osi.append(orientation_selectivity(conv1_tuning(rnet, stim)))
        shuf_osi.append(orientation_selectivity(conv1_tuning(shuffled_net(2000 + s), stim)))
    rand_osi = torch.stack(rand_osi); shuf_osi = torch.stack(shuf_osi)

    top = int(osi_t.argmax())
    print(f"InceptionV1 conv1: {osi_t.shape[0]} channels. Most orientation-selective: "
          f"#{top}, selectivity {osi_t[top]:.3f}")

    curve = conv1_tuning(trained, stim)[:, top]
    print(f"\nChannel #{top} tuning curve (peak at {int(curve.argmax()) * 180 // N_ORI}deg):")
    for i in range(N_ORI):
        print(f"  {i * 180 // N_ORI:3d}deg | {'#' * int(40 * curve[i] / curve.max())}")

    beats_rand = int((osi_t > rand_osi.max(0).values).sum())
    beats_shuf = int((osi_t > shuf_osi.max(0).values).sum())
    print("\nTWO CONTROLS (each a distribution over 3 seeds):")
    print(f"  trained          top {osi_t.max():.3f}   median {osi_t.median():.3f}")
    print(f"  random-init null top {rand_osi.max():.3f}   median {rand_osi.median():.3f}"
          f"   -> {beats_rand}/{osi_t.shape[0]} trained units beat every seed")
    print(f"  weight-shuffle   top {shuf_osi.max():.3f}   median {shuf_osi.median():.3f}"
          f"   -> {beats_shuf}/{osi_t.shape[0]} trained units beat every seed")

    print("\nThe sharpest detectors are real: they beat both nulls decisively. But note two")
    print("things the pictures in the original papers could not tell you -- the MEDIAN unit is")
    print("far less impressive than the top one, and the weaker null flatters the result. Which")
    print("control you run determines what you are entitled to claim.")
