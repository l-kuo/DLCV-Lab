# Deep Learning for Computer Vision — Lab Sessions

**Asian Institute of Technology · August 2026**

| | |
|---|---|
| **Lecturer** | Assoc. Prof. Dr. Mongkol Ekpanyapong |
| **Teaching Assistant** | Lin Tun Naing |
| **Language** | Python 3.11 · PyTorch |
| **Compute** | Google Colab free tier (NVIDIA T4, 16 GB) |

Every lab runs end to end on a **free Colab T4**.

---

## Lab outline

| # | Lab | You build | Min: GPU time |
|---|---|---|---|
| 00 | [Introduction and Setup](DLCV-00-Introduction-and-Setup/) | environment check, four demos of what CV can do today | 5 min |
| 01 | [Python Basics](DLCV-01-Python-Basics/) | Python fluency, `nn.Module` prerequisites | — |
| 02 | [Neural Networks from Scratch](DLCV-02-Neural-Networks-from-Scratch/) | forward pass, loss, **backprop**, SGD — in NumPy | — |
| 03 | [CNN and PyTorch](DLCV-03-CNN-and-PyTorch/) | convolution by hand, then PyTorch idioms | 10 min |
| 04 | [CNN Architectures](DLCV-04-CNN-Architectures/) | ResNet, SE blocks, ConvNeXt | 20 min |
| 05 | [Transfer Learning and Evaluation](DLCV-05-Transfer-Learning-and-Evaluation/) | scratch vs probe vs fine-tune, calibration, Grad-CAM, DINOv2 | 20 min |
| 06 | [Vision Transformer](DLCV-06-Vision-Transformer/) | self-attention, ViT from scratch, attention rollout | 15 min |
| 07 | [Object Detection](DLCV-07-Object-Detection/) | IoU/NMS/mAP by hand, YOLO11 fine-tuning, DETR set prediction | 15 min |
| 08 | [Segmentation](DLCV-08-Segmentation/) | U-Net from scratch, Dice loss, SAM prompting, auto-labelling | 20 min |
| 09 | [Diffusion Models](DLCV-09-Diffusion-Models/) | DDPM, DDIM, classifier-free guidance — from scratch | 30 min |
| 10 | [Stable Diffusion and LoRA](DLCV-10-Stable-Diffusion-and-LoRA/) | latent diffusion internals, LoRA fine-tuning, ControlNet | 20 min |
| 11 | [CLIP and VLMs](DLCV-11-CLIP-and-VLM/) | InfoNCE, zero-shot classification, **a working mini-LLaVA** | 25 min |

`archive/` holds the retired GAN lab, kept for reference.

---

## The through-line

The labs are ordered so that each one reuses machinery from an earlier one. This is
deliberate — point it out to students when you reach each junction.

```
Lab 02  backprop
   └─► Lab 03  convolution ──► Lab 04  residual connections
                                  │
                                  ├─► Lab 05  pretrained features ──► Lab 07 · 08 · 11
                                  │
                                  └─► Lab 08  U-Net ──────────────► Lab 09  diffusion backbone
Lab 06  self-attention
   ├─► Lab 07  DETR set prediction
   ├─► Lab 10  cross-attention conditioning
   └─► Lab 11  CLIP towers ──────────────────────────────────────► mini-LLaVA
```

Three specific hand-offs worth naming in class:

- The **U-Net** built in Lab 08 is the denoising backbone in Lab 09.
- The **classifier-free guidance** derived in Lab 09 is the guidance slider in Lab 10.
- The **CLIP text encoder** in Lab 11 is the thing conditioning the generator in Lab 10.

Students carry **one dataset of their own** from Lab 04 Exercise 5 through Labs 05, 07,
08 and 11. By the end they have classified, detected, segmented and captioned their own
data.

---

## Setup

### Colab (recommended)

Open any notebook via its badge, then **Runtime → Change runtime type → T4 GPU**.
Nothing else to install; each lab installs what it needs in its first cell.

### Local

```bash
conda create -n dlcv python=3.11 -y
conda activate dlcv
pip install -r requirements.txt
```

You need an NVIDIA GPU with 8 GB or more for Labs 09–11. Apple Silicon works via `mps`
for Labs 02–08 but is slow for the generative labs.

---

## Hardware notes for the T4

The free Colab GPU is a **T4 (Turing, compute capability 7.5)**. Three consequences that
every lab is written around, and that students hit constantly if unwarned:

1. **`torch.float16`, not `torch.bfloat16`.** bf16 requires Ampere (8.0+).
2. **No FlashAttention-2.** Use `attn_implementation="sdpa"`.
3. **16 GB, shared with the display.** Lab 10 uses SD 1.5 rather than SDXL for this
   reason, with attention slicing and gradient checkpointing enabled.

---

### Assessment

Every lab ends with graded exercises, ordered easy → open-ended. Exercises marked
*capstone* in Labs 10 and 11 are suitable as final-project seeds. The
"carry your own dataset" thread (Lab 04 Ex. 5 onward) works well as a running assignment
worth more than any single lab.

---

## References

### Course-level
- Justin Johnson — [EECS 498/598: Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/) (Michigan)
- [CS231n](https://cs231n.github.io/) — still the best notes on backprop and CNNs
- [Lilian Weng's blog](https://lilianweng.github.io/) — diffusion, contrastive learning
- HuggingFace [`transformers`](https://huggingface.co/docs/transformers) · [`diffusers`](https://huggingface.co/docs/diffusers) · [`peft`](https://huggingface.co/docs/peft)
- [`timm`](https://huggingface.co/docs/timm) — the model zoo used throughout

### Papers by lab

| Lab | Papers |
|---|---|
| 04 | [ResNet](https://arxiv.org/abs/1512.03385) · [SENet](https://arxiv.org/abs/1709.01507) · [ConvNeXt](https://arxiv.org/abs/2201.03545) |
| 05 | [Grad-CAM](https://arxiv.org/abs/1610.02391) · [Bag of Tricks](https://arxiv.org/abs/1812.01187) · [mixup](https://arxiv.org/abs/1710.09412) · [Calibration](https://arxiv.org/abs/1706.04599) · [DINOv2](https://arxiv.org/abs/2304.07193) · [MAE](https://arxiv.org/abs/2111.06377) |
| 06 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [ViT](https://arxiv.org/abs/2010.11929) |
| 07 | [YOLOv1](https://arxiv.org/abs/1506.02640) · [Focal Loss](https://arxiv.org/abs/1708.02002) · [DETR](https://arxiv.org/abs/2005.12872) · [RT-DETR](https://arxiv.org/abs/2304.08069) · [YOLOv10](https://arxiv.org/abs/2405.14458) |
| 08 | [U-Net](https://arxiv.org/abs/1505.04597) · [FCN](https://arxiv.org/abs/1411.4038) · [SAM](https://arxiv.org/abs/2304.02643) · [SAM 2](https://arxiv.org/abs/2408.00714) · [Mask2Former](https://arxiv.org/abs/2112.01527) |
| 09 | [DDPM](https://arxiv.org/abs/2006.11239) · [DDIM](https://arxiv.org/abs/2010.02502) · [CFG](https://arxiv.org/abs/2207.12598) · [Improved DDPM](https://arxiv.org/abs/2102.09672) |
| 10 | [Latent Diffusion](https://arxiv.org/abs/2112.10752) · [LoRA](https://arxiv.org/abs/2106.09685) · [DreamBooth](https://arxiv.org/abs/2208.12242) · [ControlNet](https://arxiv.org/abs/2302.05543) · [DiT](https://arxiv.org/abs/2212.09748) · [SD3](https://arxiv.org/abs/2403.03206) |
| 11 | [CLIP](https://arxiv.org/abs/2103.00020) · [SigLIP](https://arxiv.org/abs/2303.15343) · [LLaVA](https://arxiv.org/abs/2304.08485) · [LLaVA-1.5](https://arxiv.org/abs/2310.03744) · [BLIP-2](https://arxiv.org/abs/2301.12597) · [Flamingo](https://arxiv.org/abs/2204.14198) |

### Lineage

This course builds on materials by:
- Dr. Alisa Kunapinun — [Deep Learning for Computer Vision, Aug 2022](https://github.com/Alisa-Kunapinun/Deep-learning-for-Computer-Vision-Aug-2022)
- Prof. Matthew N. Dailey — [RTML](https://github.com/dsai-asia/RTML/)
- Coursera Deep Learning Specialization
- Learn by Example · [learnbyexample.org](https://www.learnbyexample.org/python-introduction/)
