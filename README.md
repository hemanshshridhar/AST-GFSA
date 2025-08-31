



## AST-GFSA

This repository focuses on modifying  Audio spectrogram Transformer from <a href="https://arxiv.org/abs/2104.01778">AST</a> paper with Graph Filter based Self Attention (GFSA)  introduced in the Neurips paper <a href="https://arxiv.org/pdf/2312.04234">"Graph Convolutions Enrich the Self-Attention in
Transformers"</a>

## Usage

1 AST-GFSA paper

```python

B, T, FREQ = 8, 1024, 128
x = torch.randn(B, 1, T, FREQ)

model = AST_GFSA(
        max_length=T,
        num_classes=4,
        final_output="CLS",
        model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593",
        order_h=4,
        renormalize=False,   # set True if you want row-stochastic attention after GFSA
    )

```



## Todo

- [ ] experiment on some siganl processing focussed datasets.
- [ ] add results.
- [ ] Maybe add some fitler response curves?



## References
1. [jeongwhanchoi/GFSA](https://github.com/jeongwhanchoi/GFSA/blob/main/Image/gfsa.py)



## Citations

```bibtex
@inproceedings{choi2024gfsa,
   title={Graph Convolutions Enrich the Self-Attention in Transformers!},
   author={Jeongwhan Choi and Hyowon Wi and Jayoung Kim and Yehjin Shin and Kookjin Lee and Nathaniel Trask and Noseong Park},
   booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
   year={2024},
   url={https://openreview.net/forum?id=ffNrpcBpi6}
}

@inproceedings{gong21b_interspeech,
  author={Yuan Gong and Yu-An Chung and James Glass},
  title={{AST: Audio Spectrogram Transformer}},
  year=2021,
  booktitle={Proc. Interspeech 2021},
  pages={571--575},
  doi={10.21437/Interspeech.2021-698}
}

```
