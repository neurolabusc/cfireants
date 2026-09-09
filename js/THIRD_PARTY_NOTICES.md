# Third-party notices

## FireANTs

This package is a **derivative work of FireANTs** and is distributed under the
FireANTs License, Version 1.0 (July 2025), a copy of which travels with it as
`LICENSE`. FireANTs is <https://github.com/rohitrango/FireANTs>, by Rohit Jena
and collaborators at the University of Pennsylvania.

**Files have been changed.** The wasm module this package loads is compiled from
`cfireants`, a pure C reimplementation of FireANTs (upstream commit `0d13a3f`)
written for CPU, CUDA, Metal and WebGPU. None of the original Python source is
carried over; the algorithms, the multi-scale schedule, the WarpAdam optimizer,
the CC and MI metrics and the numerical conventions are ports, verified against
the Python reference. It is a fork in the sense of section 4(a)(iii) of the
License and it is not endorsed by the FireANTs authors.

Section 4 of that License requires that redistribution retain the original
copyright notices, the license text, and **the bibliography references and
research papers cited in the project's documentation**. That bibliography
follows. The upstream repository's documentation is authoritative; if it cites
work not listed here, that omission is a bug in this file and not a waiver.

```bibtex
@article{jena2024fireants,
  title   = {FireANTs: Adaptive Riemannian Optimization for Multi-Scale
             Diffeomorphic Registration},
  author  = {Jena, Rohit and Chaudhari, Pratik and Gee, James C.},
  journal = {arXiv preprint arXiv:2404.01249},
  year    = {2024},
  url     = {https://arxiv.org/abs/2404.01249}
}
```

FireANTs itself builds on the ANTs/SyN registration literature, which its
documentation cites:

```bibtex
@article{avants2008symmetric,
  title   = {Symmetric diffeomorphic image registration with cross-correlation:
             Evaluating automated labeling of elderly and neurodegenerative brain},
  author  = {Avants, Brian B. and Epstein, Charles L. and Grossman, Murray and
             Gee, James C.},
  journal = {Medical Image Analysis},
  volume  = {12},
  number  = {1},
  pages   = {26--41},
  year    = {2008}
}
```

## NIfTI I/O

The module's `nifti_io.c`/`nifti_io.h` are the separately managed public-domain
NIfTI I/O consolidation based on work by Robert W. Cox, Mark Jenkinson, Rick
Reynolds and Chris Rorden.
