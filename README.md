# DAtools

Python library for processing Kik-Net ground motions and interacting with Albert Kottke's pyStrata package [pystrata](https://github.com/arkottke/pystrata).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![YouTube](hhttps://www.youtube.com/playlist?list=PLDJOjJHvPIGwjSnY3X_1JczhZWMNXdyZa)


## Introduction

Allows for easy retrieval, processing, and implementation of Kik-Net downhole array ground motions in Albert Kottke's pystrata.

The library is composed of two main components:
-   Ground Motion Processing:
    -   detrends/demeans
    -   trims motion (w/ help of STA/LTA picker)
    -   estimates frequency filter frequencies (SNR ≤ 3 and [UCLA high-pass frequency calculation](https://ascelibrary.org/doi/10.1061/9780784484692.034) (Ramos-Sepulveda 2023)
    -   buttersworth bandpass
    -   kappa calculations

-   pystrata helper functions:
    -   analyze multiple sites efficiently
    -   incorporate strength correction
    -   Dmin scale factor implementation
        - kappa scale factors
        - ground motion parameter scale factors



## Installation
`datools` is not available yet on PyPI; installation will need to be done instead with:
```bash
pip install https://github.com/GBerse/datools