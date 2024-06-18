# Two Heads Better than One: Dual Degradation Representation for Blind Super-Resolution #

### | [preprint](https://www.techrxiv.org/707966/tvwm_bTzDzrNrE6mQHfGxg) | [bibtex](/doc/bibtex.bib) |

![Method](/doc/image/network_v1.png)

The pytorch implementation for "Two Heads Better than One: Dual Degradation Representation for Blind Super-Resolution".

Abstract *Previous methods have demonstrated remarkable performance in single image super-resolution (SISR) tasks with
known and fixed degradation (e.g., bicubic downsampling). However, when the actual degradation deviates from these
assumptions, these methods may experience significant declines in performance. In this paper, we propose a Dual
Branch Degradation Extractor Network to address the blind SR problem. While some BlindSR methods assume noise free
degradation and others do not explicitly consider the presence of noise in the degradation model, our approach
predicts two unsupervised degradation embeddings that represent blurry and noisy information, respectively. The
SR network can then be adapted to blur embedding and noise embedding in distinct ways. Furthermore, we treat the
degradation extractor as a regularizer to capitalize on the differences between SR and HR images. Extensive experiments on several benchmarks demonstrate that our method achieves SOTA performance in the blind SR problem.*

## Setup

This repo was tested on linux-64 platform.

``` bash
conda create --name ddsr --file requirements.txt
conda activate ddsr
```

## Acknowledgments

This work was financially supported in part (project number: 112UA10019) by the Co-creation Platform of the Industry Academia Innovation School, NYCU, under the framework of the National Key Fields Industry-University Cooperation and Skilled Personnel Training Act, from the Ministry of Education (MOE) and industry partners in Taiwan, and in part by MediaTek Inc. It also supported in part by the National Science and Technology Council, Taiwan, under Grant NSTC-112-2221-E-A49-089-MY3, Grant NSTC-110-2221-E-A49-066-MY3, Grant NSTC-111- 2634-F-A49-010, Grant NSTC-112-2425-H-A49-001-, and in part by the Higher Education Sprout Project of the National Yang Ming Chiao Tung University and the Ministry of Education (MOE), Taiwan.

## Citation

```
@article{ddsr,
title={Two Heads Better than One: Dual Degradation Representation for Blind Super-Resolution},
url={http://dx.doi.org/10.36227/techrxiv.170723395.59369240/v1},
publisher={Institute of Electrical and Electronics Engineers (IEEE)},
author={Yuan, Hsuan and Weng, Shao-Yu and Lo, I-Hsuan and Chiu, Wei-Chen and Xu, Yu-Syuan and Hsueh, Hao-Chien and Chuang, Jen-Hui and Huang, Ching-Chun},
year={2024},
month=feb 
}
```
<!-- 
© 2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works. -->