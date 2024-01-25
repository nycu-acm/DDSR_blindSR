## Two Heads Better than One: Dual Degradation Representation for Blind Super-Resolution
## Abstract 
Previous methods have demonstrated remarkable performance in single image super-resolution (SISR) tasks with
known and fixed degradation (e.g., bicubic downsampling). However, when the actual degradation deviates from these
assumptions, these methods may experience significant declines in performance. In this paper, we propose a Dual
Branch Degradation Extractor Network to address the blind SR problem. While some BlindSR methods assume noise free
degradation and others do not explicitly consider the presence of noise in the degradation model, our approach
predicts two unsupervised degradation embeddings that represent blurry and noisy information, respectively. The
SR network can then be adapted to blur embedding and noise embedding in distinct ways. Furthermore, we treat the
degradation extractor as a regularizer to capitalize on the differences between SR and HR images. Extensive experiments on several benchmarks demonstrate that our method achieves SOTA performance in the blind SR problem.
