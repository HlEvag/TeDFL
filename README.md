# TeDFL: Towards Text-Driven Few-Shot Learning for Forest Tree Species Classification with Airborne Hyperspectral Images
The code of the proposed TeDFL method will be publicly available here. 

# Abstract
Tree species classification is fundamental for forest inventory and sustainable silviculture. Airborne hyperspectral images can capture subtle spectral differences among tree species through dense spectral channels, aiding the discrimination of similar species. However, in multi-species field scenarios, mixed pixels and environmental interference exacerbate inter-class spectral similarity and intra-class spectral variation, making it difficult to accurately distinguish tree species solely through visual representation. To this end, a text-driven few-shot learning (TeDFL) paradigm is developed that can leverage linguistic prior knowledge to guide the generation of discriminative visual features. \hl{The aim of the TeDFL method is to train a highly flexible and robust model that can adapt to classifying diverse yet similar tree species data based on only a few examples.} \hl{Our TeDFL framework contains two primary encoders}: a visual encoder and a text encoder. The former exploits a ghost attention network (GANet) to explore spatial-spectral visual features without excessive parameters, where GANet consists of ghost and normalization-based attention blocks. The latter leverages the contrastive language-image pretraining model to incorporate additional rich semantic priors, enriching the visual feature space. Furthermore, we design cross-modal alignment (CMA) and cross-modal integration (CMI) strategies to bridge the modal discrepancy between visual and textual representations. The CMA strategy aims to facilitate a visual-text contrastive objective by contrasting hyperspectral images with class-specific text prompts, while the CMI strategy adaptively fuses \hl{multimodal} support data to obtain more reliable prototypes. \hl{Experiments on four realistic forest scenarios confirm the feasibility and effectiveness of the TeDFL method compared with other state-of-the-art methods.} Moreover, \hl{TeDFL method} could serve as a standard visual-language backbone network for few-shot tree species classification studies. The source code will be publicly accessible at https://github.com/HlEvag/TeDFL.git

# References
Some of our reference projects are listed below, and we are very grateful for their research:
* Gia-CFSL: [Graph Information Aggregation Cross-Domain Few-Shot Learning for Hyperspectral Image Classification, TNNLS, 2022.](https://github.com/YuxiangZhang-BIT/IEEE_TNNLS_Gia-CFSL)
* DCFSL: [Deep Cross-domain Few-shot Learning for Hyperspectral Image Classification, TGRS, 2022.](https://github.com/Li-ZK/DCFSL-2021)
* CMFSL: [Few-shot Learning with Class-Covariance Metric for Hyperspectral Image Classification, TIP, 2022.](https://github.com/B-Xi/TIP_2022_CMFSL)
* HFSL: [Heterogeneous Few-shot Learning for Hyperspectral Image Classification, TGRSL, 2022](https://github.com/Li-ZK/HFSL)
* RCAPDA: [Few-shot learning using residual channel attention and prototype domain adaptation for hyperspectral image classification, TGRSL, 2023](https://github.com/XtaoS/protypical-fsl)
* ADAFSL: [Adaptive Domain-Adversarial Few-Shot Learning for Cross-Domain Hyperspectral Image Classification, TGRS, 2023](https://github.com/JieW-ww/ADAFSL)
* STBDIP: [Multi-level relation learning for cross-domain few-shot hyperspectral image classification, Applied intelligence, 2024](https://github.com/HENULWY/STBDIP)
# Requirements
CUDA Version = 11.7

Python = 3.9.7

Pytorch = 1.12.0

Sklearn = 0.24.2

Numpy = 1.20.0

Matplotlib = 3.4.3

Spectral = 0.22.4

# Dataset
---Source domain data set: Chikusei [1]:

The source domain hyperspectral datasets (Chikusei) in mat format is available in: https://naotoyokoya.com/Download.html or https://github.com/Li-ZK/DCFSL-2021

You can also download our preprocessed source domain data set (Chikusei_imdb_128.pickle) directly in pickle format, please move to the ./Datasets folder. However, the data is too large to upload, if necessary, please contact my email: hulei.eva@whu.edu.cn.

---Target domain data set: GFF-A/GFF-B/Tea farm/Xiong An

GFF-A [2]: data will be made available on request.

GFF-B [3]: https://github.com/longbaochenlong/SCL-P-Net.

Tea farm [4]: https://www.geodoi.ac.cn/WebCn/doi.aspx?Id=720.

Xiong An [5]: https://www.geodoi.ac.cn/doi.aspx?Id=1687.

The specific details of the original data can be referred to the following literature:

[1] Yokoya N, Iwasaki A. Airborne hyperspectral data over Chikusei[J]. Space Appl. Lab., Univ. Tokyo, Tokyo, Japan, Tech. Rep. SAL-2016-05-27, 2016, 5(5): 5.

[2] Tian X, Chen L, Zhang X, et al. Improved prototypical network model for forest species classification in complex stand[J]. Remote Sensing, 2020, 12(22): 3839.

[3] Zhang B, Zhao L, Zhang X. Three-dimensional convolutional neural network model for tree species classification using airborne hyperspectral images[J]. Remote Sensing of Environment, 2020, 247: 111938.

[4] Zhang X, Zhang B, Zhang L F, et al. Hyperspectral remote sensing dataset for tea farm[J]. Digit. J. Global Change Data Repository, 2017.

[5] Yi C E N, Zhang L, Zhang X, et al. Aerial hyperspectral remote sensing classification dataset of xiongan new area (matiwan village)[J]. National Remote Sensing Bulletin, 2020, 24(11): 1299-1306.

# Licensing
Copyright (C) 2025 Lei Hu, Wei He

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
