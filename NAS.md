# NAS Papers&Projects

## Table of Contents

- [Awesome Blogs](#awesome-blogs)

- [Neural Architecture Search](#NAS)
  - [2020 Venues](#2020)
  - [2019 Venues](#2019)
  - [2018 Venues](#2018)
  - [2017 Venues](#2017)
  - [Previous Venues](#2012-2016)
  - [arXiv](#arxiv)

## Awesome Blogs
- [AutoML info](http://automl.chalearn.org/) and [AutoML Freiburg-Hannover](https://www.automl.org/)
- [What’s the deal with Neural Architecture Search?](https://determined.ai/blog/neural-architecture-search/)
- [Google Could AutoML](https://cloud.google.com/vision/automl/docs/beginners-guide) and [PocketFlow](https://pocketflow.github.io/)
- [AutoML Challenge](http://automl.chalearn.org/) and [AutoDL Challenge](https://autodl.chalearn.org/)


## Neural Architecture Search (NAS)

|      Type   |        G       |                  RL    |            EA           |        PD              |    Other   |
|:------------|:--------------:|:----------------------:|:-----------------------:|:----------------------:|:----------:|
| Explanation | gradient-based | reinforcement learning | evaluationary algorithm | performance prediction | other types |

### arXiv
|  Title  |   Date  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [NAS-Bench-1Shot1: Benchmarking and Dissecting One-shot Neural Architecture Search](https://arxiv.org/abs/2001.10422) | 2020.1 | - | - |
| [Learning Architectures for Binary Networks](https://arxiv.org/abs/2002.06963) | 2020.2 |G | - | 
| [Binarized Neural Architecture Search](https://arxiv.org/abs/1911.10862)| 2019.11 | G | - |
| [Neural Architecture Search on Acoustic Scene Classificatio](http://arxiv.org/abs/1912.12825) | 2019.12 | EA | - |
| [Performance-Oriented Neural Architecture Search](https://arxiv.org/abs/2001.02976)| 2020.1 | - | - |
| [RC-DARTS: Resource Constrained Differentiable Architecture Search](https://arxiv.org/abs/1912.12814)| 2019.12 | G | - |
| [Best of Both Worlds: AutoML Codesign of a CNN and its Hardware Accelerator](https://arxiv.org/abs/2002.05022)| 2020.2 | RL | - |
| [Neural Architecture Search for Deep Image Prior](https://arxiv.org/abs/2001.04776) | 2020.1 | EA | - |
| [SM-NAS: Structural-to-Modular Neural Architecture Search for Object Detection](https://arxiv.org/abs/1911.09929) | 2019.11 | EA | - |
| [Search to Distill: Pearls are Everywhere but not the Eyes](https://arxiv.org/abs/1911.09074)| 2019.11 | RL | - |
| [SGAS: Sequential Greedy Architecture Search](https://arxiv.org/abs/1912.00195v1) | 2019.12 | G | - |
| [EDAS: Efficient and Differentiable Architecture Search](https://arxiv.org/abs/1912.01237)| 2019.12 | G | - |
| [Multi-objective Neural Architecture Search via Non-stationary Policy Gradient](https://arxiv.org/abs/2001.08437) | 2020.1 | RL | - |
| [Channel Pruning via Automatic Structure Search](https://arxiv.org/abs/2001.08565) | 2020.1 | - | - |
| [DEEPER INSIGHTS INTO WEIGHT SHARING IN NEU- RAL ARCHITECTURE SEARCH](https://arxiv.org/abs/2001.01431)| 2020.1 | - | - |
| [MixPath: A Unified Approach for One-shot Neural Architecture Search](https://arxiv.org/abs/2001.05887) | 2020.1 | EA | [Github](https://github.com/xiaomi-automl/MixPath.git) | 
| [Efficient Neural Architecture Search: A Broad Version](https://arxiv.org/abs/2001.06679) | 2020.1 | RL | - |
| [Latency-Aware Differentiable Neural Architecture Search](https://arxiv.org/abs/2001.06392) | 2020.1 | G | - |
| [EcoNAS: Finding Proxies for Economical Neural Architecture Search](https://arxiv.org/abs/2001.01233) | 2020.1| EA | - | 
| [Stabilizing Differentiable Architecture Search via Perturbation-based Regularization](https://arxiv.org/abs/2002.05283) | 2020.2 | G | [Github](https://github.com/xiangning-chen/SmoothDARTS) |
| [Scalable NAS with Factorizable Architectural Parameters](https://arxiv.org/abs/1912.13256) | 2019.12 | G | - |
| [DARTS+: Improved Differentiable Architecture Search with Early Stopping](https://arxiv.org/abs/1909.06035v1) | 2019.9 | G | - |
| [Stabilizing Darts With Amended Gradient Estimation On Architectural Parameters](https://arxiv.org/abs/1910.11831) | 2019.10 | G | [Code](https://www.dropbox.com/sh/j4rfzi6586iw3me/AAB1bnUMid-5DLzaEGxmQAkCa?dl=0) |
| [StacNAS: Towards stable and consistent optimization for differentiable Neural Architecture Search](https://128.84.21.199/abs/1909.11926v3) | 2019.11 | G | [Github](https://github.com/susan0199/stacnas) |
| [Fair DARTS: Eliminating Unfair Advantages in Differentiable Architecture Search](https://arxiv.org/abs/1911.12126) | 2019.11 | G | [Github](https://github.com/xiaomi-automl/FairDARTS) |
| [FairNAS: Rethinking Evaluation Fairness of Weight Sharing Neural Architecture Search](https://arxiv.org/abs/1907.01845) | 2019.11 | EA | [Github](https://github.com/xiaomi-automl/FairNAS) |
| [SCARLET-NAS: Bridging the gap Between Stability and Scalability in Neural Architecture Search](https://arxiv.org/abs/1908.06022) | 2019.8 | EA | [Github](https://github.com/xiaomi-automl/SCARLET-NAS) |
| [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) | 2017.11 | EA | - |
| [NSGA-NET: A Multi-Objective Genetic Algorithm for Neural Architecture Search](https://arxiv.org/pdf/1810.03522.pdf) | 2018.10 | EA | - |
| [Training Frankenstein’s Creature to Stack: HyperTree Architecture Search](https://arxiv.org/pdf/1810.11714.pdf) | 2018.10 | G | - |
| [Fast, Accurate and Lightweight Super-Resolution with Neural Architecture Search](https://arxiv.org/pdf/1901.07261.pdf) | 2019.01 | G | [Github](https://github.com/falsr/FALSR) |


### 2020

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [AtomNAS: Fine-Grained End-to-End Neural Architecture Search](https://openreview.net/forum?id=BylQSxHFwr) | ICLR | - | [Github](https://github.com/meijieru/AtomNAS) |
| [Understanding and Robustifying Differentiable Architecture Search](https://openreview.net/forum?id=H1gDNyrKDS) | ICLR | G | [Github](https://github.com/automl/RobustDARTS) |
| [NAS-BENCH-201: Extending the Scope of Reproducible Neural Architecture Search](https://openreview.net/forum?id=HJxyZkBKDr) | ICLR | -  | [Github](https://github.com/D-X-Y/NAS-Projects/blob/master/NAS-Bench-102.md) |
| [Understanding Architectures Learnt by Cell-based Neural Architecture Search](https://openreview.net/pdf?id=H1gDNyrKDS) | ICLR | G | [Github](https://github.com/automl/RobustDARTS) |
| [Evaluating The Search Phase of Neural Architecture Search](https://openreview.net/forum?id=H1loF2NFwr) | ICLR | - | - |
| [Fast Neural Network Adaptation via Parameter Remapping and Architecture Search](https://openreview.net/forum?id=rklTmyBKPH) | ICLR | _ | [Github](https://github.com/JaminFong/FNA) |
| [Once for All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791) | ICLR | G | - |
| [Efficient Transformer for Mobile Applications](https://openreview.net/pdf?id=ByeMPlHKPH) | ICLR | - | - |
| [PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search](https://arxiv.org/abs/1907.05737.pdf) | ICLR | G | [Github](https://github.com/yuhuixu1993/PC-DARTS) |
| [Adversarial AutoAugment](https://openreview.net/pdf?id=ByxdUySKvS) | ICLR | - | - |
| [NAS evaluation is frustratingly hard](https://openreview.net/pdf?id=HygrdpVKvr) | ICLR | - | - |
| [FasterSeg: Searching for Faster Real-time Semantic Segmentation](https://openreview.net/pdf?id=BJgqQ6NYvB) | ICLR | G | [Github](https://github.com/TAMU-VITA/FasterSeg) |
| [Computation Reallocation for Object Detection](https://openreview.net/forum?id=SkxLFaNKwB) | ICLR | - | - | 
| [Towards Fast Adaptation of Neural Architectures with Meta Learning](https://openreview.net/forum?id=r1eowANFvr) | ICLR | - | - |
| [AssembleNet: Searching for Multi-Stream Neural Connectivity in Video Architectures](https://arxiv.org/abs/1905.13209.pdf) | ICLR | EA | - |
| [How to Own the NAS in Your Spare Time](https://openreview.net/pdf?id=S1erpeBFPB) | ICLR | - | - |

### 2019

|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [DATA: Differentiable ArchiTecture Approximation](http://papers.nips.cc/paper/8374-data-differentiable-architecture-approximation) | NeurIPS | G | - |
| [Random Search and Reproducibility for Neural Architecture Search]() | UAI | G | [Github](https://github.com/D-X-Y/NAS-Projects/blob/master/scripts-search/algos/RANDOM-NAS.sh) |
| [Improved Differentiable Architecture Search for Language Modeling and Named Entity Recognition](https://www.aclweb.org/anthology/D19-1367.pdf/) | EMNLP | - |
| [Continual and Multi-Task Architecture Search](https://www.aclweb.org/anthology/P19-1185.pdf) | ACL | RL | - |
| [Progressive Differentiable Architecture Search: Bridging the Depth Gap Between Search and Evaluation](https://arxiv.org/abs/1904.12760.pdf) | ICCV | - | - |
| [Multinomial Distribution Learning for Effective Neural Architecture Search](https://arxiv.org/abs/1905.07529.pdf) | ICCV | - | - |
| [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244.pdf) | ICCV | EA | - |
| [Fast and Practical Neural Architecture Search](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cui_Fast_and_Practical_Neural_Architecture_Search_ICCV_2019_paper.pdf) | ICCV | - | - |
| [Teacher Guided Architecture Search](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bashivan_Teacher_Guided_Architecture_Search_ICCV_2019_paper.pdf) | ICCV | - | - |
| [AutoDispNet: Improving Disparity Estimation With AutoML](http://openaccess.thecvf.com/content_ICCV_2019/papers/Saikia_AutoDispNet_Improving_Disparity_Estimation_With_AutoML_ICCV_2019_paper.pdf) | ICCV | G | - |
| [Resource Constrained Neural Network Architecture Search: Will a Submodularity Assumption Help?](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xiong_Resource_Constrained_Neural_Network_Architecture_Search_Will_a_Submodularity_Assumption_ICCV_2019_paper.pdf) | ICCV | EA | - |
| [Towards modular and programmable architecture search](https://arxiv.org/abs/1909.13404) | NeurIPS | [Other](https://github.com/D-X-Y/Awesome-NAS/issues/10) | [Github](https://github.com/negrinho/deep_architect) |
| [Network Pruning via Transformable Architecture Search](https://arxiv.org/abs/1905.09717) | NeurIPS | G | [Github](https://github.com/D-X-Y/NAS-Projects) |
| [Deep Active Learning with a NeuralArchitecture Search](https://arxiv.org/pdf/1811.07579.pdf) | NeurIPS | - | - |
| [DetNAS: Backbone Search for ObjectDetection](https://arxiv.org/abs/1903.10979) | NeurIPS | - | [Github](https://github.com/megvii-model/ShuffleNet-Series/tree/master/DetNAS) |
| [SpArSe: Sparse Architecture Search for CNNs on Resource-Constrained Microcontrollers](https://arxiv.org/abs/1905.12107.pdf) | NeurIPS | - | - |
| [Efficient Forward Architecture Search ](https://arxiv.org/abs/1905.13360) | NeurIPS | G | [Github](https://github.com/microsoft/petridishnn) |
| [Efficient Neural ArchitectureTransformation Search in Channel-Level for Object Detection](https://arxiv.org/abs/1909.02293.pdf) | NeurIPS | G | - |
| [XNAS: Neural Architecture Search with Expert Advice](https://arxiv.org/abs/1906.08031.pdf) | NeurIPS | G | - |
| [One-Shot Neural Architecture Search via Self-Evaluated Template Network](https://arxiv.org/abs/1910.05733) | ICCV | G/PD | [Github](https://github.com/D-X-Y/NAS-Projects) |
| [Evolving Space-Time Neural Architectures for Videos](https://arxiv.org/abs/1811.10636) | ICCV | EA | [GitHub](https://sites.google.com/view/evanet-video) |
| [AutoGAN: Neural Architecture Search for Generative Adversarial Networks](https://arxiv.org/pdf/1908.03835.pdf) | ICCV | RL | [Github](https://github.com/TAMU-VITA/AutoGAN) |
| [Neural architecture search: A survey](http://www.jmlr.org/papers/volume20/18-598/18-598.pdf) | JMLR | Survey | - |
| [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) | ICLR | G | [Github](https://github.com/quark0/darts) |
| [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://openreview.net/pdf?id=HylVB3AqYm) | ICLR | RL/G | [Github](https://github.com/MIT-HAN-LAB/ProxylessNAS) |
| [Graph HyperNetworks for Neural Architecture Search](https://arxiv.org/pdf/1810.05749.pdf) | ICLR | G | - |
| [Learnable Embedding Space for Efficient Neural Architecture Compression](https://openreview.net/forum?id=S1xLN3C9YX) | ICLR | Other | [Github](https://github.com/Friedrich1006/ESNAC) |
| [Efficient Multi-Objective Neural Architecture Search via Lamarckian Evolution](https://arxiv.org/abs/1804.09081) | ICLR | EA | - |
| [SNAS: stochastic neural architecture search](https://openreview.net/pdf?id=rylqooRqK7) | ICLR | G | - |
| [Searching for A Robust Neural Architecture in Four GPU Hours](http://xuanyidong.com/publication/gradient-based-diff-sampler/) | CVPR | G | [Github](https://github.com/D-X-Y/NAS-Projects) |
| [ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_ChamNet_Towards_Efficient_Network_Design_Through_Platform-Aware_Model_Adaptation_CVPR_2019_paper.pdf) | CVPR | EA/PD | - |
| [Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search](https://arxiv.org/pdf/1903.03777.pdf) | CVPR | EA | [Github](https://github.com/lixincn2015/Partial-Order-Pruning) |
| [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443) | CVPR | G | - | 
| [RENAS: Reinforced Evolutionary Neural Architecture Search	](https://arxiv.org/abs/1808.00193) | CVPR | EA | - |
| [Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation](https://arxiv.org/pdf/1901.02985.pdf) | CVPR |  G | [GitHub](https://github.com/tensorflow/models/tree/master/research/deeplab) |
| [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626) | CVPR | RL | [Github](https://github.com/AnjieZheng/MnasNet-PyTorch) |
| [MFAS: Multimodal Fusion Architecture Search](https://arxiv.org/pdf/1903.06496.pdf) | CVPR | EA | - |
| [A Neurobiological Evaluation Metric for Neural Network Model Search](https://arxiv.org/pdf/1805.10726.pdf) | CVPR | Other | - |
| [Fast Neural Architecture Search of Compact Semantic Segmentation Models via Auxiliary Cells](https://arxiv.org/abs/1810.10804) | CVPR | RL | - |
| [Customizable Architecture Search for Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Customizable_Architecture_Search_for_Semantic_Segmentation_CVPR_2019_paper.pdf) | CVPR | - | - |
| [Regularized Evolution for Image Classifier Architecture Search](https://arxiv.org/pdf/1802.01548.pdf) | AAAI | EA | - |
| [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501.pdf) | CVPR | RL | - |
| [Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules](https://arxiv.org/abs/1905.05393) | ICML | EA | - |
| [The Evolved Transformer](https://arxiv.org/pdf/1901.11117.pdf) | ICML | EA | [Github](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/evolved_transformer.py) |
| [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946.pdf) | ICML | RL | - |
| [NAS-Bench-101: Towards Reproducible Neural Architecture Search](https://arxiv.org/abs/1902.09635) | ICML | Other | [Github](https://github.com/google-research/nasbench) | 

### 2018
|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Towards Automatically-Tuned Deep Neural Networks](https://ml.informatik.uni-freiburg.de/papers/16-AUTOML-AutoNet.pdf) | BOOK | - | [GitHub](https://github.com/automl/Auto-PyTorch) |
| [Efficient Architecture Search by Network Transformation](https://arxiv.org/pdf/1707.04873.pdf) | AAAI | RL | [Github](https://github.com/han-cai/EAS) |
| [Learning Transferable Architectures for Scalable Image Recognition](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zoph_Learning_Transferable_Architectures_CVPR_2018_paper.pdf) | CVPR | RL | [Github](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) |
| [N2N learning: Network to Network Compression via Policy Gradient Reinforcement Learning](https://openreview.net/forum?id=B1hcZZ-AW) | ICLR | RL | - |
| [A Flexible Approach to Automated RNN Architecture Generation](https://openreview.net/forum?id=SkOb1Fl0Z) | ICLR | RL/PD | - |
| [Practical Block-wise Neural Network Architecture Generation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhong_Practical_Block-Wise_Neural_CVPR_2018_paper.pdf) | CVPR | RL | - | [Efficient Neural Architecture Search via Parameter Sharing](http://proceedings.mlr.press/v80/pham18a.html) | ICML | RL | [Github](https://github.com/melodyguan/enas) |
| [Path-Level Network Transformation for Efficient Architecture Search](https://arxiv.org/abs/1806.02639) | ICML | RL | [Github](https://github.com/han-cai/PathLevel-EAS) |
| [Hierarchical Representations for Efficient Architecture Search](https://openreview.net/forum?id=BJQRKzbA-) | ICLR | EA | - |
| [Understanding and Simplifying One-Shot Architecture Search](http://proceedings.mlr.press/v80/bender18a/bender18a.pdf) | ICML | G | - |
| [SMASH: One-Shot Model Architecture Search through HyperNetworks](https://arxiv.org/pdf/1708.05344.pdf) | ICLR | G | [Github](https://github.com/ajbrock/SMASH) |
| [Neural Architecture Optimization](https://arxiv.org/pdf/1808.07233.pdf) | NeurIPS | G | [Github](https://github.com/renqianluo/NAO) |
| [Searching for efficient multi-scale architectures for dense image prediction](https://papers.nips.cc/paper/8087-searching-for-efficient-multi-scale-architectures-for-dense-image-prediction.pdf) | NeurIPS | Other | - |
| [Progressive Neural Architecture Search](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chenxi_Liu_Progressive_Neural_Architecture_ECCV_2018_paper.pdf) | ECCV | PD | [Github](https://github.com/chenxi116/PNASNet) |
| [Neural Architecture Search with Bayesian Optimisation and Optimal Transport](https://arxiv.org/pdf/1802.07191.pdf) | NeurIPS | Other | [Github](https://github.com/kirthevasank/nasbot) |
| [Differentiable Neural Network Architecture Search](https://openreview.net/pdf?id=BJ-MRKkwG) | ICLR-W | G | - |
| [Accelerating Neural Architecture Search using Performance Prediction](https://arxiv.org/abs/1705.10823) | ICLR-W | PD | - |



### 2017
|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) | ICLR | RL | - |
| [Designing Neural Network Architectures using Reinforcement Learning](https://openreview.net/pdf?id=S1c2cvqee) | ICLR | RL | - | [Github](https://github.com/bowenbaker/metaqnn) |
| [Neural Optimizer Search with Reinforcement Learning](http://proceedings.mlr.press/v70/bello17a/bello17a.pdf) | ICML | RL | - | [Large-Scale Evolution of Image Classifiers](https://arxiv.org/pdf/1703.01041.pdf) | ICML | EA | - |
| [Learning Curve Prediction with Bayesian Neural Networks](http://ml.informatik.uni-freiburg.de/papers/17-ICLR-LCNet.pdf) | ICLR | PD | - |
| [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) | ICLR | PD | - |
| [Hyperparameter Optimization: A Spectral Approach](https://arxiv.org/abs/1706.00764) | NeurIPS-W | Other | [Github](https://github.com/callowbird/Harmonica) |
| [Learning to Compose Domain-Specific Transformations for Data Augmentation](https://arxiv.org/abs/1709.01643.pdf) | NeurIPS | - | - |

### 2012-2016
|  Title  |   Venue  |   Type   |   Code   |
|:--------|:--------:|:--------:|:--------:|
| [Speeding up Automatic Hyperparameter Optimization of Deep Neural Networksby Extrapolation of Learning Curves](http://ml.informatik.uni-freiburg.de/papers/15-IJCAI-Extrapolation_of_Learning_Curves.pdf) | IJCAI | PD | [Github](https://github.com/automl/pylearningcurvepredictor) |
