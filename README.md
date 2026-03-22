# Awesome Neural Network Interpretability [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A comprehensive collection of research papers, tools, and resources on **Neural Network Interpretability** — covering self-interpretable architectures, mechanistic interpretability, representation analysis, causal methods, safety applications, and more.

Last updated: **March 2026**

---

## Table of Contents

- [Surveys](#surveys)
- [Self-Interpretable Methods](#self-interpretable-methods)
  - [Attribution-Based](#attribution-based)
  - [Function-Based](#function-based)
  - [Concept-Based](#concept-based)
  - [Prototype-Based](#prototype-based)
  - [Rule-Based](#rule-based)
- [Mechanistic Interpretability](#mechanistic-interpretability)
  - [Circuit Discovery](#circuit-discovery)
  - [Sparse Autoencoders](#sparse-autoencoders)
  - [Superposition and Features](#superposition-and-features)
- [Representation Analysis and Probing](#representation-analysis-and-probing)
  - [Probing Classifiers](#probing-classifiers)
  - [Logit Lens and Tuned Lens](#logit-lens-and-tuned-lens)
  - [Representation Engineering and Control Vectors](#representation-engineering-and-control-vectors)
  - [Linear Representation Hypothesis](#linear-representation-hypothesis)
  - [Representation Similarity](#representation-similarity)
  - [Concept Erasure](#concept-erasure)
- [Causal Interpretability](#causal-interpretability)
  - [Causal Abstraction](#causal-abstraction)
  - [Activation Patching and Causal Tracing](#activation-patching-and-causal-tracing)
  - [Path Patching and Automated Circuit Discovery](#path-patching-and-automated-circuit-discovery)
- [Feature Visualization](#feature-visualization)
- [Disentangled Representations](#disentangled-representations)
- [Kolmogorov-Arnold Networks (KAN)](#kolmogorov-arnold-networks-kan)
- [Neuro-Symbolic AI](#neuro-symbolic-ai)
  - [Frameworks and Languages](#frameworks-and-languages)
  - [Program Synthesis and Library Learning](#program-synthesis-and-library-learning)
  - [Symbolic Regression and Scientific Discovery](#symbolic-regression-and-scientific-discovery)
  - [LLM + Symbolic Reasoning](#llm--symbolic-reasoning)
- [Foundation Model Interpretability](#foundation-model-interpretability)
  - [LLM Internals](#llm-internals)
  - [Diffusion Model Interpretability](#diffusion-model-interpretability)
  - [Vision-Language Models](#vision-language-models)
  - [Multimodal Interpretability](#multimodal-interpretability)
- [Safety and Alignment](#safety-and-alignment)
  - [Deception Detection](#deception-detection)
  - [Truthfulness Representations](#truthfulness-representations)
  - [Steering Vectors and Activation Control](#steering-vectors-and-activation-control)
  - [Jailbreak Analysis](#jailbreak-analysis)
- [Training for Interpretability](#training-for-interpretability)
  - [Sparsity and Regularization](#sparsity-and-regularization)
  - [Modular Networks](#modular-networks)
  - [Knowledge Distillation for Interpretability](#knowledge-distillation-for-interpretability)
- [Adversarial Robustness and Interpretability](#adversarial-robustness-and-interpretability)
- [Scaling Laws for Interpretability](#scaling-laws-for-interpretability)
- [Interpretable Transformers](#interpretable-transformers)
- [Applications](#applications)
  - [Vision](#vision)
  - [NLP and LLMs](#nlp-and-llms)
  - [Graph Data](#graph-data)
  - [Science and Medicine](#science-and-medicine)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Time Series](#time-series)
- [Evaluation and Benchmarks](#evaluation-and-benchmarks)
- [Tools and Frameworks](#tools-and-frameworks)
  - [Mechanistic Interpretability Tools](#mechanistic-interpretability-tools)
  - [General Interpretability Libraries](#general-interpretability-libraries)
  - [Model Implementations](#model-implementations)
- [Datasets](#datasets)
- [Tutorials and Workshops](#tutorials-and-workshops)
- [Related Awesome Lists](#related-awesome-lists)

---

## Surveys

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **A Comprehensive Survey on Self-Interpretable Neural Networks** | Proceedings of the IEEE | [arXiv](https://arxiv.org/abs/2501.15638) / [GitHub](https://github.com/yangji721/Awesome-Self-Interpretable-Neural-Network) |
| 2025 | **Ante-Hoc Methods for Interpretable Deep Models: A Survey** | ACM Computing Surveys | [Paper](https://dl.acm.org/doi/10.1145/3728637) |
| 2025 | **A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of LLMs** | EMNLP Findings | [arXiv](https://arxiv.org/abs/2503.05613) |
| 2025 | **Open Problems in Mechanistic Interpretability** | arXiv | [arXiv](https://arxiv.org/abs/2501.16496) |
| 2025 | **Trends in NLP Model Interpretability in the Era of LLMs** | NAACL | [Paper](https://aclanthology.org/2025.naacl-long.29.pdf) |
| 2025 | **Representation Engineering for LLMs: Survey and Research Challenges** | arXiv | [arXiv](https://arxiv.org/abs/2502.17601) |
| 2025 | **Interpreting Language Models Through Concept Descriptions: A Survey** | arXiv | [arXiv](https://arxiv.org/abs/2510.01048) |
| 2025 | **Neuro-Symbolic AI in 2024: A Systematic Review** | arXiv | [arXiv](https://arxiv.org/abs/2501.05435) |
| 2024 | **Mechanistic Interpretability for AI Safety — A Review** | TMLR | [arXiv](https://arxiv.org/abs/2404.14082) |
| 2024 | **The Quest for the Right Mediator: A Survey of Mechanistic Interpretability via Causal Mediation Analysis** | Computational Linguistics | [arXiv](https://arxiv.org/abs/2408.01416) |
| 2024 | **Causal Abstraction: A Compact Survey** | arXiv | [arXiv](https://arxiv.org/abs/2410.20161) |
| 2024 | **Disentangled Representation Learning** | IEEE TPAMI | [arXiv](https://arxiv.org/abs/2211.11695) |
| 2023 | **Interpretable Scientific Discovery with Symbolic Regression: A Review** | AI Review | [arXiv](https://arxiv.org/abs/2211.10873) |
| 2022 | **Probing Classifiers: Promises, Shortcomings, and Advances** | Computational Linguistics | [arXiv](https://arxiv.org/abs/2102.12452) |
| 2022 | **Interpretable Machine Learning — A Brief History, State-of-the-Art and Challenges** | ECML PKDD | [Book](https://christophm.github.io/interpretable-ml-book/) |
| 2019 | **Understanding Neural Networks via Feature Visualization: A Survey** | Springer LNCS | [arXiv](https://arxiv.org/abs/1904.08939) |

---

## Self-Interpretable Methods

### Attribution-Based

Methods that highlight which input features contribute most to a prediction, built directly into the model.

#### Attention Mechanisms

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2016 | **Rationalizing Neural Predictions** | EMNLP | [arXiv](https://arxiv.org/abs/1606.04155) |
| 2017 | **A Structured Self-Attentive Sentence Embedding** | ICLR | [arXiv](https://arxiv.org/abs/1703.03130) |
| 2018 | **Self-Explaining Neural Networks (SENN)** | NeurIPS | [arXiv](https://arxiv.org/abs/1806.07538) |
| 2019 | **Attention is not Explanation** | NAACL | [arXiv](https://arxiv.org/abs/1902.10186) |
| 2019 | **Attention is not not Explanation** | EMNLP | [arXiv](https://arxiv.org/abs/1908.04626) |
| 2020 | **Towards Transparent and Explainable Attention Models** | ACL | [arXiv](https://arxiv.org/abs/2004.14243) |
| 2021 | **Is Sparse Attention More Interpretable?** | ACL | [arXiv](https://arxiv.org/abs/2106.01087) |

#### Feature Selection and Additive Models

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2019 | **Learning to Explain: An Information-Theoretic Perspective on Feature Attribution** | ICML | [arXiv](https://arxiv.org/abs/1802.07814) |
| 2021 | **TabNet: Attentive Interpretable Tabular Learning** | AAAI | [arXiv](https://arxiv.org/abs/1908.07442) |
| 2021 | **Neural Additive Models: Interpretable ML with Neural Nets** | NeurIPS | [arXiv](https://arxiv.org/abs/2004.13912) / [Site](https://neural-additive-models.github.io/) |
| 2022 | **NODE-GAM: Neural Generalized Additive Model** | ICLR | [arXiv](https://arxiv.org/abs/2106.01613) |
| 2022 | **B-cos Networks: Alignment is All We Need for Interpretability** | CVPR | [arXiv](https://arxiv.org/abs/2205.10268) |
| 2023 | **Neural Basis Models for Interpretability** | NeurIPS | [arXiv](https://arxiv.org/abs/2205.14120) |
| 2024 | **Sparse Modern Hopfield Networks for Interpretable Classification** | NeurIPS | [Paper](https://openreview.net/forum?id=nnGTef8Wor) |

#### Information Bottleneck

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2017 | **Opening the Black Box of Deep Neural Networks via Information** | arXiv | [arXiv](https://arxiv.org/abs/1703.00810) |
| 2020 | **Restricting the Flow: Information Bottlenecks for Attribution** | ICLR | [arXiv](https://arxiv.org/abs/2001.00396) |
| 2022 | **Interpretability with Full Complexity by Constraining Feature Information** | ICLR | [Paper](https://openreview.net/forum?id=kTYJJBqbrR) |

---

### Function-Based

Methods that learn explicit symbolic or mathematical representations.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2018 | **Extrapolation and Learning Equations (EQL)** | ICML | [arXiv](https://arxiv.org/abs/1806.02962) |
| 2020 | **AI Feynman 2.0: Pareto-Optimal Symbolic Regression** | NeurIPS | [arXiv](https://arxiv.org/abs/2006.10782) |
| 2021 | **SymbolicGPT: A Generative Transformer for Symbolic Regression** | arXiv | [arXiv](https://arxiv.org/abs/2106.14131) |
| 2024 | **KAN: Kolmogorov-Arnold Networks** | ICLR 2025 | [arXiv](https://arxiv.org/abs/2404.19756) / [Code](https://github.com/KindXiaoming/pykan) |

> See also: [Kolmogorov-Arnold Networks (KAN)](#kolmogorov-arnold-networks-kan) and [Symbolic Regression](#symbolic-regression-and-scientific-discovery)

---

### Concept-Based

Models that predict human-interpretable concepts as intermediate steps.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2020 | **Concept Bottleneck Models (CBM)** | ICML | [arXiv](https://arxiv.org/abs/2007.04612) |
| 2022 | **Concept Embedding Models** | NeurIPS | [arXiv](https://arxiv.org/abs/2209.09056) |
| 2022 | **Post-hoc Concept Bottleneck Models** | ICLR | [arXiv](https://arxiv.org/abs/2205.15480) |
| 2023 | **Label-Free Concept Bottleneck Models** | ICLR | [arXiv](https://arxiv.org/abs/2304.06129) |
| 2023 | **Language in a Bottle: Language Model Guided Concept Bottlenecks** | CVPR | [arXiv](https://arxiv.org/abs/2211.11158) |
| 2025 | **Counterfactual Concept Bottleneck Models** | ICLR | |
| 2025 | **Causal Concept Graph Models** | ICLR | |
| 2025 | **Concept Bottleneck LLMs** | ICLR | |
| 2025 | **CONDA: Condensed Deep Association Learning** | ICLR | |
| 2025 | **ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features** | ICML | [arXiv](https://arxiv.org/abs/2502.04320) / [Project](https://alechelbling.com/ConceptAttention/) |
| 2025 | **Interpretable Generative Models through Post-hoc Concept Bottlenecks** | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kulkarni_Interpretable_Generative_Models_through_Post-hoc_Concept_Bottlenecks_CVPR_2025_paper.pdf) |
| 2025 | **Interpretable Prognostics with Concept Bottleneck Models** | Information Fusion | [Paper](https://www.sciencedirect.com/science/article/pii/S1566253525005007) |

---

### Prototype-Based

Models that explain predictions by comparing inputs to learned prototypical parts.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2019 | **This Looks Like That: ProtoPNet** | NeurIPS | [arXiv](https://arxiv.org/abs/1806.10574) |
| 2021 | **Neural Prototype Trees (ProtoTree)** | CVPR | [arXiv](https://arxiv.org/abs/2012.02046) |
| 2021 | **Deformable ProtoPNet** | CVPR | [arXiv](https://arxiv.org/abs/2111.15000) |
| 2023 | **PIP-Net: Patch-Based Intuitive Prototypes** | CVPR | [arXiv](https://arxiv.org/abs/2307.01112) |
| 2023 | **PGIB: Prototype Graph Information Bottleneck** | NeurIPS | |
| 2025 | **LucidPPN: Lucid Prototypical Parts Network** | ICLR | |
| 2025 | **Rashomon Sets for Prototypical-Part Networks** | CVPR | [arXiv](https://arxiv.org/abs/2503.01087) |
| 2025 | **ProtoLens: Fine-Grained Interpretability in Text Classification** | ACL | [arXiv](https://arxiv.org/abs/2410.17546) |
| 2025 | **A Robust Prototype-Based Network with Interpretable RBF Classifier** | arXiv | [arXiv](https://arxiv.org/abs/2412.15499) |
| 2025 | **ProtoPGTN: Gated Transformer Network for Time Series** | Information (MDPI) | [Paper](https://www.mdpi.com/2078-2489/16/12/1056) |
| 2026 | **Interpretable Prototype Parts-Based NN for Medical Tabular Data** | arXiv | [arXiv](https://arxiv.org/abs/2603.05423) |

---

### Rule-Based

Models using decision trees, logic rules, or differentiable logic gates.

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2018 | **Deep Neural Decision Trees (DNDT)** | arXiv | [arXiv](https://arxiv.org/abs/1806.06988) |
| 2021 | **Neural-Backed Decision Trees (NBDT)** | ICLR | [arXiv](https://arxiv.org/abs/2004.00221) |
| 2022 | **Differentiable Logic Gate Networks** | NeurIPS | [arXiv](https://arxiv.org/abs/2210.08277) |
| 2023 | **Logic Explained Networks** | Artificial Intelligence | [arXiv](https://arxiv.org/abs/2108.05149) |
| 2024 | **Convolutional Differentiable Logic Gate Networks** | NeurIPS | |

---

## Mechanistic Interpretability

The study of reverse-engineering the computations learned by neural networks, understanding them as computational graphs.

### Circuit Discovery

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **Interpretability in the Wild: IOI Circuit in GPT-2 Small** | ICLR 2023 | [arXiv](https://arxiv.org/abs/2211.00593) |
| 2023 | **Towards Automated Circuit Discovery (ACDC)** | NeurIPS (Spotlight) | [arXiv](https://arxiv.org/abs/2304.14997) / [Code](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) |
| 2024 | **Sparse Feature Circuits: Discovering Interpretable Causal Graphs** | ICLR 2025 | [arXiv](https://arxiv.org/abs/2403.19647) / [Code](https://github.com/saprmarks/feature-circuits) |
| 2024 | **LLM Circuit Analyses Are Consistent Across Training and Scale** | NeurIPS | [arXiv](https://arxiv.org/abs/2407.10827) |
| 2025 | **Circuit Tracing: Revealing Computational Graphs in Language Models** | Anthropic | [Methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) / [Biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) / [Code](https://github.com/anthropics/attribution-graphs-frontend) |
| 2025 | **Weight-Sparse Transformers Have Interpretable Circuits** | OpenAI | [arXiv](https://arxiv.org/abs/2511.13653) |
| 2025 | **Circuit Insights: Towards Interpretability Beyond Activations** | arXiv | [arXiv](https://arxiv.org/abs/2510.14936) |
| 2025 | **The Computational Complexity of Circuit Discovery** | NeurIPS | [Paper](https://openreview.net/forum?id=QogcGNXJVw) |
| 2026 | **Formal Mechanistic Interpretability: Circuit Discovery with Provable Guarantees** | arXiv | [arXiv](https://arxiv.org/abs/2602.16823) |
| 2026 | **Certified Circuits: Stability Guarantees for Mechanistic Circuits** | arXiv | [arXiv](https://arxiv.org/abs/2602.22968) |

### Sparse Autoencoders

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **Towards Monosemanticity: Decomposing Language Models with Dictionary Learning** | Anthropic | [Paper](https://transformer-circuits.pub/2023/monosemantic-features/) |
| 2024 | **Sparse Autoencoders Find Highly Interpretable Features in Language Models** | ICLR 2025 | [arXiv](https://arxiv.org/abs/2309.08600) |
| 2024 | **Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet** | Anthropic | [Paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/) |
| 2025 | **Gemma Scope 2** | Google DeepMind | [Blog](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/) / [HuggingFace](https://huggingface.co/google/gemma-scope-2) / [Neuronpedia](https://www.neuronpedia.org/gemma-scope-2) |
| 2025 | **Decoding Dark Matter: Specialized SAEs for Rare Concepts** | NAACL Findings | [arXiv](https://arxiv.org/abs/2411.00743) |
| 2025 | **TIDE: Temporal-Aware SAEs for Diffusion Transformers** | AAAI | [arXiv](https://arxiv.org/abs/2503.07050) |
| 2025 | **SAEs Uncover Biologically Interpretable Features in Protein Language Models** | PNAS | [Paper](https://www.pnas.org/doi/10.1073/pnas.2506316122) |
| 2025 | **Emergence and Evolution of Interpretable Concepts in Diffusion Models** | NeurIPS (Spotlight) | [arXiv](https://arxiv.org/abs/2504.15473) |
| 2026 | **DLM-Scope: SAE-based Interpretability for Diffusion Language Models** | arXiv | [arXiv](https://arxiv.org/abs/2602.05859) |

### Superposition and Features

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **Toy Models of Superposition** | Anthropic | [Paper](https://transformer-circuits.pub/2022/toy_model/) / [arXiv](https://arxiv.org/abs/2209.10652) |
| 2022 | **Polysemanticity and Capacity in Neural Networks** | arXiv | [arXiv](https://arxiv.org/abs/2210.01892) |
| 2025 | **PRISM: Multi-Concept Feature Description Framework** | NeurIPS | [Poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/117141) |
| 2025 | **Sparse but not Simpler: Multi-Level Interpretability of Vision Transformers** | arXiv | [arXiv](https://arxiv.org/abs/2603.15919) |

---

## Representation Analysis and Probing

### Probing Classifiers

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2016 | **Understanding Intermediate Layers Using Linear Classifier Probes** | ICLR Workshop | [arXiv](https://arxiv.org/abs/1610.01644) |
| 2018 | **What You Can Cram into a Single Vector: Probing Sentence Embeddings** | ACL | [arXiv](https://arxiv.org/abs/1805.01070) / [Code](https://github.com/facebookresearch/SentEval) |
| 2019 | **A Structural Probe for Finding Syntax in Word Representations** | NAACL | [Paper](https://aclanthology.org/N19-1419/) / [Code](https://github.com/john-hewitt/structural-probes) |
| 2019 | **Designing and Interpreting Probes with Control Tasks** | EMNLP | [arXiv](https://arxiv.org/abs/1909.03368) / [Code](https://github.com/john-hewitt/control-tasks) |
| 2019 | **BERT Rediscovers the Classical NLP Pipeline** | ACL | [arXiv](https://arxiv.org/abs/1905.05950) |
| 2020 | **A Primer in BERTology: What We Know About How BERT Works** | TACL | [arXiv](https://arxiv.org/abs/2002.12327) |
| 2022 | **Probing Classifiers: Promises, Shortcomings, and Advances** | Computational Linguistics | [arXiv](https://arxiv.org/abs/2102.12452) |

### Logit Lens and Tuned Lens

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2020 | **Interpreting GPT: The Logit Lens** | LessWrong | [Post](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) |
| 2023 | **Eliciting Latent Predictions from Transformers with the Tuned Lens** | NeurIPS | [arXiv](https://arxiv.org/abs/2303.08112) |
| 2025 | **LogitLens4LLMs: Extending Logit Lens to Modern LLMs** | arXiv | [arXiv](https://arxiv.org/abs/2503.11667) / [Code](https://github.com/zhenyu-02/LogitLens4LLMs) |

### Representation Engineering and Control Vectors

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **Representation Engineering: A Top-Down Approach to AI Transparency** | arXiv | [arXiv](https://arxiv.org/abs/2310.01405) / [Code](https://github.com/andyzoujm/representation-engineering) |
| 2023 | **Activation Addition: Steering Language Models Without Optimization** | arXiv | [arXiv](https://arxiv.org/abs/2308.10248) |
| 2024 | **Steering Llama 2 via Contrastive Activation Addition (CAA)** | ACL | [arXiv](https://arxiv.org/abs/2312.06681) / [Code](https://github.com/nrimsky/CAA) |
| 2025 | **Programming Refusal with Conditional Activation Steering (CAST)** | ICLR (Spotlight) | [arXiv](https://arxiv.org/abs/2409.05907) / [Code](https://github.com/IBM/activation-steering) |
| 2025 | **Interpretable Steering of LLMs with Feature Guided Activation Additions** | ICLR | [arXiv](https://arxiv.org/abs/2501.09929) |
| 2025 | **Representation Engineering for LLMs: Survey and Research Challenges** | arXiv | [arXiv](https://arxiv.org/abs/2502.17601) |

### Linear Representation Hypothesis

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024 | **The Linear Representation Hypothesis and the Geometry of Large Language Models** | ICML | [arXiv](https://arxiv.org/abs/2311.03658) |
| 2023 | **The Geometry of Truth: Emergent Linear Structure in LLM Representations** | COLM 2024 | [arXiv](https://arxiv.org/abs/2310.06824) |

### Representation Similarity

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2008 | **Representational Similarity Analysis (RSA)** | Frontiers in Systems Neuroscience | [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2605405/) |
| 2017 | **SVCCA: Singular Vector Canonical Correlation Analysis** | NeurIPS | [arXiv](https://arxiv.org/abs/1706.05806) / [Code](https://github.com/google/svcca) |
| 2018 | **Projection Weighted CCA (PWCCA)** | NeurIPS | [arXiv](https://arxiv.org/abs/1806.05759) |
| 2019 | **Similarity of Neural Network Representations Revisited (CKA)** | ICML | [arXiv](https://arxiv.org/abs/1905.00414) |

### Concept Erasure

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2020 | **INLP: Guarding Protected Attributes by Iterative Nullspace Projection** | ACL | [arXiv](https://arxiv.org/abs/2004.07667) / [Code](https://github.com/shauli-ravfogel/nullspace_projection) |
| 2021 | **Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals** | TACL | [arXiv](https://arxiv.org/abs/2006.00995) |
| 2023 | **LEACE: Perfect Linear Concept Erasure in Closed Form** | NeurIPS | [arXiv](https://arxiv.org/abs/2306.03819) |

---

## Causal Interpretability

### Causal Abstraction

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2020 | **Causal Mediation Analysis for Interpreting Neural NLP** | NeurIPS | [arXiv](https://arxiv.org/abs/2004.12265) |
| 2021 | **Causal Abstractions of Neural Networks** | NeurIPS | [arXiv](https://arxiv.org/abs/2106.02997) |
| 2022 | **Inducing Causal Structure for Interpretable Neural Networks (IIT)** | ICML | [arXiv](https://arxiv.org/abs/2112.00826) |
| 2023 | **Finding Alignments: Distributed Alignment Search (DAS)** | CLeaR 2024 | [arXiv](https://arxiv.org/abs/2303.02536) |
| 2023 | **Interpretability at Scale: Boundless DAS for Alpaca** | NeurIPS | [arXiv](https://arxiv.org/abs/2305.08809) |
| 2023 | **Causal Scrubbing: Rigorously Testing Interpretability Hypotheses** | Alignment Forum | [Post](https://www.alignmentforum.org/s/h95ayYYwMebGEYN5y) |
| 2025 | **Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability** | JMLR | [arXiv](https://arxiv.org/abs/2301.04709) |
| 2025 | **Combining Causal Models for More Accurate Abstractions** | arXiv | [arXiv](https://arxiv.org/abs/2503.11429) |
| 2025 | **The Non-Linear Representation Dilemma: Is Causal Abstraction Enough?** | arXiv | [arXiv](https://arxiv.org/abs/2507.08802) |
| 2026 | **Transformer Is Inherently a Causal Learner** | arXiv | [arXiv](https://arxiv.org/abs/2601.05647) |

### Activation Patching and Causal Tracing

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **Locating and Editing Factual Associations in GPT (ROME)** | NeurIPS | [arXiv](https://arxiv.org/abs/2202.05262) / [Code](https://rome.baulab.info/) |
| 2023 | **Attribution Patching: Activation Patching at Industrial Scale** | Alignment Forum | [Post](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching) |
| 2024 | **Towards Best Practices of Activation Patching** | ICLR | [arXiv](https://arxiv.org/abs/2309.16042) |
| 2024 | **How to Use and Interpret Activation Patching** | arXiv | [arXiv](https://arxiv.org/abs/2404.15255) |
| 2024 | **Attribution Patching Outperforms Automated Circuit Discovery** | BlackboxNLP | [arXiv](https://arxiv.org/abs/2310.10348) / [Code](https://github.com/Aaquib111/edge-attribution-patching) |
| 2024 | **Towards Principled Evaluations of SAEs for Interpretability and Control** | arXiv | [arXiv](https://arxiv.org/abs/2405.08366) |

### Path Patching and Automated Circuit Discovery

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **Localizing Model Behavior with Path Patching** | arXiv | [arXiv](https://arxiv.org/abs/2304.05969) |
| 2023 | **ACDC: Towards Automated Circuit Discovery** | NeurIPS (Spotlight) | [arXiv](https://arxiv.org/abs/2304.14997) / [Code](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) |
| 2024 | **Efficient Automated Circuit Discovery via Contextual Decomposition** | arXiv | [arXiv](https://arxiv.org/abs/2407.00886) |

---

## Feature Visualization

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2009 | **Visualizing Higher-Layer Features of a Deep Network** | Technical Report | [Paper](https://www.semanticscholar.org/paper/Visualizing-Higher-Layer-Features-of-a-Deep-Network-Erhan-Bengio/65d994fb778a8d9e0f632659fb33a082949a50d3) |
| 2014 | **Visualizing and Understanding Convolutional Networks** | ECCV | [arXiv](https://arxiv.org/abs/1311.2901) |
| 2015 | **Inceptionism: Going Deeper into Neural Networks (DeepDream)** | Google Research | [Blog](https://research.google/blog/inceptionism-going-deeper-into-neural-networks/) / [Code](https://github.com/google/deepdream) |
| 2015 | **Understanding Neural Networks Through Deep Visualization** | ICML Workshop | [arXiv](https://arxiv.org/abs/1506.06579) / [Code](https://github.com/yosinski/deep-visualization-toolbox) |
| 2016 | **Multifaceted Feature Visualization** | ICML Workshop (Best Paper) | [arXiv](https://arxiv.org/abs/1602.03616) |
| 2017 | **Feature Visualization** | Distill | [Paper](https://distill.pub/2017/feature-visualization/) |
| 2017 | **Network Dissection: Quantifying Interpretability of Deep Visual Representations** | CVPR | [arXiv](https://arxiv.org/abs/1704.05796) / [Code](https://github.com/CSAILVision/NetDissect) |
| 2019 | **GAN Dissection: Visualizing and Understanding GANs** | ICLR | [arXiv](https://arxiv.org/abs/1811.10597) / [Code](https://github.com/CSAILVision/gandissect) |
| 2019 | **Exploring Neural Networks with Activation Atlases** | Distill | [Paper](https://distill.pub/2019/activation-atlas/) |
| 2023 | **MACO: Magnitude Constrained Optimization for Feature Visualization** | NeurIPS | [arXiv](https://arxiv.org/abs/2306.06805) |

---

## Disentangled Representations

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2016 | **InfoGAN: Interpretable Representation Learning** | NeurIPS | [arXiv](https://arxiv.org/abs/1606.03657) |
| 2017 | **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework** | ICLR | [Paper](https://openreview.net/forum?id=Sy2fzU9gl) |
| 2018 | **DCI: A Framework for Quantitative Evaluation of Disentangled Representations** | ICLR | [Paper](https://www.researchgate.net/publication/323727238) |
| 2018 | **Disentangling by Factorising (FactorVAE)** | ICML | [arXiv](https://arxiv.org/abs/1802.05983) / [Code](https://github.com/paruby/FactorVAE) |
| 2018 | **Isolating Sources of Disentanglement (β-TCVAE)** | NeurIPS | [arXiv](https://arxiv.org/abs/1802.04942) / [Code](https://github.com/rtqichen/beta-tcvae) |
| 2018 | **Towards a Definition of Disentangled Representations** | arXiv | [arXiv](https://arxiv.org/abs/1812.02230) |
| 2019 | **Challenging Common Assumptions in Unsupervised Disentanglement** | ICML (Best Paper) | [arXiv](https://arxiv.org/abs/1811.12359) / [Code](https://github.com/google-research/disentanglement_lib) |
| 2023 | **DCI-ES: Extended Disentanglement Framework with Identifiability** | ICLR | [arXiv](https://arxiv.org/abs/2210.00364) |
| 2024 | **Graph-based Unsupervised Disentangled Representation Learning** | NeurIPS | [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/bac4d92b3f6decfe47eab9a5893dd1f6-Paper-Conference.pdf) |
| 2025 | **Disentangled Representation Learning with the Gromov-Monge Gap** | ICLR | [Paper](https://openreview.net/forum?id=ehr4oTe6XI) |
| 2025 | **Disentangling Disentangled Representations via Diffusion Models** | WACV | [Paper](https://openaccess.thecvf.com/content/WACV2025/html/Jun_Disentangling_Disentangled_Representations_Towards_Improved_Latent_Units_via_Diffusion_Models_WACV_2025_paper.html) |

---

## Kolmogorov-Arnold Networks (KAN)

Networks with learnable activation functions on edges, offering a fundamentally interpretable alternative to MLPs.

### Core Papers

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024 | **KAN: Kolmogorov-Arnold Networks** | ICLR 2025 | [arXiv](https://arxiv.org/abs/2404.19756) / [Code](https://github.com/KindXiaoming/pykan) |
| 2024 | **KAN 2.0: Kolmogorov-Arnold Networks Meet Science** | arXiv | [arXiv](https://arxiv.org/abs/2408.10205) / [Code](https://github.com/KindXiaoming/pykan) |
| 2025 | **Kolmogorov-Arnold Networks Meet Science** | Physical Review X | [Paper](https://link.aps.org/doi/10.1103/4t7t-v19l) |

### Variants and Applications

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **KA-GNN: Kolmogorov-Arnold Graph Neural Networks** | Nature Machine Intelligence | [arXiv](https://arxiv.org/abs/2410.11323) / [Paper](https://www.nature.com/articles/s42256-025-01087-7) |
| 2025 | **CoxKAN: Interpretable Survival Analysis** | Bioinformatics | [arXiv](https://arxiv.org/abs/2409.04290) / [Code](https://github.com/knottwill/CoxKAN) |
| 2025 | **Interpretable KANs for Enzyme Commission Number Prediction** | npj AI | [Paper](https://www.nature.com/articles/s44387-025-00059-x) |
| 2025 | **KAN-AFT: Interpretable Nonlinear Survival Model** | arXiv | [arXiv](https://arxiv.org/abs/2512.20305) |

### Resources

- [pykan](https://github.com/KindXiaoming/pykan) — Official KAN implementation
- [awesome-kan](https://github.com/mintisan/awesome-kan) — Comprehensive KAN resource list
- [All-KAN](https://github.com/hoangthangta/All-KAN) — All KAN variants
- [KAN Papers Collection](https://ramtinmoslemi.github.io/KAN-Papers/)

---

## Neuro-Symbolic AI

### Frameworks and Languages

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2018 | **DeepProbLog: Neural Probabilistic Logic Programming** | NeurIPS | [arXiv](https://arxiv.org/abs/1805.10872) |
| 2019 | **The Neuro-Symbolic Concept Learner** | ICLR | [arXiv](https://arxiv.org/abs/1904.12584) |
| 2020 | **Neurosymbolic AI: The 3rd Wave** | AI Review | [arXiv](https://arxiv.org/abs/2012.05876) |
| 2022 | **Logic Tensor Networks** | Artificial Intelligence | [arXiv](https://arxiv.org/abs/2012.13635) |
| 2023 | **Scallop: A Language for Neurosymbolic Programming** | PLDI | [arXiv](https://arxiv.org/abs/2304.04812) / [Code](https://github.com/scallop-lang/scallop) |

### Program Synthesis and Library Learning

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2017 | **Differentiable Programs with Neural Libraries** | ICML | [arXiv](https://arxiv.org/abs/1611.02109) |
| 2021 | **DreamCoder: Growing Generalizable, Interpretable Knowledge** | PLDI | [arXiv](https://arxiv.org/abs/2006.08381) |
| 2024 | **LILO: Learning Interpretable Libraries by Compressing and Documenting Code** | ICLR | [arXiv](https://arxiv.org/abs/2310.19791) |
| 2024 | **AlphaGeometry: Solving Olympiad Geometry without Human Demonstrations** | Nature | [Paper](https://www.nature.com/articles/s41586-023-06747-5) |

### Symbolic Regression and Scientific Discovery

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **LIES Networks: Sparse Interpretable Deep Learning for Symbolic Regression** | arXiv | [arXiv](https://arxiv.org/abs/2506.08267) |
| 2025 | **Learning Interpretable Network Dynamics via Universal Neural Symbolic Regression** | Nature Communications | [Paper](https://www.nature.com/articles/s41467-025-61575-7) |
| 2025 | **Discovering Physical Laws with Parallel Symbolic Enumeration** | Nature Computational Science | [Paper](https://www.nature.com/articles/s43588-025-00904-8) |

### LLM + Symbolic Reasoning

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **NeSyCoCo: Neuro-Symbolic Concept Composer for Compositional Generalization** | AAAI | [arXiv](https://arxiv.org/abs/2412.15588) |
| 2025 | **Right for the Right Reasons: Prototypical Neurosymbolic AI** | NeurIPS | [arXiv](https://arxiv.org/abs/2510.25497) |
| 2025 | **Delta-1-LLM: Symbolic-Neural Integration for Credible Reasoning** | arXiv | [arXiv](https://arxiv.org/abs/2603.12953) |

---

## Foundation Model Interpretability

### LLM Internals

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **In-context Learning and Induction Heads** | Anthropic | [arXiv](https://arxiv.org/abs/2209.11895) / [Blog](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) |
| 2024 | **Summing Up the Facts: Additive Mechanisms Behind Factual Recall in LLMs** | arXiv | [arXiv](https://arxiv.org/abs/2402.07321) |
| 2025 | **On the Biology of a Large Language Model** | Anthropic | [Paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) |
| 2025 | **Selective Induction Heads: How Transformers Select Among Causal Structures** | ICLR | [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/d7ed243b13831bdd468f35039936bcef-Paper-Conference.pdf) |
| 2025 | **Beyond Induction Heads: Multi-Phase Circuit Emergence** | arXiv | [arXiv](https://arxiv.org/abs/2505.16694) |
| 2025 | **Which Attention Heads Matter for In-Context Learning?** | arXiv | [arXiv](https://arxiv.org/abs/2502.14010) |
| 2025 | **Tracing and Dissecting How LLMs Recall Factual Knowledge** | ACL | [Paper](https://aclanthology.org/2025.acl-long.1133.pdf) |
| 2025 | **Interpretability in Parameter Space** | arXiv | [arXiv](https://arxiv.org/abs/2501.14926) |
| 2025 | **Mechanistic Interpretability of Code Correctness in LLMs via SAEs** | arXiv | [arXiv](https://arxiv.org/abs/2510.02917) |
| 2025 | **A Review of Developmental Interpretability in LLMs** | arXiv | [arXiv](https://arxiv.org/abs/2508.15841) |

### Diffusion Model Interpretability

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **The Hidden Language of Diffusion Models (Conceptor)** | ICLR 2024 | [Paper](https://openreview.net/forum?id=awWpHnEJDw) |
| 2023 | **Discovering Interpretable Directions in Diffusion Latent Space** | arXiv | [arXiv](https://arxiv.org/abs/2303.11073) |
| 2024 | **Self-Discovering Interpretable Diffusion Latent Directions** | CVPR | [arXiv](https://arxiv.org/abs/2311.17216) |
| 2025 | **ConceptAttention: Diffusion Transformers Learn Interpretable Features** | ICML | [arXiv](https://arxiv.org/abs/2502.04320) / [Code](https://github.com/helblazer811/ConceptAttention) |
| 2025 | **DiffLens: Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability** | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Shi_Dissecting_and_Mitigating_Diffusion_Bias_via_Mechanistic_Interpretability_CVPR_2025_paper.pdf) |
| 2025 | **Emergence and Evolution of Interpretable Concepts in Diffusion Models** | NeurIPS (Spotlight) | [arXiv](https://arxiv.org/abs/2504.15473) |
| 2025 | **Mechanistic Interpretability of Diffusion Models: Circuit-Level Analysis** | arXiv | [arXiv](https://arxiv.org/abs/2506.17237) |
| 2025 | **Decoding Vision Transformers: the Diffusion Steering Lens** | arXiv | [arXiv](https://arxiv.org/abs/2504.13763) |

### Vision-Language Models

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024 | **Interpreting CLIP's Image Representation via Text-based Decomposition (TEXTSPAN)** | ICLR | [Paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/5085a2b8c298edeadc46b9ffe6df5f64-Paper-Conference.pdf) |
| 2024 | **Vision Transformers Need Registers** | ICLR | [arXiv](https://arxiv.org/abs/2309.16588) |
| 2025 | **Boosting the Visual Interpretability of CLIP via Adversarial Fine-Tuning** | ICLR | [Paper](https://openreview.net/forum?id=khuIvzxPRp) |
| 2025 | **Interpreting Attention Heads in Vision-Language Models** | arXiv | [arXiv](https://arxiv.org/abs/2509.17588) |

### Multimodal Interpretability

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models** | arXiv | [arXiv](https://arxiv.org/abs/2502.17516) |
| 2025 | **Mechanistic Interpretability Meets Vision Language Models: Insights and Limitations** | ICLR Blog Track | [Blog](https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/blog/vlm-understanding/) |
| 2025 | **Causal Tracing of Object Representations in Large Vision Language Models (FCCT)** | arXiv | [arXiv](https://arxiv.org/abs/2511.05923) |

---

## Safety and Alignment

### Deception Detection

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024 | **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** | arXiv | [arXiv](https://arxiv.org/abs/2401.05566) |
| 2024 | **Simple Probes Can Catch Sleeper Agents** | Anthropic | [Blog](https://www.anthropic.com/research/probes-catch-sleeper-agents) |
| 2025 | **Detecting Strategic Deception Using Linear Probes** | arXiv | [arXiv](https://arxiv.org/abs/2502.03407) |
| 2025 | **Caught in the Act: A Mechanistic Approach to Detecting Deception** | arXiv | [arXiv](https://arxiv.org/abs/2508.19505) |
| 2025 | **Liars' Bench: Evaluating Lie Detectors for Language Models** | arXiv | [arXiv](https://arxiv.org/abs/2511.16035) / [Data](https://huggingface.co/datasets/Cadenza-Labs/liars-bench) |
| 2025 | **Preference Learning with Lie Detectors can Induce Honesty or Evasion** | NeurIPS | [arXiv](https://arxiv.org/abs/2505.13787) |

### Truthfulness Representations

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **The Geometry of Truth: Emergent Linear Structure in LLM Representations** | COLM 2024 | [arXiv](https://arxiv.org/abs/2310.06824) |
| 2023 | **Inference-Time Intervention: Eliciting Truthful Answers** | NeurIPS | [arXiv](https://arxiv.org/abs/2306.03341) / [Code](https://github.com/likenneth/honest_llama) |
| 2025 | **Truth Neurons** | arXiv | [arXiv](https://arxiv.org/abs/2505.12182) |
| 2025 | **The Truthfulness Spectrum Hypothesis** | arXiv | [arXiv](https://arxiv.org/abs/2602.20273) |
| 2025 | **The Confidence Manifold: Geometric Structure of Correctness Representations** | arXiv | [arXiv](https://arxiv.org/abs/2602.08159) |

### Steering Vectors and Activation Control

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024 | **Refusal in Language Models Is Mediated by a Single Direction** | NeurIPS | [arXiv](https://arxiv.org/abs/2406.11717) / [Code](https://github.com/andyrdt/refusal_direction) |
| 2024 | **Improving Alignment and Robustness with Circuit Breakers** | NeurIPS | [arXiv](https://arxiv.org/abs/2406.04313) / [Code](https://github.com/GraySwanAI/circuit-breakers) |
| 2025 | **RepBend: Representation Bending for LLM Safety** | ACL | [arXiv](https://arxiv.org/abs/2504.01550) |
| 2025 | **Steering LLMs using Conceptors** | arXiv | [arXiv](https://arxiv.org/abs/2410.16314) |
| 2025 | **EasySteer: A Unified Framework for LLM Steering** | arXiv | [arXiv](https://arxiv.org/abs/2509.25175) / [Code](https://github.com/ZJU-REAL/EasySteer) |
| 2025 | **Activation Steering with a PID Feedback Controller** | arXiv | [arXiv](https://arxiv.org/abs/2510.04309) |
| 2025 | **Steering Awareness: Detecting Activation Steering from Within** | arXiv | [arXiv](https://arxiv.org/abs/2511.21399) |

### Jailbreak Analysis

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **JBShield: Defending LLMs via Activated Concept Analysis** | USENIX Security | [arXiv](https://arxiv.org/abs/2502.07557) / [Code](https://github.com/NISPLab/JBShield) |
| 2025 | **Jailbreaking Leaves a Trace: Understanding from Internal Representations** | arXiv | [arXiv](https://arxiv.org/abs/2602.11495) |
| 2025 | **LLM Salting** | CAMLIS | [Blog](https://www.sophos.com/en-us/blog/getting-salty-with-llms-sophosai-unveils-new-defense-against-jailbreaking-at-camlis-2025) |

---

## Training for Interpretability

### Sparsity and Regularization

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2019 | **The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks** | ICLR | [arXiv](https://arxiv.org/abs/1803.03635) |
| 2020 | **The Lottery Ticket Hypothesis for Pre-trained BERT** | NeurIPS | [Paper](https://proceedings.neurips.cc/paper/2020/file/b6af2c9703f203a2794be03d443af2e3-Paper.pdf) |
| 2024 | **Position: A Theory of Deep Learning Must Include Compositional Sparsity** | ICML | [Paper](https://openreview.net/pdf?id=A0HtZM0MpZ) |
| 2025 | **Geometric Sparsification in Recurrent Neural Networks** | npj AI | [Paper](https://www.nature.com/articles/s44387-025-00013-x) |

### Modular Networks

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2024 | **Are Neural Nets Modular? Inspecting via Differentiable Weight Masks** | ICLR | [Paper](https://openreview.net/forum?id=7uVcpu-gMD) |
| 2024 | **Training Neural Networks for Modularity aids Interpretability** | arXiv | [arXiv](https://arxiv.org/abs/2409.15747) |
| 2025 | **Modular Training of Neural Networks aids Interpretability** | arXiv | [arXiv](https://arxiv.org/abs/2502.02470) |

### Knowledge Distillation for Interpretability

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2018 | **Born Again Neural Networks** | ICML | [arXiv](https://arxiv.org/abs/1805.04770) |
| 2023 | **On the Impact of Knowledge Distillation for Model Interpretability** | ICML | [arXiv](https://arxiv.org/abs/2305.15734) |
| 2024 | **Explainability-based Knowledge Distillation (Exp-KD)** | Pattern Recognition | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S003132032400846X) |
| 2024 | **DiXtill: XAI-Driven Knowledge Distillation** | Journal of Big Data | [Paper](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-024-00928-3) |
| 2025 | **VICO-KD: Visual Concept KD for Concept Bottleneck Models** | Applied Sciences | [Paper](https://www.mdpi.com/2076-3417/15/2/493) |

---

## Adversarial Robustness and Interpretability

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2018 | **Improving Adversarial Robustness and Interpretability by Regularizing Input Gradients** | AAAI | [arXiv](https://arxiv.org/abs/1711.09404) |
| 2019 | **On the Connection Between Adversarial Robustness and Saliency Map Interpretability** | ICML | [arXiv](https://arxiv.org/abs/1905.04172) |
| 2020 | **Proper Network Interpretability Helps Adversarial Robustness in Classification** | arXiv | [arXiv](https://arxiv.org/abs/2006.14748) |
| 2023 | **Interpretable Computer Vision Models through Adversarial Training** | arXiv | [arXiv](https://arxiv.org/abs/2307.02500) |
| 2025 | **Adversarial Attacks in Explainable ML: A Survey of Threats Against Models and Humans** | WIREs | [Paper](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1567) |

---

## Scaling Laws for Interpretability

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **Toy Models of Superposition** | Anthropic | [Paper](https://transformer-circuits.pub/2022/toy_model/) |
| 2024 | **Scaling Monosemanticity** | Anthropic | [Paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/) |
| 2025 | **Scaling Interpretable Language Models to 8 Billion Parameters** | Guide Labs | [Blog](https://www.guidelabs.ai/post/scaling-interpretable-models-8b/) |
| 2025 | **Sparse but not Simpler: Multi-Level Interpretability of Vision Transformers** | arXiv | [arXiv](https://arxiv.org/abs/2603.15919) |
| 2025 | **SAFR: Neuron Redistribution for Interpretability** | NAACL Findings | [Paper](https://aclanthology.org/2025.findings-naacl.112.pdf) |

---

## Interpretable Transformers

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **A Simple Interpretable Transformer for Fine-Grained Image Classification** | arXiv | [Paper](https://par.nsf.gov/biblio/10530247) |
| 2025 | **Interpretable Pre-Trained Transformers for Heart Time-Series Data** | OpenReview | [Paper](https://openreview.net/forum?id=eciCtsqGc8) |
| 2025 | **Condition Guided Self-Attention for Interpretable Transformers** | OpenReview | [Paper](https://openreview.net/forum?id=c7SXaXZcUi) |
| 2025 | **Universal Set Transformer (UST)** | ICLR | [Paper](https://openreview.net/forum?id=gIpuW5Ekiw) |
| 2025 | **Lightweight Interpretable Transformer via Unrolling Mixed Graph Algorithms** | OpenReview | [Paper](https://openreview.net/forum?id=4xal4WSkQt) |

---

## Applications

### Vision

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **B-cos Networks: Alignment is All We Need for Interpretability** | CVPR | [arXiv](https://arxiv.org/abs/2205.10268) |
| 2023 | **PIP-Net: Patch-Based Intuitive Prototypes** | CVPR | [arXiv](https://arxiv.org/abs/2307.01112) |
| 2025 | **ConceptAttention: Diffusion Transformers Learn Interpretable Features** | ICML | [arXiv](https://arxiv.org/abs/2502.04320) |
| 2025 | **Rashomon Sets for Prototypical-Part Networks** | CVPR | [arXiv](https://arxiv.org/abs/2503.01087) |
| 2025 | **Interpretable Generative Models through Post-hoc Concept Bottlenecks** | CVPR | [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Kulkarni_Interpretable_Generative_Models_through_Post-hoc_Concept_Bottlenecks_CVPR_2025_paper.pdf) |

### NLP and LLMs

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | NeurIPS | [arXiv](https://arxiv.org/abs/2201.11903) |
| 2025 | **ProtoLens: Fine-Grained Interpretability in Text Classification** | ACL | [arXiv](https://arxiv.org/abs/2410.17546) |
| 2025 | **Concept Bottleneck LLMs** | ICLR | |
| 2025 | **Circuit Tracing in Claude 3.5 Haiku** | Anthropic | [Paper](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) |

### Graph Data

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2023 | **PGIB: Prototype Graph Information Bottleneck** | NeurIPS | |
| 2025 | **TopInG: Topologically Interpretable Graph Learning** | ICML | [arXiv](https://arxiv.org/abs/2510.05102) |
| 2025 | **DyExplainer: Self-Explainable Dynamic Graph Neural Network** | ACM TKDD | [Paper](https://dl.acm.org/doi/10.1145/3729173) |
| 2025 | **KA-GNN: Kolmogorov-Arnold Graph Neural Networks** | Nature MI | [Paper](https://www.nature.com/articles/s42256-025-01087-7) |
| 2025 | **Concept-Induced Graph Perception Model for Interpretable Diagnosis** | MICCAI | [Paper](https://papers.miccai.org/miccai-2025/0162-Paper0253.html) |

### Science and Medicine

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **DUNL: Interpretable Deep Learning for Deconvolutional Analysis of Neural Signals** | Neuron | [Paper](https://www.cell.com/neuron/abstract/S0896-6273(25)00119-9) |
| 2025 | **Bridging Clinical Knowledge and AI: Interpretable Transformer for ECG Diagnosis** | npj Digital Medicine | [Paper](https://www.nature.com/articles/s41746-025-02215-8) |
| 2025 | **SAEs Uncover Biologically Interpretable Features in Protein Language Models** | PNAS | [Paper](https://www.pnas.org/doi/10.1073/pnas.2506316122) |
| 2025 | **CoxKAN: Interpretable Survival Analysis** | Bioinformatics | [arXiv](https://arxiv.org/abs/2409.04290) / [Code](https://github.com/knottwill/CoxKAN) |
| 2025 | **Interpretable Prognostics with Concept Bottleneck Models** | Information Fusion | [Paper](https://www.sciencedirect.com/science/article/pii/S1566253525005007) |
| 2026 | **Interpretable Prototype Parts-Based NN for Medical Tabular Data** | arXiv | [arXiv](https://arxiv.org/abs/2603.05423) |

### Reinforcement Learning

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2021 | **Learning to Synthesize Programs as Interpretable RL Policies** | NeurIPS | [Paper](https://dl.acm.org/doi/10.5555/3540261.3542187) |
| 2023 | **Interpretable Logical Policies via Neurally Guided Symbolic Abstraction** | NeurIPS | |
| 2025 | **Interpretable Deep RL Via Concept-Based Policy Distillation** | Machine Learning (Springer) | [Paper](https://link.springer.com/article/10.1007/s10994-025-06928-5) |

### Time Series

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2025 | **ProtoPGTN: Gated Transformer Network for Interpretable Time Series** | Information (MDPI) | [Paper](https://www.mdpi.com/2078-2489/16/12/1056) |
| 2025 | **Interpretable Pre-Trained Transformers for Heart Time-Series Data** | OpenReview | [Paper](https://openreview.net/forum?id=eciCtsqGc8) |
| 2025 | **Interpretable Sequence Classification Via Prototype Trajectory** | OpenReview | [Paper](https://openreview.net/forum?id=KwgQn_Aws3_) |

---

## Evaluation and Benchmarks

| Year | Title | Venue | Links |
|------|-------|-------|-------|
| 2022 | **Quantus: An Explainability Toolkit for Responsible Evaluation of NN Explanations** | JMLR | [arXiv](https://arxiv.org/abs/2202.06861) |
| 2023 | **M4: A Unified XAI Benchmark for Faithfulness** | NeurIPS D&B | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/05957c194f4c77ac9d91e1374d2def6b-Paper-Datasets_and_Benchmarks.pdf) |
| 2023 | **Faithfulness Tests for Natural Language Explanations** | ACL | [arXiv](https://arxiv.org/abs/2305.18029) |
| 2024 | **Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations** | arXiv | [arXiv](https://arxiv.org/abs/2402.04614) |
| 2024 | **Challenging the Performance-Interpretability Trade-Off** | Business & IS Eng. | [Paper](https://link.springer.com/article/10.1007/s12599-024-00922-2) |
| 2025 | **MIB: A Mechanistic Interpretability Benchmark** | OpenReview | [Paper](https://openreview.net/forum?id=sSrOwve6vb) |
| 2025 | **EvalxNLP: Benchmarking Post-Hoc Explainability for NLP** | arXiv | [arXiv](https://arxiv.org/abs/2505.01238) |
| 2025 | **FaithCoT-Bench: Benchmarking CoT Faithfulness** | OpenReview | [Paper](https://openreview.net/forum?id=lN3yKqqzF1) |
| 2025 | **A Causal Lens for Evaluating Faithfulness Metrics** | arXiv | [arXiv](https://arxiv.org/abs/2502.18848) |
| 2025 | **Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?** | OpenReview | [Paper](https://openreview.net/forum?id=5IWJBStfU7) |

---

## Tools and Frameworks

### Mechanistic Interpretability Tools

| Tool | Description | Links |
|------|-------------|-------|
| **TransformerLens** | Library for mechanistic interpretability of GPT-style models. HookPoints on every activation, supports 50+ architectures. | [GitHub](https://github.com/TransformerLensOrg/TransformerLens) / [Docs](https://transformerlensorg.github.io/TransformerLens/) |
| **SAELens** | Train and analyze Sparse Autoencoders for decomposing neural network representations. | [GitHub](https://github.com/decoderesearch/SAELens) |
| **NNsight** | Access and intervene on internals of any PyTorch model. Supports local and remote execution. | [GitHub](https://github.com/ndif-team/nnsight) / [Website](https://nnsight.net/) |
| **Circuit Tracer** | Anthropic's tool for generating attribution graphs. | [GitHub](https://github.com/decoderesearch/circuit-tracer) / [Frontend](https://github.com/anthropics/attribution-graphs-frontend) |
| **Gemma Scope 2** | Google DeepMind's SAEs and transcoders for Gemma 3 models (270M–27B). | [HuggingFace](https://huggingface.co/google/gemma-scope-2) / [Neuronpedia](https://www.neuronpedia.org/gemma-scope-2) |
| **Neuronpedia** | Interactive platform for exploring SAE features and attribution graphs. | [Website](https://www.neuronpedia.org/) |
| **Google PAIR SAE Explorer** | Interactive visualization of SAE-discovered features. | [Explorer](https://pair.withgoogle.com/explorables/sae/) |
| **LogitLens4LLMs** | Logit lens toolkit for modern LLM architectures. | [GitHub](https://github.com/zhenyu-02/LogitLens4LLMs) |

### General Interpretability Libraries

| Tool | Description | Links |
|------|-------------|-------|
| **Captum** | Meta/PyTorch library for model interpretability. Gradient and perturbation-based attributions. | [GitHub](https://github.com/meta-pytorch/captum) / [Website](https://captum.ai/) |
| **SHAP** | Game-theoretic Shapley value explanations for any ML model. ~23k stars. | [GitHub](https://github.com/shap/shap) |
| **InterpretML** | Microsoft's package for glassbox (EBMs) and blackbox (SHAP, LIME) interpretability. | [GitHub](https://github.com/interpretml/interpret) |
| **Quantus** | Toolkit for evaluating neural network explanations with multiple metrics. | [GitHub](https://github.com/understandable-machine-intelligence-lab/Quantus) |
| **Lucid** | TensorFlow-based feature visualization and activation atlas generation. (Legacy) | [GitHub](https://github.com/tensorflow/lucid) |
| **EasySteer** | Unified vLLM-based framework for LLM steering across 8 domains. | [GitHub](https://github.com/ZJU-REAL/EasySteer) |

### Model Implementations

| Tool | Description | Links |
|------|-------------|-------|
| **pykan** | Official Kolmogorov-Arnold Networks implementation. | [GitHub](https://github.com/KindXiaoming/pykan) |
| **CoxKAN** | KAN-based interpretable survival analysis. | [GitHub](https://github.com/knottwill/CoxKAN) |
| **NAM Library** | Neural Additive Models with CLI for research. | [GitHub](https://github.com/AmrMKayid/nam) |
| **NAMpy** | Wide range of interpretable deep NN implementations. | [GitHub](https://github.com/AnFreTh/NAMpy) |
| **LA-NAM** | Bayesian NAM with Laplace approximation and uncertainty. | [GitHub](https://github.com/fortuinlab/LA-NAM) |
| **DiCE** | Microsoft's counterfactual explanations library. | [GitHub](https://github.com/interpretml/DiCE) |
| **Scallop** | Neurosymbolic programming language based on Datalog. | [GitHub](https://github.com/scallop-lang/scallop) |
| **representation-engineering** | Official RepE codebase (Zou, Hendrycks et al.). | [GitHub](https://github.com/andyzoujm/representation-engineering) |
| **circuit-breakers** | Reroutes harmful representations to orthogonal space. | [GitHub](https://github.com/GraySwanAI/circuit-breakers) |
| **refusal_direction** | Surgical refusal direction erasure/addition. | [GitHub](https://github.com/andyrdt/refusal_direction) |
| **feature-circuits** | Discover and edit interpretable causal graphs in LMs. | [GitHub](https://github.com/saprmarks/feature-circuits) |
| **ACDC** | Automated Circuit Discovery. | [GitHub](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) |
| **JBShield** | Jailbreak defense via concept analysis. | [GitHub](https://github.com/NISPLab/JBShield) |
| **disentanglement_lib** | Google's library for training and evaluating disentangled representations. | [GitHub](https://github.com/google-research/disentanglement_lib) |

---

## Datasets

| Name | Description | Links |
|------|-------------|-------|
| **MIB** | Mechanistic Interpretability Benchmark — circuit and causal variable localization | [Paper](https://openreview.net/forum?id=sSrOwve6vb) |
| **M4** | Unified XAI benchmark for faithfulness evaluation | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/05957c194f4c77ac9d91e1374d2def6b-Paper-Datasets_and_Benchmarks.pdf) |
| **Liars' Bench** | 72,863-example testbed of lies/honest responses for lie detection | [HuggingFace](https://huggingface.co/datasets/Cadenza-Labs/liars-bench) |
| **DERI1000** | Dataset explainability readiness benchmark | [Paper](https://www.mdpi.com/2673-2688/6/12/320) |
| **FaithCoT-Bench** | Chain-of-thought faithfulness benchmark | [Paper](https://openreview.net/forum?id=lN3yKqqzF1) |

---

## Tutorials and Workshops

| Year | Event | Links |
|------|-------|-------|
| 2025 | **Concept-based Interpretable Deep Learning** (AAAI Tutorial) | [Website](https://conceptlearning.github.io/) |
| 2025 | **Mechanistic Interpretability Workshop** (NeurIPS) | [Website](https://mechinterpworkshop.com/) |
| 2025 | **MIV Workshop: Mechanistic Interpretability for Vision** (CVPR) | [Website](https://sites.google.com/view/miv-cvpr2025/) |
| 2025 | **BlackboxNLP Workshop** | [Website](https://blackboxnlp.github.io/2025/) |
| 2025 | **2nd New England Mechanistic Interpretability (NEMI) Workshop** | [Website](https://nemiconf.github.io/summer25/) |

### Books

- [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) by Christoph Molnar — Freely available online

---

## Related Awesome Lists

| List | Focus |
|------|-------|
| [Awesome-Self-Interpretable-Neural-Network](https://github.com/yangji721/Awesome-Self-Interpretable-Neural-Network) | Self-interpretable NNs (companion to IEEE survey) |
| [Awesome-LLM-Interpretability](https://github.com/cooperleong00/Awesome-LLM-Interpretability) | LLM interpretability papers, tools, tutorials |
| [Awesome-LMMs-Mechanistic-Interpretability](https://github.com/itsqyh/Awesome-LMMs-Mechanistic-Interpretability) | Multimodal model mechanistic interpretability |
| [Awesome-Attention-Heads](https://github.com/IAAR-Shanghai/Awesome-Attention-Heads) | Attention head interpretability |
| [Awesome-SAE](https://github.com/zepingyu0512/awesome-SAE) | Sparse autoencoder papers |
| [Awesome-Representation-Engineering](https://github.com/chrisliu298/awesome-representation-engineering) | Representation engineering papers |
| [awesome-kan](https://github.com/mintisan/awesome-kan) | Kolmogorov-Arnold Networks |
| [awesome-machine-learning-interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability) | Responsible ML (interpretability + fairness) |
| [awesome-interpretable-machine-learning](https://github.com/lopusz/awesome-interpretable-machine-learning) | Papers and tools |
| [awesome-graph-explainability-papers](https://github.com/flyingdoog/awesome-graph-explainability-papers) | GNN explainability |
| [Awesome-XAI](https://github.com/altamiracorp/awesome-xai) | Explainable AI |
| [Awesome-Explainable-AI](https://github.com/wangyongjie-ntu/Awesome-explainable-AI) | Explainable AI methods |
| [Awesome-Explainable-RL](https://github.com/Plankson/awesome-explainable-reinforcement-learning) | Explainable reinforcement learning |
| [Awesome-LLM-Understanding-Mechanism](https://github.com/zepingyu0512/awesome-llm-understanding-mechanism) | LLM mechanism understanding |
| [LLM-Honesty-Survey](https://github.com/SihengLi99/LLM-Honesty-Survey) | LLM honesty survey with papers |

---

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to suggest additions.

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

This work is licensed under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/).
