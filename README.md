# Awesome-TripleR 

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated collection covering models, datasets, reward designs, optimization methods, hyperparameters, empirical findings, theoretical insights, and everything about reasoning with reinforcement learning.

[TOC]

## Overview

| Date       | Name     | Backbone            | Tasks                   | Training | Findings | Details                 |
| ---------- | -------- | ------------------- | ----------------------- | -------- | -------- | ----------------------- |
| 2025/01/24 | TinyZero | Qwen2.5-3B-Instruct | Countdown               | 4 A800s  |          | [[TinyZero]](#tinyzero) |
| 2025/03/18 | DAPO     | Qwen2.5-32B         | Mathematical  Reasoning | -        |          | [[DAPO]](#dapo)         |

## Projecst

### Template

| Project or Paper      | [Project name or Paper title]()                          |
| :-------------------- | :------------------------------------------------------- |
| GitHub                | [Username/Project]()                                     |
| Backbone Model        | (Base / Instruct / Reasoning;  HF Model)                 |
| RL Algorithm          | (PPO / GRPO / RLOO / REINFORCE++; OpenRLHF / Verl / Trl) |
| Training Dataset      | (Size / Source / HF Dataset)                             |
| Rollout Configuration | (Batch Size * N Samples ; Temperature; Dynamic Sampling) |
| Reward Function       | (Outcome; Process; Repetition & Length)                  |
| Policy Optimization   | (KL Loss; Length Penalty; Token-level loss)              |
| Benchmark             | (MATH/GPQA; R1 level; GPT-4o level)                      |
| Core Insights         | ——                                                       |
| Additional Notes      | ——                                                       |



### 2025.0102, PRIME-RL



### 2025.0122, DeepSeek-R1



### 2025.0122, Kimi k1.5

 [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599?)



### 2025.0124, TinyZero



### 2025.0124, Open-R1



### 2025.0125, SimpleRL



### 2025.0126, RAGEN



### 2025.0203, Verifier



### 2025.0206, Demysitify-long-CoT



### 2025.0210, DeepScaler



### 2025.0210, OREAL

[Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning](https://arxiv.org/abs/2502.06781)



### 2025.0217, LIMR

[LIMR: Less is More for RL Scaling](https://arxiv.org/abs/2502.11886)



### 2025.0220, Open-Reasoner-Zero



### 2025.0220, Logic-RL



### 2025.0303, VC-PPO

[What’s Behind PPO’s Collapse in Long-CoT? Value Optimization Holds the Secret](https://arxiv.org/pdf/2503.01491)



### 2025.0306, LCPO-L1

[L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning](https://www.arxiv.org/pdf/2503.04697)



### 2025.0310, MetaRL

[Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning](https://arxiv.org/pdf/2503.07572)



### 2025.0318, TOPR

[Tapered Off-Policy REINFORCE: Stable and efficient reinforcement learning for LLMs](https://arxiv.org/abs/2503.14286v2)



### 2025.0318, DAPO

| Project or Paper      | [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476v1) |
| :-------------------- | :----------------------------------------------------------- |
| GitHub                | [BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO) |
| Backbone Model        | Qwen2.5-32B                                                  |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) (AoPS Website) |
| Rollout Configuration | 256*16, temperature=1.0                                      |
| Reward Function       |                                                              |
| Loss Function         |                                                              |
| Benchmark             |                                                              |
| Core Insights         | Outperforming DeepSeek-R1-Zero-Qwen-32B with 50% training steps |
| Additional Notes      |                                                              |



### 2025.0321, Oat-Zero



### 2025.02xx, VLM-R1



### 2025.02xx, R1-V



### 2025.02xx, r1-vlm



### 2025.02xx, open-r1-multimodal



### 2025.02xx, R1-Searcher



### 2025.02xx, Med-RLVR



### 2025.02xx, SWE-RL



### 2025.02xx, LLM-R1



### 2025.xxxx, xxx



## Acknowledgment

We thank the following projects for providing valuable resources:

- https://github.com/huggingface/open-r1
- https://github.com/haoyangliu123/awesome-deepseek-r1
- https://github.com/JarvisUSTC/Awesome-DeepSeek-R1-Reproduction
- https://github.com/modelscope/awesome-deep-reasoning
- https://github.com/bruno686/Awesome-RL-based-LLM-Reasoning
- https://github.com/pemami4911/awesome-hyperparams



## Contributing



If you have any updates or improvements for this document, please feel free to submit a **Pull Request**. Thank you!



## Citation

