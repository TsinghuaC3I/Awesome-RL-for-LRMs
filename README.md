# Awesome Reasoning with RL Recipes ("Triple R")

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated collection covering models, datasets, reward designs, optimization methods, hyperparameters, empirical findings, theoretical insights, and everything about reasoning with reinforcement learning.

[TOC]

## Overview

**This collection covers recent progress in reinforcement learning for large language model reasoning, starting from 2025 in the timeline.”**



| Date      | Project               | Org                                      | HF Model                                                     | HF Dataset                                                   | Contribution                                                 | Note                                                         |
| --------- | --------------------- | ---------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2025.0102 | PRIME-RL              | Tsinghua <br /> UIUC <br /> Shang AI Lab | [Eurus-2-7B-PRIME](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME) <br />[Eurus-2-7B-PRIME-Zero](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME-Zero) | [Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data) | <details><summary>Read</summary>PRIME offers scalable Reinforcement Learning by using dense, token-level implicit rewards derived only from final outcomes. This bypasses costly step-by-step annotations, providing fine-grained feedback to improve sample efficiency and reasoning.</details> | [Paper](https://arxiv.org/abs/2502.01456)<br />[GitHub](https://github.com/PRIME-RL/PRIME)<br /> [More](#primerl) |
| 2025.0122 | DeepSeek-R1           | DeepSeek                                 | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) <br />[DeepSeek-R1-Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero) | ——                                                           | <details><summary>Read</summary>DeepSeek-R1's core contribution is demonstrating large-scale RL from scratch (600B+) without SFT, achieving emergent "aha moments" (self-reflective reasoning) and matching OpenAI o1's performance at 1/30 cost</details> | [Paper](https://arxiv.org/abs/2501.12948)<br />[GitHub](https://github.com/deepseek-ai/DeepSeek-R1/tree/main) |
| 2025.0122 | Kimi k1.5             | Kimi                                     | ——                                                           | ——                                                           | <details><summary>Read</summary>Kimi 1.5 introduces a simplified RL framework that leverages long-context scaling (128k tokens) and improved policy optimization (e.g., online mirror descent) to enhance reasoning and multimodal performance.</details> | [Paper](https://arxiv.org/abs/2501.12599)<br />[GitHub](https://github.com/MoonshotAI/Kimi-k1.5) |
| 2025.0124 | TinyZero              | Berkeley                                 | ——                                                           | [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | <details><summary>Read</summary>TinyZero's core contribution is demonstrating that smaller language models (e.g., 1.5B-3B parameters) can develop complex reasoning, search, and self-verification abilities through Reinforcement Learning, replicating capabilities of larger models like DeepSeek R1-Zero at extremely low cost (<$30).</details> | [Twitter](https://x.com/jiayi_pirate/status/1882839370505621655)<br />[GitHub](https://github.com/Jiayi-Pan/TinyZero) |
| 2025.0124 | Open-R1               | Huggingface                              | [OpenR1-Qwen-7B](https://huggingface.co/open-r1/OpenR1-Qwen-7B)<br />[OlympicCoder-7B](https://huggingface.co/open-r1/OlympicCoder-7B)<br />[OlympicCoder-32B](https://huggingface.co/open-r1/OlympicCoder-32B) | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)<br />[codeforces](https://huggingface.co/datasets/open-r1/codeforces) | Open-R1's core contribution is providing the first fully open-source replication and release of the DeepSeek R1-Zero Reinforcement Learning training pipeline. Its main insight or goal is to democratize access to these advanced RL techniques for enhancing LLM reasoning and planning. | [GitHub](https://github.com/huggingface/open-r1)             |
| 2025.0125 | simpleRL-reason       | HKUST                                    | [Qwen-2.5-Math-7B-SimpleRL-Zero](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero)<br />[Qwen-2.5-Math-7B-SimpleRL](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL) | [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | Researchers replicated the DeepSeek-R1-Zero and DeepSeek-R1 training using a 7B model with only 8K MATH examples, achieving strong results on complex mathematical reasoning. | [Paper](https://hkust-nlp.notion.site/simplerl-reason)<br />[GitHub](https://github.com/hkust-nlp/simpleRL-reason)<br />[More](#simplerl) |
| 2025.0126 | RAGEN                 | RAGEN-AI                                 | ——                                                           | ——                                                           | RAGEN introduces a RL framework to train reasoning-capable LLM agents for interactive, stochastic environments. Its core contribution is the Reasoning-Interaction Chain Optimization (RICO) algorithm, which jointly optimizes reasoning and action strategies by reinforcing entire trajectories. | [GitHub](https://github.com/RAGEN-AI/RAGEN)                  |
| 2025.0203 | Verifiers             | Independent                              | ——                                                           | ——                                                           | This repository contains a set of tools for reinforcement learning with LLMs in verifiable environments. It can be used for LLM Agent RL in verifable environments. | [GitHub](https://github.com/willccbb/verifiers)<br />[More](#verifiers) |
| 2025.0205 | Demystify-long-cot    | CMU                                      | ——                                                           | ——                                                           | The paper elucidates the role of RL in stabilizing and enhancing long CoT reasoning in LLMs, highlighting the necessity of reward shaping and verifiable reward signals for complex reasoning tasks. | [Paper](https://arxiv.org/abs/2502.03373)<br />[GitHub](https://github.com/eddycmu/demystify-long-cot)<br />[More](#demystify) |
| 2025.0210 | DeepScaler            | Agentica-Org                             | [DeepScaleR-1.5B-Preview](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview) | [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | DeepScaleR's core contribution is demonstrating that a small 1.5B parameter model, fine-tuned using scaled Reinforcement Learning (RL) and an iterative context lengthening scheme, can surpass the reasoning performance of larger, state-of-the-art models like OpenAI's O1-Preview on complex benchmarks (e.g., AIME math problems). | [Blog](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)<br />[GitHub](https://github.com/agentica-project/deepscaler) |
| 2025.0210 | Logic-RL              | MSRA && Ubiquant                         | ——                                                           | [knights-and-knaves](https://huggingface.co/datasets/K-and-K/knights-and-knaves) | The paper introduces Logic-RL, a rule-based reinforcement learning framework that enables large language models to develop o3-mini-level reasoning skills through training on logic puzzles. The reasoning capabilities can also be transferred to other domains like math. | [Paper](https://arxiv.org/pdf/2502.14768)<br />[GitHub](https://github.com/Unakar/Logic-RL)<br />[More](#logicrl) |
| 2025.0210 | OREAL                 | Shanghai AI Lab && SJTU && CUHK          | [OREAL-32B](https://huggingface.co/internlm/OREAL-32B)  [OREAL-7B](https://huggingface.co/internlm/OREAL-7B)<br />[OREAL-DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/internlm/OREAL-DeepSeek-R1-Distill-Qwen-7B)<br />[OREAL-32B-SFT](https://huggingface.co/internlm/OREAL-32B-SFT)<br />[OREAL-7B-SFT](https://huggingface.co/internlm/OREAL-7B-SFT) | [OREAL-RL-Prompts](https://huggingface.co/datasets/internlm/OREAL-RL-Prompts) | The paper introduces OREAL, a reinforcement learning framework for mathematical reasoning with binary feedback. It proves that behavior cloning on positive samples is sufficient for optimal learning and proposes reward reshaping for negative samples. A token-level reward model addresses sparse rewards in long reasoning chains. OREAL achieves state-of-the-art results on math benchmarks. | [Paper](https://arxiv.org/abs/2502.06781)<br /> [GitHub](https://github.com/InternLM/OREAL)<br /> [More](#oreal) |
| 2025.0217 | LIMR                  | SJTU                                     | [LIMR](https://huggingface.co/GAIR/LIMR)                     | [LIMR](https://huggingface.co/datasets/GAIR/LIMR)            | The paper challenges the assumption that scaling up RL training data inherently improves performance in language models, instead finding that a strategically selected subset of 1,389 samples can outperform a full 8,523-sample dataset. | [Paper](https://arxiv.org/pdf/2502.11886)<br />[GitHub](https://github.com/GAIR-NLP/LIMR)<br /> [More](#limr) |
| 2025.0218 | Open-Reasoner-Zero    | StepFun & Tsinghua                       | [Open-Reasoner-Zero-7B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-7B)<br />[Open-Reasoner-Zero-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B) | [ORZ-Math-57k](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/data) | The Open-Reasoner-Zero model has achieved notable performance, with Open-Reasoner-Zero-32B outperforming DeepSeek-R1-Zero-Qwen-32B on the GPQA Diamond benchmark while requiring significantly fewer training steps. | [Paper](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf) <br />[GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/)<br />   [More](#openreaon) |
| 2025.0225 | SWE-RL                | FAIR at Meta                             | ——                                                           | ——                                                           | SWE-RL enhances LLMs' code reasoning through RL using open-source software evolution data, achieving state-of-the-art results in software engineering tasks and demonstrating generalized reasoning capabilities beyond coding. | [Paper](https://arxiv.org/abs/2502.18449)<br />[GitHub](https://github.com/facebookresearch/swe-rl)<br />[More](#swerl) |
| 2025.0303 | VC-PPO                | Bytedance                                | ——                                                           | ——                                                           | VC-PPO (Value-Calibrated PPO) diagnoses PPO's collapse in long CoT tasks as stemming from value function inaccuracies (initialization bias and reward signal decay in long sequences). Its core contribution is modifying PPO with value pretraining and decoupled GAE for actor and critic. | [Paper](https://arxiv.org/abs/2503.01491)                    |
| 2025.0306 | LCPO-L1               | CMU                                      | [L1-Qwen-1.5B-Max](https://huggingface.co/l3lab/L1-Qwen-1.5B-Max)<br /> [L1-Qwen-1.5B-Exact](https://huggingface.co/l3lab/L1-Qwen-1.5B-Exact) | ——                                                           | L1 introduces Length Controlled Policy Optimization (LCPO), a RL method enabling precise control over a reasoning model's thinking time (output length) via prompt instructions. It shows that RL effectively controls reasoning duration and unexpectedly enhances even short-chain reasoning capabilities. | [Paper](https://arxiv.org/abs/2503.04697)<br />[GitHub](https://github.com/cmu-l3/l1) |
| 2025.0310 | MRT                   | CMU                                      | ——                                                           | ——                                                           | MRT (Mixed-Reality Trajectory Preferences) introduces a novel method for fine-tuning cooperative LLM agents. It effectively blends human preferences on real interaction trajectories with AI preferences on synthetic variations, improving data efficiency. This mixed-reality approach surpasses purely AI-driven feedback (RLAIF), especially for complex, multi-turn collaborative tasks. | [Paper](https://arxiv.org/pdf/2503.07572)<br />[Project](https://cohenqu.github.io/mrt.github.io/)<br />[GitHub](https://github.com/CMU-AIRe/MRT) |
| 2025.0318 | TOPR                  | Mila & Reliant AI                        | ——                                                           | ——                                                           | TOPR (Tapered Off-Policy REINFORCE) introduces a novel RL algorithm for fine-tuning LLMs. Its core contribution is using asymmetric, tapered importance sampling to modify REINFORCE, enabling stable and efficient off-policy learning. This allows reusing past data effectively without the instability often seen in other methods and without needing explicit KL regularization. | [Paper](https://arxiv.org/abs/2503.14286v2)                  |
| 2025.0318 | DAPO                  | Bytedance & Tsinghua Univ.               | ——                                                           | [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | DAPO algorithm introduces four key techniques (Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Shaping) for stable and efficient long-chain-of-thought RL training, surpassing previous SoTA results efficiently. | [Paper](https://arxiv.org/pdf/2503.14476)<br />[GitHub](https://github.com/BytedTsinghua-SIA/DAPO) |
| 2025.0319 | SWEET-RL              | Meta AI                                  | ——                                                           | [collaborative_agent_bench](https://huggingface.co/datasets/facebook/collaborative_agent_bench) | Sweet-RL introduces a novel RL algorithm for multi-turn collaborative reasoning LLM agents. Its core contribution is improving credit assignment across long interactions by using an asymmetric actor-critic structure where the critic leverages additional training-time information for step-wise evaluation. | [Paper](https://arxiv.org/abs/2503.15478)<br />[GitHub](https://github.com/facebookresearch/sweet_rl/tree/main) |
| 2025.0321 | Oat-Zero              | Sail-Sg                                  | [Qwen2.5-Math-7B-Oat-Zero](https://huggingface.co/sail/Qwen2.5-Math-7B-Oat-Zero)<br />[Qwen2.5-Math-1.5B-Oat-Zero](https://huggingface.co/sail/Qwen2.5-Math-1.5B-Oat-Zero)<br />[Llama-3.2-3B-Oat-Zero](https://huggingface.co/sail/Llama-3.2-3B-Oat-Zero) | [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | This work critically analyzes R1-Zero-like RL training. It reveals base model properties and GRPO algorithm biases (e.g., length bias) significantly impact outcomes. It contributes the efficient, unbiased Dr. GRPO algorithm and an open-source recipe/codebase for better understanding and reproduction. | [Paper](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf)<br />[GitHub](https://github.com/sail-sg/understand-r1-zero) |
| 2025.0321 | FastCuRL              | Tencent Hunyuan                          | [FastCuRL-1.5B-Preview](https://huggingface.co/Nickyang/FastCuRL-1.5B-Preview) | [FastCuRL](https://huggingface.co/datasets/Nickyang/FastCuRL) | FastCuRL introduces a simple, efficient Curriculum RL method for LLMs. Its core contribution uses target perplexity to dynamically scale the standard RL loss (like PPO), creating an effective curriculum without complex reward models or auxiliary components, enabling faster, more stable training. | [Paper](https://arxiv.org/abs/2503.17287)<br />[GitHub](https://github.com/nick7nlp/FastCuRL) |
|           |                       |                                          |                                                              |                                                              |                                                              |                                                              |
| 2025.0128 | Open-R1-MultiModal    |                                          | [Qwen2-VL-2B-GRPO-8k](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k)<br />[Qwen2-VL-7B-GRPO-8k](https://huggingface.co/lmms-lab/Qwen2-VL-7B-GRPO-8k) | [multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) | Open-R1-MultiModal provides an open-source replication of R1-Zero-like RL for Multimodal LLMs, aiming to enhance complex visual reasoning. It demonstrates the effectiveness of these RL techniques for boosting multimodal performance and promotes reproducibility in the field. | [GitHub](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) |
| 2025.0202 | R1-V                  |                                          | ——                                                           | [Clevr_CoGenT_TrainA_R1](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_R1) | R1-V applies RL, specifically RLV-Instruct, to fine-tune VLMs. It enhances complex visual reasoning and instruction-following capabilities in VLMs beyond standard supervised fine-tuning. | [Blog](https://deepagent.notion.site/rlvr-in-vlms)<br />[GitHub](https://github.com/Deep-Agent/R1-V) |
| 2025.0215 | VLM-R1                |                                          | [OVD](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321)<br />[Math](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-Math-0305) <br />[REC](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps) | ——                                                           | VLM-R1 applies R1-style RL to VLMs, improving stability and generalization on visual reasoning tasks. It shows that RL enhances VLM generalization beyond standard fine-tuning, achieving SOTA results, particularly on complex or out-of-domain benchmarks. | [Blog](https://om-ai-lab.github.io/index.html) <br />[GitHub](https://github.com/om-ai-lab/VLM-R1) |
| 2025.0227 | Med-RLVR              | Microsoft Research                       | ——                                                           | ——                                                           | The Med-RLVR framework demonstrates emergent medical reasoning via RL, achieving performance parity with SFT on in-distribution tasks and improving out-of-distribution generalization, all without explicit reasoning supervision, showcasing RL's potential in medicine. | [Paper](https://arxiv.org/pdf/2502.19655)                    |
| 2025.0303 | ReSearch              | Agent-RL                                 | ——                                                           | ——                                                           | The project train LLMs from scratch, utilizing RL with GRPO to learn to reason via search operations, without reliance on pre-existing reasoning frameworks or supervised data. | [GitHub](https://github.com/Agent-RL/ReSearch)               |
| 2025.0306 | R1-VLM                |                                          | ——                                                           | ——                                                           | R1-VLM enhances VLMs using RL, contributing significantly improved performance on complex visual reasoning tasks (spatial, counting, logic) where standard models falter. It shows that RL effectively unlocks advanced, multi-step reasoning capabilities specifically for vision-language understanding. | [Blog](https://www.groundlight.ai/blog/visual-reasoning-models)<br />[GitHub](https://github.com/groundlight/r1_vlm) |
| 2025.0310 | VisualThinker-R1-Zero | TurningPoint                             | [VisualThinker-R1-Zero](https://huggingface.co/turningpoint-ai/VisualThinker-R1-Zero) | ——                                                           | VisualThinker-R1-Zero adapts the R1-Zero RL paradigm (no supervised fine-tuning) to VLMs, achieving SoTa visual reasoning. It shows that complex visual reasoning can be effectively cultivated directly via RL on a base VLM, bypassing supervised data needs. | [Paper](https://arxiv.org/pdf/2503.05132) <br />[GitHub](https://github.com/turningpoint-ai/VisualThinker-R1-Zero) |
| 2025.0311 | LLM-R1                | CUHK&Ant Group                           | ——                                                           | ——                                                           | LLM-R1 contributes the RMAVO algorithm to stably enhance LLM reasoning using RL, preventing reward hacking and achieving SOTA results with smaller models via an open-source implementation. It shows that reward model assistance in value optimization is key for stable RL. | [Paper](https://arxiv.org/pdf/2503.07536)<br />[GitHub](https://github.com/TideDra/lmm-r1) |
| 2025.0311 | Vision-R1             | ECNU & Xiaohongshu                       | ——                                                           | [Vision-R1-cold](https://huggingface.co/datasets/Osilly/Vision-R1-cold) | Vision-R1 adapts the R1-Zero RL paradigm for VLMs, training them on visual reasoning chains. Its contribution is significantly boosting complex multimodal reasoning performance. It shows that RL applied to explicit reasoning steps effectively enhances VLM capabilities. | [Paper](https://arxiv.org/abs/2503.06749)<br />[GitHub](https://github.com/Osilly/Vision-R1) |
| 2025.0318 | R1-Searcher           | RUC                                      | [Llama-3.1-8B-instruct-RAG-RL](https://huggingface.co/XXsongLALA/Llama-3.1-8B-instruct-RAG-RL) <br />[Qwen-2.5-7B-base-RAG-RL](https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL) | [RAG-RL-Hotpotqa-with-2wiki](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki) | R1-Searcher enhances LLM reasoning via RL by training the model to perform adaptive model-based search during generation. This integration enables flexible thinking depth, improving reasoning efficiency and performance compared to fixed-step methods like R1-Zero. | [Paper](https://arxiv.org/pdf/2503.05592)<br />[GitHub](https://github.com/RUCAIBox/R1-Searcher) |



## Projects

### <div id="primerl">2025.0102, PRIME-RL</div>



### 2025.0122, DeepSeek-R1



### 2025.0122, Kimi k1.5

 [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599?)



### <div id="tinyzero">2025.0124, TinyZero</div>



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



### 2025.0225, SWE-RL



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



### 2025.02xx, VisualThinker-R1-Zero

[R1-Zero's "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model](https://arxiv.org/abs/2503.05132)



### 2025.02xx, R1-Searcher



### 2025.02xx, Med-RLVR



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

### <div id="template">202x.0x0x, Template</div>

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
| Core Insights         | (Empirical / Theoretical / Insightful Curves)            |
| Additional Notes      | (e.g., code snippet)                                     |



## Citation

```tex

```

