# Awesome RL Reasoning Recipes ("Triple R")

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated collection covering models, datasets, reward designs, optimization methods, hyperparameters, empirical findings, theoretical insights, and everything about reasoning with reinforcement learning.

## Contents

**The outline only includes part of the projects; for the latest projects, please jump to the bottom of the table ([GO](#latest)).**

- [Awesome RL Reasoning Recipes ("Triple R")](#awesome-rl-reasoning-recipes-triple-r)
  - [Contents](#contents)
  - [Overview](#overview)
    - [Large Language Models](#large-language-models)
    - [Multimodal and Applications](#multimodal-and-applications)
  - [Projects](#projects)
    - [Large Language Models](#large-language-models-1)
        - [2025.0102, PRIME-RL](#20250102-prime-rl)
        - [2025.0122, DeepSeek-R1](#20250122-deepseek-r1)
      - [2025.0122, Kimi k1.5](#20250122-kimi-k15)
      - [2025.0124, TinyZero](#20250124-tinyzero)
      - [2025.0125, SimpleRL](#20250125-simplerl)
      - [2025.0206, Demysitify-long-CoT](#20250206-demysitify-long-cot)
      - [2025.0210, DeepScaler](#20250210-deepscaler)
      - [2025.0210, Logic-RL](#20250210-logic-rl)
      - [2025.0210, OREAL](#20250210-oreal)
      - [2025.0217, LIMR](#20250217-limr)
      - [2025.0217, Open-Reasoner-Zero](#20250217-open-reasoner-zero)
      - [ 2025.0225, SWE-RL](#-20250225-swe-rl)
      - [2025.0303, VC-PPO ](#20250303-vc-ppo-)
      - [2025.0306, LCPO-L1 ](#20250306-lcpo-l1-)
      - [2025.0310, MetaRL ](#20250310-metarl-)
      - [2025.0318, TOPR ](#20250318-topr-)
      - [2025.0318, DAPO](#20250318-dapo)
      - [2025.0321, Oat-Zero](#20250321-oat-zero)
    - [Multimodal and Applications](#multimodal-and-applications-1)
      - [2025.0128, open-r1-multimodal](#20250128-open-r1-multimodal)
      - [2025.0202, R1-V](#20250202-r1-v)
      - [2025.0215, VLM-R1](#20250215-vlm-r1)
      - [2025.0306, r1-vlm](#20250306-r1-vlm)
      - [2025.0310, VisualThinker-R1-Zero](#20250310-visualthinker-r1-zero)
      - [2025.0310, MM-Eureka](#20250310-mm-eureka)
  - [Contributing](#contributing)
      - [202x.0x0x, Template](#202x0x0x-template)
  - [Citation](#citation)



## Overview

**This collection covers recent progress in reinforcement learning for large language model reasoning, starting from 2025 in the timeline.**


### Large Language Models

| Date      | Project            | Org                                | Intro                                                        | HF Model                                                     | HF Dataset                                                   | Contribution                                                 |
| --------- | ------------------ | ---------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2025.0102 | PRIME-RL           | THU & UIUC <br /> Shang AILab      | [Paper](https://arxiv.org/abs/2502.01456)<br />[GitHub](https://github.com/PRIME-RL/PRIME)<br /> [More](#primerl) | [Eurus-2-7B-PRIME](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME) <br />[Eurus-2-7B-PRIME-Zero](https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME-Zero) | [Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data) | <details><summary>Click</summary>PRIME offers scalable Reinforcement Learning by using dense, token-level implicit rewards derived only from final outcomes. This bypasses costly step-by-step annotations, providing fine-grained feedback to improve sample efficiency and reasoning.</details> |
| 2025.0122 | DeepSeek-R1        | DeepSeek                           | [Paper](https://arxiv.org/abs/2501.12948)<br />[GitHub](https://github.com/deepseek-ai/DeepSeek-R1/tree/main)<br />[More](#deepseek-r1) | [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) <br />[DeepSeek-R1-Zero](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero) | ——                                                           | <details><summary>Click</summary>DeepSeek-R1's core contribution is demonstrating large-scale RL from scratch (600B+) without SFT, achieving emergent "aha moments" (self-reflective reasoning) and matching OpenAI o1's performance at 1/30 cost</details> |
| 2025.0122 | Kimi k1.5          | Kimi                               | [Paper](https://arxiv.org/abs/2501.12599)<br />[GitHub](https://github.com/MoonshotAI/Kimi-k1.5)<br />[More](#kimi-k1.5) | ——                                                           | ——                                                           | <details><summary>Click</summary>Kimi 1.5 introduces a simplified RL framework that leverages long-context scaling (128k tokens) and improved policy optimization (e.g., online mirror descent) to enhance reasoning and multimodal performance.</details> |
| 2025.0124 | TinyZero           | Berkeley                           | [Twitter](https://x.com/jiayi_pirate/status/1882839370505621655)<br />[GitHub](https://github.com/Jiayi-Pan/TinyZero)<br />[More](#tinyzero) | ——                                                           | [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | <details><summary>Click</summary>TinyZero's core contribution is demonstrating that smaller language models (e.g., 1.5B-3B parameters) can develop complex reasoning, search, and self-verification abilities through Reinforcement Learning, replicating capabilities of larger models like DeepSeek R1-Zero at extremely low cost (<$30).</details> |
| 2025.0124 | Open-R1            | Huggingface                        | [GitHub](https://github.com/huggingface/open-r1)             | [OpenR1-Qwen-7B](https://huggingface.co/open-r1/OpenR1-Qwen-7B)<br />[OlympicCoder-7B](https://huggingface.co/open-r1/OlympicCoder-7B)<br />[OlympicCoder-32B](https://huggingface.co/open-r1/OlympicCoder-32B) | [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)<br />[codeforces](https://huggingface.co/datasets/open-r1/codeforces) | <details><summary>Click</summary>Open-R1's core contribution is providing the first fully open-source replication and release of the DeepSeek R1-Zero Reinforcement Learning training pipeline. Its main insight or goal is to democratize access to these advanced RL techniques for enhancing LLM reasoning and planning.</details> |
| 2025.0125 | simpleRL-reason    | HKUST                              | [Paper](https://hkust-nlp.notion.site/simplerl-reason)<br />[GitHub](https://github.com/hkust-nlp/simpleRL-reason)<br />[More](#simplerl) | [Qwen-2.5-Math-7B-SimpleRL-Zero](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero)<br />[Qwen-2.5-Math-7B-SimpleRL](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL) | [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | <details><summary>Click</summary>Researchers replicated the DeepSeek-R1-Zero and DeepSeek-R1 training using a 7B model with only 8K MATH examples, achieving strong results on complex mathematical reasoning.</details> |
| 2025.0126 | RAGEN              | RAGEN-AI                           | [GitHub](https://github.com/RAGEN-AI/RAGEN)                  | ——                                                           | ——                                                           | <details><summary>Click</summary>RAGEN introduces a RL framework to train reasoning-capable LLM agents for interactive, stochastic environments. Its core contribution is the Reasoning-Interaction Chain Optimization (RICO) algorithm, which jointly optimizes reasoning and action strategies by reinforcing entire trajectories.</details> |
| 2025.0203 | Verifiers          | Independent                        | [GitHub](https://github.com/willccbb/verifiers) | ——                                                           | ——                                                           | <details><summary>Click</summary>This repository contains a set of tools for reinforcement learning with LLMs in verifiable environments. It can be used for LLM Agent RL in verifable environments.</details> |
| 2025.0205 | Demystify-long-cot | CMU                                | [Paper](https://arxiv.org/abs/2502.03373)<br />[GitHub](https://github.com/eddycmu/demystify-long-cot)<br />[More](#demystify) | ——                                                           | ——                                                           | <details><summary>Click</summary>The paper elucidates the role of RL in stabilizing and enhancing long CoT reasoning in LLMs, highlighting the necessity of reward shaping and verifiable reward signals for complex reasoning tasks.</details> |
| 2025.0210 | DeepScaler         | Agentica-Org                       | [Blog](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)<br />[GitHub](https://github.com/agentica-project/deepscaler)<br />[More](#deepscaler) | [DeepScaleR-1.5B-Preview](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview) | [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset) | <details><summary>Click</summary>DeepScaleR's core contribution is demonstrating that a small 1.5B parameter model, fine-tuned using scaled Reinforcement Learning (RL) and an iterative context lengthening scheme, can surpass the reasoning performance of larger, state-of-the-art models like OpenAI's O1-Preview on complex benchmarks (e.g., AIME math problems).</details> |
| 2025.0210 | Logic-RL           | MSRA & Ubiquant                    | [Paper](https://arxiv.org/pdf/2502.14768)<br />[GitHub](https://github.com/Unakar/Logic-RL)<br />[More](#logicrl) | ——                                                           | [knights-and-knaves](https://huggingface.co/datasets/K-and-K/knights-and-knaves)   [knights-and-knaves-ZH](https://huggingface.co/datasets/Trae1ounG/knights-and-knaves-ZH)  | <details><summary>Click</summary>The paper introduces Logic-RL, a rule-based reinforcement learning framework that enables large language models to develop o3-mini-level reasoning skills through training on logic puzzles. The reasoning capabilities can also be transferred to other domains like math.</details> |
| 2025.0210 | OREAL              | Shanghai AI Lab <br /> SJTU & CUHK | [Paper](https://arxiv.org/abs/2502.06781)<br /> [GitHub](https://github.com/InternLM/OREAL)<br /> [More](#oreal) | [OREAL-32B](https://huggingface.co/internlm/OREAL-32B)  [OREAL-7B](https://huggingface.co/internlm/OREAL-7B)<br />[OREAL-DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/internlm/OREAL-DeepSeek-R1-Distill-Qwen-7B)<br />[OREAL-32B-SFT](https://huggingface.co/internlm/OREAL-32B-SFT)<br />[OREAL-7B-SFT](https://huggingface.co/internlm/OREAL-7B-SFT) | [OREAL-RL-Prompts](https://huggingface.co/datasets/internlm/OREAL-RL-Prompts) | <details><summary>Click</summary>The paper introduces OREAL, a reinforcement learning framework for mathematical reasoning with binary feedback. It proves that behavior cloning on positive samples is sufficient for optimal learning and proposes reward reshaping for negative samples. A token-level reward model addresses sparse rewards in long reasoning chains. OREAL achieves state-of-the-art results on math benchmarks.</details> |
| 2025.0217 | LIMR               | SJTU                               | [Paper](https://arxiv.org/pdf/2502.11886)<br />[GitHub](https://github.com/GAIR-NLP/LIMR)<br /> [More](#limr) | [LIMR](https://huggingface.co/GAIR/LIMR)                     | [LIMR](https://huggingface.co/datasets/GAIR/LIMR)            | <details><summary>Click</summary>The paper challenges the assumption that scaling up RL training data inherently improves performance in language models, instead finding that a strategically selected subset of 1,389 samples can outperform a full 8,523-sample dataset.</details> |
| 2025.0218 | Open-Reasoner-Zero | StepFun & THU                      | [Paper](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf) <br />[GitHub](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/)<br />   [More](#openreaon-zero) | [Open-Reasoner-Zero-7B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-7B)<br />[Open-Reasoner-Zero-32B](https://huggingface.co/Open-Reasoner-Zero/Open-Reasoner-Zero-32B) | [ORZ-Math-57k](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/data) | <details><summary>Click</summary>The Open-Reasoner-Zero model has achieved notable performance, with Open-Reasoner-Zero-32B outperforming DeepSeek-R1-Zero-Qwen-32B on the GPQA Diamond benchmark while requiring significantly fewer training steps.</details> |
| 2025.0225 | SWE-RL             | FAIR at Meta                       | [Paper](https://arxiv.org/abs/2502.18449)<br />[GitHub](https://github.com/facebookresearch/swe-rl)<br />[More](#swerl) | ——                                                           | ——                                                           | <details><summary>Click</summary>SWE-RL enhances LLMs' code reasoning through RL using open-source software evolution data, achieving state-of-the-art results in software engineering tasks and demonstrating generalized reasoning capabilities beyond coding.</details> |
| 2025.0303 | VC-PPO             | Bytedance                          | [Paper](https://arxiv.org/abs/2503.01491)<br />[More](#vcppo) | ——                                                           | ——                                                           | <details><summary>Click</summary>VC-PPO (Value-Calibrated PPO) diagnoses PPO's collapse in long CoT tasks as stemming from value function inaccuracies (initialization bias and reward signal decay in long sequences). Its core contribution is modifying PPO with value pretraining and decoupled GAE for actor and critic.</details> |
| 2025.0306 | LCPO-L1            | CMU                                | [Paper](https://arxiv.org/abs/2503.04697)<br />[GitHub](https://github.com/cmu-l3/l1)<br />[More](#lcpol1) | [L1-Qwen-1.5B-Max](https://huggingface.co/l3lab/L1-Qwen-1.5B-Max)<br /> [L1-Qwen-1.5B-Exact](https://huggingface.co/l3lab/L1-Qwen-1.5B-Exact) | ——                                                           | <details><summary>Click</summary>L1 introduces Length Controlled Policy Optimization (LCPO), a RL method enabling precise control over a reasoning model's thinking time (output length) via prompt instructions. It shows that RL effectively controls reasoning duration and unexpectedly enhances even short-chain reasoning capabilities.</details> |
| 2025.0310 | MRT                | CMU                                | [Paper](https://arxiv.org/pdf/2503.07572)<br />[Project](https://cohenqu.github.io/mrt.github.io/)<br />[GitHub](https://github.com/CMU-AIRe/MRT) | ——                                                           | ——                                                           | <details><summary>Click</summary>MRT (Mixed-Reality Trajectory Preferences) introduces a novel method for fine-tuning cooperative LLM agents. It effectively blends human preferences on real interaction trajectories with AI preferences on synthetic variations, improving data efficiency. This mixed-reality approach surpasses purely AI-driven feedback (RLAIF), especially for complex, multi-turn collaborative tasks.</details> |
| 2025.0318 | TOPR               | Mila & Reliant AI                  | [Paper](https://arxiv.org/abs/2503.14286v2)<br />[More](#topr) | ——                                                           | ——                                                           | <details><summary>Click</summary>TOPR (Tapered Off-Policy REINFORCE) introduces a novel RL algorithm for fine-tuning LLMs. Its core contribution is using asymmetric, tapered importance sampling to modify REINFORCE, enabling stable and efficient off-policy learning. This allows reusing past data effectively without the instability often seen in other methods and without needing explicit KL regularization.</details> |
| 2025.0318 | DAPO               | Bytedance <br /> THU               | [Paper](https://arxiv.org/pdf/2503.14476)<br />[GitHub](https://github.com/BytedTsinghua-SIA/DAPO)<br />[More](#dapo) | ——                                                           | [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) | <details><summary>Click</summary>DAPO algorithm introduces four key techniques (Clip-Higher, Dynamic Sampling, Token-Level Loss, Overlong Shaping) for stable and efficient long-chain-of-thought RL training, surpassing previous SoTA results efficiently.</details> |
| 2025.0319 | SWEET-RL           | Meta AI                            | [Paper](https://arxiv.org/abs/2503.15478)<br />[GitHub](https://github.com/facebookresearch/sweet_rl/tree/main) | ——                                                           | [collaborative_agent_bench](https://huggingface.co/datasets/facebook/collaborative_agent_bench) | <details><summary>Click</summary>Sweet-RL introduces a novel RL algorithm for multi-turn collaborative reasoning LLM agents. Its core contribution is improving credit assignment across long interactions by using an asymmetric actor-critic structure where the critic leverages additional training-time information for step-wise evaluation.</details> |
| 2025.0321 | Oat-Zero           | Sail-Sg                            | [Paper](https://arxiv.org/abs/2503.20783)<br />[GitHub](https://github.com/sail-sg/understand-r1-zero)<br />[More](#oat-zero) | [Qwen2.5-Math-7B-Oat-Zero](https://huggingface.co/sail/Qwen2.5-Math-7B-Oat-Zero)<br />[Qwen2.5-Math-1.5B-Oat-Zero](https://huggingface.co/sail/Qwen2.5-Math-1.5B-Oat-Zero)<br />[Llama-3.2-3B-Oat-Zero](https://huggingface.co/sail/Llama-3.2-3B-Oat-Zero) | [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | <details><summary>Click</summary>This work critically analyzes R1-Zero-like RL training. It reveals base model properties and GRPO algorithm biases (e.g., length bias) significantly impact outcomes. It contributes the efficient, unbiased Dr. GRPO algorithm and an open-source recipe/codebase for better understanding and reproduction.</details> |
| 2025.0321 | FastCuRL           | Tencent Hunyuan                    | [Paper](https://arxiv.org/abs/2503.17287)<br />[GitHub](https://github.com/nick7nlp/FastCuRL) | [FastCuRL-1.5B-Preview](https://huggingface.co/Nickyang/FastCuRL-1.5B-Preview) | [FastCuRL](https://huggingface.co/datasets/Nickyang/FastCuRL) | <details><summary>Click</summary>FastCuRL introduces a simple, efficient Curriculum RL method for LLMs. Its core contribution uses target perplexity to dynamically scale the standard RL loss (like PPO), creating an effective curriculum without complex reward models or auxiliary components, enabling faster, more stable training.</details> |
| 2025.0401 | Z1           | THU                    | [Paper](https://arxiv.org/abs/2504.00810)<br />[GitHub](https://github.com/efficientscaling/Z1) | [Z1-7B](https://huggingface.co/efficientscaling/Z1-7B) | [Z1-Code-Reasoning-107K](https://huggingface.co/datasets/efficientscaling/Z1-Code-Reasoning-107K) | <details><summary>Click</summary>This paper proposes training LLMs on code-related reasoning trajectories using a curated dataset and a "Shifted Thinking Window" technique. This allows models to reduce excessive thinking tokens, achieving efficient test-time scaling and generalizing reasoning abilities.</details> |
| <div id="latest">2025.0x0x</div> |             |                      | [Paper]()<br />[GitHub]() | [hf models]() | [hf datasets]() | <details><summary>Click</summary>insights and contributions about RL for reasoning within 30 words.</details> |


### Multimodal and Applications
| Date      | Project               | Org                | Intro                                                        | HF Model                                                     | HF Dataset                                                   | Contribution                                                 |
| --------- | --------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2025.0128 | Open-R1-MultiModal    | LLMs Lab           | [GitHub](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)<br />[More](#open-r1-mm) | [Qwen2-VL-2B-GRPO-8k](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k)<br />[Qwen2-VL-7B-GRPO-8k](https://huggingface.co/lmms-lab/Qwen2-VL-7B-GRPO-8k) | [multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) | <details><summary>Click</summary>Open-R1-MultiModal provides an open-source replication of R1-Zero-like RL for Multimodal LLMs, aiming to enhance complex visual reasoning. It demonstrates the effectiveness of these RL techniques for boosting multimodal performance and promotes reproducibility in the field.</details> |
| 2025.0202 | R1-V                  | Deep Agent         | [Blog](https://deepagent.notion.site/rlvr-in-vlms)<br />[GitHub](https://github.com/Deep-Agent/R1-V)<br />[More](#r1v) | ——                                                           | [Clevr_CoGenT_TrainA_R1](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_R1) | <details><summary>Click</summary>R1-V applies RL, specifically RLV-Instruct, to fine-tune VLMs. It enhances complex visual reasoning and instruction-following capabilities in VLMs beyond standard supervised fine-tuning.</details> |
| 2025.0215 | VLM-R1                | OmAI Lab           | [Blog](https://om-ai-lab.github.io/index.html) <br />[GitHub](https://github.com/om-ai-lab/VLM-R1)<br />[More](#vlmr1) | [OVD](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321)<br />[Math](https://huggingface.co/omlab/VLM-R1-Qwen2.5VL-3B-Math-0305) <br />[REC](https://huggingface.co/omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps) | ——                                                           | <details><summary>Click</summary>VLM-R1 applies R1-style RL to VLMs, improving stability and generalization on visual reasoning tasks. It shows that RL enhances VLM generalization beyond standard fine-tuning, achieving SOTA results, particularly on complex or out-of-domain benchmarks.</details> |
| 2025.0227 | Med-RLVR              | Microsoft Research | [Paper](https://arxiv.org/pdf/2502.19655)<br />[More](#medrlvr) | ——                                                           | ——                                                           | <details><summary>Click</summary>The Med-RLVR framework demonstrates emergent medical reasoning via RL, achieving performance parity with SFT on in-distribution tasks and improving out-of-distribution generalization, all without explicit reasoning supervision, showcasing RL's potential in medicine.</details> |
| 2025.0303 | ReSearch              | Agent-RL           | [GitHub](https://github.com/Agent-RL/ReSearch)<br />[More](#research) | ——                                                           | ——                                                           | <details><summary>Click</summary>The project train LLMs from scratch, utilizing RL with GRPO to learn to reason via search operations, without reliance on pre-existing reasoning frameworks or supervised data.</details> |
| 2025.0306 | R1-VLM                | GroundLight        | [Blog](https://www.groundlight.ai/blog/visual-reasoning-models)<br />[GitHub](https://github.com/groundlight/r1_vlm)<br />[More](#r1-vlm) | ——                                                           | ——                                                           | <details><summary>Click</summary>R1-VLM enhances VLMs using RL, contributing significantly improved performance on complex visual reasoning tasks (spatial, counting, logic) where standard models falter. It shows that RL effectively unlocks advanced, multi-step reasoning capabilities specifically for vision-language understanding.</details> |
| 2025.0310 | VisualThinker-R1-Zero | TurningPoint       | [Paper](https://arxiv.org/pdf/2503.05132) <br />[GitHub](https://github.com/turningpoint-ai/VisualThinker-R1-Zero)<br />[More](#visual-r1-zero) | [VisualThinker-R1-Zero](https://huggingface.co/turningpoint-ai/VisualThinker-R1-Zero) | ——                                                           | <details><summary>Click</summary>VisualThinker-R1-Zero adapts the R1-Zero RL paradigm (no supervised fine-tuning) to VLMs, achieving SoTa visual reasoning. It shows that complex visual reasoning can be effectively cultivated directly via RL on a base VLM, bypassing supervised data needs.</details> |
| 2025.0310 | MM-EUREKA | Shanghai AI Lab & SJTU & HKU       | [Paper](https://arxiv.org/pdf/2503.07365) <br />[Github](https://github.com/ModalMinds/MM-EUREKA) <br /> [More](#mm-eureka) | [MM-Eureka-Qwen-7B](https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B) | [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)       | <details><summary>Click</summary>MM-EUREKA reproduces key characteristics of text-based RL systems like DeepSeek-R1 in the multimodal space, which demonstrates that both instruction-tuned and pre-trained models can develop strong multimodal reasoning capabilities through rule-based RL without supervised fine-tuning, showing superior data efficiency compared to alternative approaches. </details> |
| 2025.0311 | LLM-R1                | CUHK & Ant Group   | [Paper](https://arxiv.org/pdf/2503.07536)<br />[GitHub](https://github.com/TideDra/lmm-r1) | ——                                                           | ——                                                           | <details><summary>Click</summary>LLM-R1 contributes the RMAVO algorithm to stably enhance LLM reasoning using RL, preventing reward hacking and achieving SOTA results with smaller models via an open-source implementation. It shows that reward model assistance in value optimization is key for stable RL.</details> |
| 2025.0311 | Vision-R1             | ECNU & Xiaohongshu | [Paper](https://arxiv.org/abs/2503.06749)<br />[GitHub](https://github.com/Osilly/Vision-R1) | ——                                                           | [Vision-R1-cold](https://huggingface.co/datasets/Osilly/Vision-R1-cold) | <details><summary>Click</summary>Vision-R1 adapts the R1-Zero RL paradigm for VLMs, training them on visual reasoning chains. Its contribution is significantly boosting complex multimodal reasoning performance. It shows that RL applied to explicit reasoning steps effectively enhances VLM capabilities.</details> |
| 2025.0318 | R1-Searcher           | RUC                | [Paper](https://arxiv.org/pdf/2503.05592)<br />[GitHub](https://github.com/RUCAIBox/R1-Searcher) | [Llama-3.1-8B-instruct-RAG-RL](https://huggingface.co/XXsongLALA/Llama-3.1-8B-instruct-RAG-RL) <br />[Qwen-2.5-7B-base-RAG-RL](https://huggingface.co/XXsongLALA/Qwen-2.5-7B-base-RAG-RL) | [RAG-RL-Hotpotqa](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki) | <details><summary>Click</summary>R1-Searcher enhances LLM reasoning via RL by training the model to perform adaptive model-based search during generation. This integration enables flexible thinking depth, improving reasoning efficiency and performance compared to fixed-step methods like R1-Zero.</details> |



## Projects

### Large Language Models

##### <div id="primerl">2025.0102, PRIME-RL</div>

| Project or Paper      | [Process Reinforcement through Implicit Rewards](https://arxiv.org/abs/2502.01456) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [PRIME-RL/PRIME](https://github.com/PRIME-RL/PRIME)          |
| Backbone Model        | Qwen2.5-Math-7B-Base                                         |
| RL Algorithm          | PPO/REINFORCE++/RLOO/GRPO + Online PRM                       |
| Training Dataset      | [PRIME-RL/Eurus-2-RL-Data](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data), 150K |
| Rollout Configuration | 256 prompts * 4 responses Online Prompt Filtering (Accuracy in [0.2,0.8]) |
| Reward Function       | Rule-based Rewards + Implicit Process Rewards                |
| Policy Optimization   | PPO loss, without KL loss                                    |
| Benchmark             | GPT-4o level on AIME 2024, AMC, MATH-500, Minerva Math, OlympiadBench, LeetCode, and LiveCodeBench |
| Core Insights         | Implicit PRM efficiently addresses reward sparsity, distribution shift, and scalability by directly learning token-level rewards within a language model framework, eliminating the need for separate value models or prior training. |
| Additional Notes      |                                                              |



##### <div id="deepseek-r1">2025.0122, DeepSeek-R1</div>

| Project or Paper      | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2501.12948) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) |
| Backbone Model        | DeepSeek-V3-Base                                             |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | Unclear                                                      |
| Rollout Configuration | 4~64 samples for each prompt, temperature = 0.6              |
| Reward Function       | Rule-based Rewards                                           |
| Policy Optimization   | vanilla GRPO Loss                                            |
| Benchmark             | OpenAI-o1 level on AIME 2024, Codeforces, GPQA Diamond, MATH-500, MMLU, SWE-bench Verified. |
| Core Insights         | RL can boost LLMs' reasoning. DeepSeek - R1 series models prove effective, and distilling reasoning to small models works, while highlighting challenges in other methods. |
| Additional Notes      |                                                              |



#### <div id="kimi-k1.5">2025.0122, Kimi k1.5</div>

| Project or Paper      | [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://arxiv.org/pdf/2501.12599) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [MoonshotAI/Kimi-k1.5](https://github.com/MoonshotAI/Kimi-k1.5) |
| Backbone Model        | Kimi k-series model (closed source)                          |
| RL Algorithm          | Online Policy Mirror Decent/Length Penalty Reward/Curriculum Sampling/Prioritized Sampling/Chain-of-Thought RM/Long2short RL |
| Training Dataset      | Code: 1000 contest problem; </br>Math: 800k in-context learning/800k chain-of-thought data; </br> Vision: unknown number of real-world/synthetic visual reasoning/text-rendered data |
| Rollout Configuration | None                                                         |
| Reward Function       | Outcome Reward+Length Penalty Reward                         |
| Policy Optimization   | Online Policy Mirror Decent                                  |
| Benchmark             | Matching OpenAI’s o1 on AIME/MATH500/Codeforces/MathVista    |
| Core Insights         | Effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results, outperforming existing short-CoT models. |
| Additional Notes      |                                                              |



#### <div id="tinyzero">2025.0124, TinyZero</div>

| Project or Paper      | not applicable                                               |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [Tiny-Zero](https://github.com/Jiayi-Pan/TinyZero) |
| Backbone Model        | QWen-2.5-3B                                                  |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | countdown                                                    |
| Rollout Configuration | 1024                                                         |
| Reward Function       | 0/1 reward                                                   |
| Policy Optimization   | vanilla GRPO: (KL Loss; Length Penalty; Token-level loss)    |
| Benchmark             | test set of countdown                                        |
| Core Insights         | Aha moment can be reproducible on puzzle-style data.         |
| Additional Notes      |                                                              |



#### <div id="simplerl">2025.0125, SimpleRL</div>

| Project or Paper      | [7B Model and 8K Examples: Emerging Reasoning with Reinforcement Learning is Both Effective and Efficient](https://hkust-nlp.notion.site/simplerl-reason) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) |
| Backbone Model        | Qwen2.5-Math-7B                                              |
| RL Algorithm          | PPO based on OpenRLHF                                        |
| Training Dataset      | [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math), 8k Level3-Level5 |
| Rollout Configuration | 128 prompts * 8 responses; Temperature = 0.6                 |
| Reward Function       | Rule-based Rewards                                           |
| Policy Optimization   | PPO loss with 0.01 KL coefficient                            |
| Benchmark             | GPT-4o level on AIME2024, AMC2023, College Math, Gaokao2023en, GSM8k, MATH500, Minerva Math, and OlympiadBench. |
| Core Insights         | Long Chain-of-Thought (CoT) and self-reflection can emerge on a 7B model with only few high-quality examples with rule-based rewards only. |
| Additional Notes      |                                                              |



#### <div id="demystify">2025.0206, Demysitify-long-CoT</div> 

| Project or Paper      | [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/pdf/2502.03373) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [eddycmu/demystify-long-cot](https://github.com/eddycmu/demystify-long-cot) |
| Backbone Model        | Qwen2.5-Math-7B Llama-3.1-8B                                 |
| RL Algorithm          | PPO (OpenRLHF)                                               |
| Training Dataset      | 7500 training samples of MATH                                |
| Rollout Configuration | 512 prompts * 8 responses; temperature=0.7, top-p=0.95 Context Length Prompt=2048, Gen=14384 tokens |
| Reward Function       | Rule-based Reward + Cosine Length Reward + Repetition Penalty Reward |
| Policy Optimization   | KL coefficy=0.01, gamma=1, lambda=1                          |
| Benchmark             | MATH, AIME, TheoremQA, MMLU-Pro-1k                           |
| Core Insights         | Reward shaping can be used to stabilize and control CoT length while improving accuracy. Cosine reward can be tuned to incentivize various length scaling behaviors, length rewards will be hacked with enough compute. But this can be mitigated using a repetition penalty. |
| Additional Notes      |                                                              |





#### <div id="deepscaler">2025.0210, DeepScaler</div>

| Project or Paper      | [DeepScaleR: Surpassing O1-Preview with a 1.5B Model by Scaling RL](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [deepscaler](https://github.com/agentica-project/deepscaler) |
| Backbone Model        | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B                    |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [Omni-MATH](https://omni-math.github.io/) and [Still](https://github.com/RUCAIBox/Slow_Thinking_with_LLMs) |
| Rollout Configuration | 8K->16K->24K                                                 |
| Reward Function       | 0/1 reward                                                   |
| Policy Optimization   | vanilla GRPO: (KL Loss; Length Penalty; Token-level loss)    |
| Benchmark             | AIME 2024/ MATH 500 /AMC 2023 / Minerva Math/ OlympiadBench  |
| Core Insights         | 1. RL scaling can manifest in small models as well. 2. Iterative lengthening enables more effective length scaling. |
| Additional Notes      | Iteratively increasing the max context for RL training.      |



#### <div id="logicrl">2025.0210, Logic-RL</div> 

| Project or Paper      | [Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning](https://arxiv.org/abs/2502.14768) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [Unakar/Logic-RL](https://github.com/Unakar/Logic-RL)        |
| Backbone Model        | Qwen2.5-Math-7B / Qwen2.5-7B-Instruct                        |
| RL Algorithm          | REINFORCE++                                                  |
| Training Dataset      | [Knights and Knaves (K&K) puzzles](https://huggingface.co/datasets/K-and-K/knights-and-knaves), 6.2k |
| Rollout Configuration | 8 prompts * 8 responses; Temperature = 0.7                   |
| Reward Function       | Rule-based Rewards                                           |
| Policy Optimization   | REINFORCE++ loss with 0.001 unbiased KL coefficient.         |
| Benchmark             | o3-mini-high level on K&K logic puzzle                       |
| Core Insights         | With simple REINFORCE++ with KL loss, 7B model develops advanced reasoning skills that are absent from the logic corpus and generates to other tasks like math. |
| Additional Notes      |  |


#### <div id="oreal">2025.0210, OREAL</div>

| Project or Paper      | [Exploring the Limit of Outcome Reward for Learning Mathematical Reasoning](https://arxiv.org/abs/2502.06781) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [InternLM/OREAL](https://github.com/InternLM/OREA)           |
| Backbone Model        | Qwen2.5-7B / Qwen2.5-32B                                     |
| RL Algorithm          |   |
| Training Dataset      | [OREAL-RL-Prompts](https://huggingface.co/datasets/internlm/OREAL-RL-Prompts), 4k |
| Rollout Configuration | 64 prompts * 16 responses; Temprature = 1.0; Online Accuracy Filtering; Retain only one correct and wrong solutions |
| Reward Function       | Outcome Reward Signal by rule-based verifier and Qwen2.5-72B-Instruct + Token level Reward |
| Policy Optimization   | OREAL loss with 0.01 KL coefficient                          |
| Benchmark             | R1-level on MATH500, AIME2024, AIME2025-I, LiveMath, Olympiad |
| Core Insights         | Behavior cloning on positive samples is sufficient for optimal learning and reward reshaping for negative samples is needed for consistent gradient estimation. A token-level reward model can be trained to addresses sparse rewards. |
| Additional Notes      | |



#### <div id="limr">2025.0217, LIMR</div>

| Project or Paper      | [LIMR: Less is More for RL Scaling](https://arxiv.org/abs/2502.11886) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [GAIR-NLP/LIMR](https://github.com/GAIR-NLP/LIMR)            |
| Backbone Model        | Qwen2.5-Math-7B                                              |
| RL Algorithm          | PPO based on OpenRLHF                                        |
| Training Dataset      | [LIMR](https://huggingface.co/datasets/GAIR/LIMR), 1.4k      |
| Rollout Configuration | 1024 prompts * 8 responses; Temperature = 1.2                |
| Reward Function       | Rule-based Rewards                                           |
| Policy Optimization   | PPO loss with 0.01 KL coefficient                            |
| Benchmark             | GPT-4o level on AIME2024, MATH500, AMC2023                   |
| Core Insights         | Precisely selected data may be the key to unlocking the enhanced reasoning capabilities of LLMs. |
| Additional Notes      | The author uses the trained model's average reward curve as a reference for measuring sample effectiveness. |



#### <div id="openreason-zero">2025.0217, Open-Reasoner-Zero</div>

| Project or Paper      | [Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Model](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/main/ORZ_paper.pdf) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [Open-Reasoner-Zero/Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) |
| Backbone Model        | Qwen2.5-7B, Qwen2.5-32B                                      |
| RL Algorithm          | PPO based on OpenRLHF                                        |
| Training Dataset      | [ORZ-MATH](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main/data), 57k |
| Rollout Configuration | 128 prompts * 64 responses; Temperature = 1.0                |
| Reward Function       | Rule-based Rewards                                           |
| Policy Optimization   | PPO loss without KL loss                                     |
| Benchmark             | GPT-4o level on GPQA-Diamond, MATH500, AIME2024              |
| Core Insights         | Vanilla PPO with GAE and rule-based rewards without KL loss is sufficient to sclae up response length and benchmark performance. |
| Additional Notes      |                                                              |



#### <div id="swerl"> 2025.0225, SWE-RL</div> 

| Project or Paper      | [SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [facebookresearch/swe-rl](https://github.com/facebookresearch/swe-rl) |
| Backbone Model        | Llama-3.3-70B-Instruct                                       |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | Publicly available repositories                              |
| Rollout Configuration | 32 prompts * 16 rollouts                                     |
| Reward Function       | Similarity Function (`difflib.SequenceMatcher`)              |
| Policy Optimization   | Normalized Rewards for Advantage Calculation                 |
| Benchmark             | GPT-4o level on SWE-bench Verified (41%)                     |
| Core Insights         | SWE-RL enhances LLMs' code reasoning through RL using open-source software evolution data, achieving state-of-the-art results in software engineering tasks and demonstrating generalized reasoning capabilities beyond coding. |
| Additional Notes      |                                                              |



#### <div id="vcppo">2025.0303, VC-PPO </div>

| Project or Paper      | [What’s Behind PPO’s Collapse in Long-CoT? Value Optimization Holds the Secret](https://arxiv.org/pdf/2503.01491) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | N/A                                                          |
| Backbone Model        | Qwen2.5-32B-Base                                             |
| RL Algorithm          | VC-PPO                                                       |
| Training Dataset      | A compilation of questions from all past AIME competitions (excluding the last two years), supplemented with artificially constructed challenging mathematical problems. |
| Rollout Configuration | N/A                                                          |
| Reward Function       | Rule-based Rewards                                           |
| Policy Optimization   | PPO loss, Value Estimate with Decoupled-GAE                  |
| Benchmark             | AIME 2024, GPQA, CodeForces                                  |
| Core Insights         | VC-PPO addresses PPO’s failure in long CoT tasks by pretraining the value model to correct initialization bias and decoupling GAE between the actor and critic to mitigate reward signal decay. |
| Additional Notes      |                                                              |



#### <div id="lcpol1">2025.0306, LCPO-L1 </div>


| Project or Paper      | [L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning ](https://www.arxiv.org/pdf/2503.04697) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [cmu-l3/l1](https://github.com/cmu-l3/l1)                    |
| Backbone Model        | DeepScaleR-1.5B-Preview                                      |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [DeepScaleR-Preview-Dataset](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset), 40k |
| Rollout Configuration | 128 prompts                                                  |
| Reward Function       | correctness reward + length penalty                          |
| Policy Optimization   | GRPO loss + length penalty loss                              |
| Benchmark             | AIME 2025, MATH, AMC, Olympiad-Bench, GPQA, LSAT, MMLU       |
| Core Insights         | Address the uncontrolled reasoning length issue in language models. It uses reinforcement learning to optimize for both accuracy and adherence to user - specified length constraints. By training models with a reward function that combines correctness and length - related terms, LCPO enables precise length control, allowing for a better trade - off between computational cost and accuracy in various reasoning tasks. |
| Additional Notes      |                                                              |



#### <div id="metarl">2025.0310, MetaRL </div>

| Project or Paper      | [Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning](https://arxiv.org/pdf/2503.07572) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [CMU-AIRe/MRT](https://github.com/CMU-AIRe/MRT)              |
| Backbone Model        | DeepSeek-R1-Distill-Qwen-32B/7B/1.5B / DeepScaleR-1.5B-Preview /  Llama-3.1-8B/3B-Instruct |
| RL Algorithm          | MRT                                                          |
| Training Dataset      | None                                                         |
| Rollout Configuration | 256 prompts * 4 responses, temperature = 0.7                 |
| Reward Function       | 0/1 reward + dense reward                                    |
| Policy Optimization   | SFT Loss + Dense Reward Bonus Loss                           |
| Benchmark             | AIME 2024, AIME 2025, AMC 2023, MinervaMATH, MATH500         |
| Core Insights         | Treating test - time computation of LLMs as a meta - RL problem. It uses a progress - based reward function to guide the model, and optimizes the policy to balance exploration and exploitation, aiming to improve the efficiency and performance of LLMs at test time. |
| Additional Notes      |                                                              |



#### <div id="topr">2025.0318, TOPR </div>

| Project or Paper      | [Tapered Off-Policy REINFORCE Stable and efficient reinforcement learning for LLMs](https://arxiv.org/pdf/2503.14286v2) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [None](http://None)                                          |
| Backbone Model        | Llama 3 8B/70B                                               |
| RL Algorithm          | Tapered Off-Policy REINFORCE (TOPR)                          |
| Training Dataset      | GSM8K and MATH                                               |
| Rollout Configuration | 64/8 solutions each question for GSM8k/MATH, respectively    |
| Reward Function       | Implicit Reward as DPO (contrastive learning with preference data |
| Policy Optimization   | Optimization with off-policy samples and without KL penalty  |
| Benchmark             | The performance of 8B language models can match with 70B-parameter model's on GSM8K and MATH |
| Core Insights         | Properly leveraging positive and negative examples alike in the off-policy regime simultaneously increases test-time accuracy and training data efficiency, all the while avoiding the “wasted inference” that comes with discarding negative examples. |
| Additional Notes      | This method may speed up learning while maintaining stable learning dynamics, without the use of KL regularization and online sampling. |



#### <div id="dapo">2025.0318, DAPO</div>

| Project or Paper      | [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO) |
| Backbone Model        | Qwen2.5-32B-Base                                             |
| RL Algorithm          | DAPO                                                         |
| Training Dataset      | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), 17K |
| Rollout Configuration | 512 prompts * 16 responses Dynamic Sampling                  |
| Reward Function       | Rule-based Rewards + Length-Aware Penalty Reward             |
| Policy Optimization   | DAPO loss, without KL loss                                   |
| Benchmark             | DeepSeek-R1-Zero-Qwen-32B level on AIME 2024                 |
| Core Insights         | DAPO introduces four key techniques: Clip-Higher to promote diversity and prevent entropy collapse; Dynamic Sampling to enhance training efficiency and stability; Token-Level Policy Gradient Loss to refine long-chain reasoning; and Overlong Reward Shaping to reduce reward noise and stabilize training. |
| Additional Notes      |                                                              |



#### <div id="oat-zero">2025.0321, Oat-Zero</div>

| Project or Paper      | [Understanding R1-Zero-Like Training: A Critical Perspective](https://github.com/sail-sg/understand-r1-zero/blob/main/understand-r1-zero.pdf) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [sail-sg/understand-r1-zero](https://github.com/sail-sg/understand-r1-zero) |
| Backbone Model        | Qwen2.5-1.5B                                                 |
| RL Algorithm          | Dr. GRPO (fixes GRPO’s bias in optimization)                 |
| Training Dataset      | Questions sampled from the MATH;                             |
| Rollout Configuration | N/A                                                          |
| Reward Function       | Rule-based Rewards                                           |
| Policy Optimization   | Dr. GRPO Loss (remove two normalization terms in GRPO)       |
| Benchmark             | AIME2024, AMC, MATH500, Minerva Math and OlympiadBench       |
| Core Insights         | 1. The DeepSeek-V3-Base model exhibits significant reasoning capabilities, termed an “aha moment,” even prior to reinforcement learning fine-tuning. 2. GRPO introduces an optimization bias that artificially increases response length during training, particularly affecting incorrect outputs. |
| Additional Notes      |                                                              |



### Multimodal and Applications

#### <div id="open-r1-mm">2025.0128, open-r1-multimodal</div>

| Project or Paper      | [EvolvingLMMs-Lab/open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [EvolvingLMMs-Lab/open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) |
| Backbone Model        | Qwen2-VL 2B/7B                                               |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified), 7.69K; |
| Rollout Configuration | 1 (prompts + images) * 8 responses; Temperature=0.9;         |
| Reward Function       | Rule-based Rewards (Choice, Format)                          |
| Policy Optimization   | PPO Loss                                                     |
| Benchmark             | MMMU:  <br />2B: +4.02% vs w./ reasoning, -4.48% vs base; <br />7B: 7.5% vs w./ reasoning, -1.2% vs base;  <br />Mathvista-mini:  <br />2B: +0.8% vs w./ reasoning, -2.2% vs base; <br />7B: -0.3% vs w./ reasoning, +3.5% vs base; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |



#### <div id="r1v">2025.0202, R1-V</div>

| Project or Paper      | [RLVR in Vision Language Models: Findings, Questions and Directions](https://deepagent.notion.site/rlvr-in-vlms) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [Deep-Agent/R1-V](https://github.com/Deep-Agent/R1-V[)       |
| Backbone Model        | Visual Counting/Complex Visual Reasoning: Qwen2-VL-2B-Instruct; <br />Geometry Reasoning: Qwen2.5-VL-7B-Instruct; |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | Visual Counting: [Clevr CoGenT-A](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train), 70K<br /> Geometry Reasoning: [GeoQA-Train](https://huggingface.co/datasets/leonardPKU/GEOQA_R1V_Train_8K), 8K<br /> Complex Visual Reasoning: [Clevr_CoGenT_TrainA_R1](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_R1), 37.8K |
| Rollout Configuration | 1 (prompts + images) * 8 responses; Temperature=1.0;         |
| Reward Function       | Rule-based Rewards (Accuracy: Number/bool, Format)           |
| Policy Optimization   | PPO Loss                                                     |
| Benchmark             | Visual Counting (Acc.): Clevr CoGenT-B: 46%, comparable vs base and CoT+SFT models; SuperClevr:  40%, 11% higher than base and CoT+SFT models;  <br />Geometry Reasoning (Acc., no CoT data): GeoQA-Test: 24%, 1% higher than base and SFT models;  <br />Complex Visual Reasoning: SuperClevr: 53.48%, 49.28% higher than base and CoT+SFT models; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |



#### <div id="vlm-r1">2025.0215, VLM-R1</div>

| Project or Paper      | [VLM-R1: A stable and generalizable R1-style Large Vision-Language Model](https://om-ai-lab.github.io/index.html) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1)      |
| Backbone Model        | Qwen2.5-VL-3B                                                |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [COCO](https://cocodataset.org/#download), 83K, no improvement for both STF and RL model<br /> [Description Detection Dataset](https://github.com/shikras/d-cube), 24K; |
| Rollout Configuration | 8 (prompts + images) * 8 responses; Temperature=0.9;         |
| Reward Function       | Rule-based Rewards (IoU, Format)                             |
| Policy Optimization   | PPO Loss + KL Loss (default 0.04)                            |
| Benchmark             | OVDEval: 6.55%, 4.51% higher than base and SFT models in NMS-AP; <br />COCO(filter out images with more than 10 bbox): 6.1%, 2.6% higher than base and SFT models in MAP; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |



#### <div id="r1-vlm">2025.0306, r1-vlm</div>

| Project or Paper      | [GRPO for vision - Teaching an LLM to reason about images](https://www.groundlight.ai/blog/visual-reasoning-models) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [groundlight/r1_vlm](https://github.com/groundlight/r1_vlm)  |
| Backbone Model        | Qwen2.5-VL-3B-Instruct                                       |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | Message Decoding: [message-decoding-words-and-sequences-r1](https://huggingface.co/datasets/sunildkumar/message-decoding-words-and-sequences-r1), 27K<br /> Message Decoding-Single Word [message-decoding-words-r1](https://huggingface.co/datasets/sunildkumar/message-decoding-words-r1), 10K<br />Digit Recognition: [digit-recognition-r1 ](https://huggingface.co/datasets/sunildkumar/digit-recognition-r1), 2K |
| Rollout Configuration | 1 (prompts + images) * 9 responses; Temperature=1.0;         |
| Reward Function       | Rule-based Rewards (Decoding, Correctness, Format)           |
| Policy Optimization   | PPO Loss + KL Loss (default 0.01)                            |
| Benchmark             | -                                                            |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |



#### <div id="visual-r1-zero">2025.0310, VisualThinker-R1-Zero</div>

| Project or Paper      | [R1-Zero’s “Aha Moment” in Visual Reasoning on a 2B Non-SFT Model](https://arxiv.org/pdf/2503.05132) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [turningpoint-ai/VisualThinker-R1-Zero](https://github.com/turningpoint-ai/VisualThinker-R1-Zero) |
| Backbone Model        | Qwen2-VL-2B                                                  |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [SAT](https://huggingface.co/datasets/array/SAT), 218K;      |
| Rollout Configuration | 4 (prompts + images) * 8 responses; Temperature=1.0;         |
| Reward Function       | Rule-based Rewards (Accuracy-String Match, Format);         |
| Policy Optimization   | PPO Loss + KL Loss (default 0.04)                            |
| Benchmark             | CV-Bench (Choice): +25% vs base, +10.83% vs SFT;<br /> BLINK: +46.44 vs base, +0.75% vs SFT; <br />VSR: +62.32% vs base, +26.53% vs SFT; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


#### <div id="mm-eureka">2025.0310, MM-Eureka</div>

| Project or Paper      | [MM-EUREKA: EXPLORING VISUAL AHA MOMENT WITH RULE-BASED LARGE-SCALE REINFORCEMENT LEARNING](https://arxiv.org/pdf/2503.07365) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [ModalMinds/MM-EURRKA](https://github.com/ModalMinds/MM-EUREKA) |
| Backbone Model        | InternVL2.5-Pretrained-38B                                                 |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset), 55K;      |
| Rollout Configuration | 128 (prompts + images) * 8 responses; Temperature=1.0;         |
| Reward Function       | Rule-based Rewards (Accuracy-String Match, Format);         |
| Policy Optimization   | PPO Loss                            |
| Benchmark             | +9.2%, +4.7% compared with base model on OlympicBench and L12 respectively; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


## Contributing

If you have any updates or improvements for this document, please feel free to submit a **Pull Request**. Thank you!

#### <div id="template">202x.0x0x, Template</div>

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

 If you find our repository useful in your research, please star us ⭐ and consider citing:

```tex
@misc{zhang2025TripleR,
  title={Awesome RL Recipes for Reasoning},
  author={Kaiyan Zhang, Yuchen Fan, Yuxin Zuo, Guoli Jia, Xingtai Lv, Xuekai Zhu, Ermo Hua, Ning Ding, Biqing Qi, Bowen Zhou},
  year={2025},
  howpublished={\url{https://github.com/}},
  note={Github Repository},
}
```

