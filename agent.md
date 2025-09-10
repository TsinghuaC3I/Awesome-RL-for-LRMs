### Agentic Applications

- [Agentic Applications](#agentic-applications)
  - [2025.0307, R1-Searcher](#20250307-r1-searcher)
  - [2025.0309, AutoCoA](#20250309-autocoa)
  - [2025.0312, Search-R1](#20250312-search-r1)
  - [2025.0404, DeepResearcher](#20250404-deepresearcher)
  - [2025.0407, SWiRL](#20250407-swirl)
  - [2025.0415, ReTool](#20250415-retool)
  - [2025.0430, WebThinker](#20250430-webthinker)


#### <div id="r1-searcher">2025.0307, R1-Searcher</div>

| Project or Paper      | [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/pdf/2503.05592) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [RUCAIBox/R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) |
| Backbone Model        |  Qwen-2.5-7B-Base</br>Llama-3.1-8B-Instruct                                             |
| RL Algorithm          | REINFORCE++                                                         |
| Training Dataset      | [RAG-RL-Hotpotqa-with-2wiki](https://huggingface.co/datasets/XXsongLALA/RAG-RL-Hotpotqa-with-2wiki) & [HotpotQA](https://huggingface.co/datasets/BeIR/hotpotqa), 350 for Stage-1 Training, 8k for Stage-2 Training      |
| Rollout Configuration | 64 prompts * 16 responses; Temperature = 1.0;         |
| Reward Function       | Rule-based Rewards (Format Reward for Stage-1 Training and Format Reward and Anser Reward for Stage-2 Training);         |
| Policy Optimization   | REINFORCE++ without KL Penalty                            |
| Benchmark             | HotpotQA, 2WikiMultiHopQA, Musique,  Bamboogle |
| Core Insights         | The proposed R1-Searcher framework integrates RAG with RL, enabling the model to invoke an external search engine during the reasoning process. The framework demonstrates the ability to generalize from in-domain training datasets to out-of-domain test datasets, and can seamlessly switch to online search to obtain up-to-date information. |

#### <div id="AutoCoA">2025.0309, AutoCoA</div>

| Project or Paper      | [Agent Models: Internalizing Chain-of-Action Generation into Reasoning Models](https://arxiv.org/abs/2503.06580) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [ADaM-BJTU/AutoCoA](https://github.com/ADaM-BJTU/AutoCoA)       |
| Backbone Model        | DeepSeek-R1-Distill-Qwen-7B                                   |
| RL Algorithm          | GRPO                                                        |
| Training Dataset      | Synthesized data based on HotpotQA and DeepSeek-R1-Distill-Qwen-32B [data](https://github.com/ADaM-BJTU/AutoCoA/tree/main/data)  |
| Rollout Configuration | 1 prompt * 5 responses; Temperature=0.6;                                                           |
| Reward Function       | Rule-based Rewards (Exact Match and Format Reward)          |
| Policy Optimization   | First through multi-stage SFT (using improved contrastive loss to train CoT+A, CoT+CoA [with/without observation mask]), then combined with an RL stage (GRPO optimization) for end-to-end fine-tuning |
| Benchmark             | Open-domain QA evaluation: single-hop (NQ, TriviaQA) and multi-hop (HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle) |
| Core Insights         | The AutoCoA framework internalizes the decision-making for external tool invocation (Chain-of-Action) into the reasoning model, enabling seamless alternation between thought and action, which significantly improves performance on multi-turn long-horizon tasks. It proposes a two-phase training method: injecting the "when-to-act" and "how-to-act" capabilities via supervised fine-tuning and reinforcement learning, effectively reducing the cost of real interactions and enhancing adaptability in real-world environments. |

#### <div id="search-r1">2025.0312, Search-R1</div>

| Project or Paper      | [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/pdf/2503.09516) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) |
| Backbone Model        | Qwen-2.5-3B (Base/Instruct) and Qwen-2.5-7B (Base/Instruct)                         |
| RL Algorithm          | GRPO and PPO                                                  |
| Training Dataset      | [2018 Wikipedia](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)      |
| Rollout Configuration | 64 prompts * 5 responses; Temperature=1.0;         |
| Reward Function       | Rule-based Rewards (Exact Match)         |
| Policy Optimization   | PPO Loss with 0.001 KL coefficient, masking retrived tokens                         |
| Benchmark             | General QA; Multi-Hop QA |
| Core Insights         | SEARCH-R1 addresses the challenges of RL framework and stability, multi-turn interleaved reasoning and search, and reward design, and introduces retrieved token masking to ensure stable optimization.                                |
| Additional Notes      |   

#### <div id="DeepResearcher">2025.0404, DeepResearcher</div>

| Project or Paper      | [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/abs/2504.03160) |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| GitHub                | [GAIR-NLP/DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)                                                     |
| Backbone Model        | Qwen2.5-7B-Instruct                                                                                                     |
| RL Algorithm          | GRPO                                               |
| Training Dataset      | [Open-domain QA datasets](https://github.com/GAIR-NLP/DeepResearcher/tree/main/data) (e.g. NQ, TQ, HotpotQA, 2Wiki) – – a total of about 80,000 examples                                           |
| Rollout Configuration | 256 prompts * 16 responses, with a maximum of 10 tool calls per rollout                                                        |
| Reward Function       | Rule-based Rewards(Word-level F1 score and Format Reward)                                                                                       |
| Policy Optimization   | GRPO with masking obvervations                         |
| Benchmark             | In-domain: NaturalQuestions, TriviaQA, HotpotQA, 2WikiMultiHopQA；Out-of-domain: Musique, Bamboogle, PopQA                                             |
| Core Insights         | By training end-to-end with reinforcement learning in a real-world web environment, DeepResearcher can autonomously plan, cross-validate from multiple sources, reflect, and maintain honesty; it significantly improves open-domain QA and deep research capabilities; while also overcoming practical challenges such as search API limitations, web noise, and anti-crawling mechanisms. |

#### <div id="SWiRL">2025.0407, SWiRL</div>

| Project or Paper | [Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use](https://arxiv.org/abs/2504.04736) |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| **GitHub**       | N/A                                                                                            |
| **Backbone Model** | Gemma 2-27b                                                                                 |
| **RL Algorithm** | Step-Wise Reinforcement Learning (SWiRL)                                                        |
| **Training Dataset** | - HotPotQA: Generated 50,000 synthetic trajectories using 10,000 multi-step questions<br>- GSM8K: Generated 37,500 synthetic trajectories using 7,500 questions |
| **Rollout Configuration** | N/A                                                                                    |
| **Reward Function** | Model-based Rewards (scoring each step’s appropriateness using Gemini 1.5 Pro)               |
| **Policy Optimization** | SWiRL: Stepwise optimization, similar to RLHF in Gemma 2                              |
| **Benchmark**    | HotPotQA; GSM8K; CofCA; MuSiQue; BeerQA                                                           |
| **Core Insights** | 1. Multi-step reasoning can be optimized via step-level rewards, benefiting even if the final answer is incorrect<br>2. Process filtering (selecting based on intermediate step quality) is superior to outcome filtering<br>3. Strong cross-task generalization: for example, training only on HotPotQA can improve GSM8K performance by approximately 16.9% |

#### <div id="ReTool">2025.0415, ReTool</div>

| Project or Paper | [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/pdf/2504.11536) |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| **GitHub**       | https://github.com/ReTool-RL/ReTool                                                                                          |
| **Backbone Model** | Qwen2.5-32B-Instruct, DeepSeek-R1-Distill-Qwen-32B                                                                    |
| **RL Algorithm** | PPO                                                        |
| **Training Dataset** | [Tool-integrated synthesis data](https://huggingface.co/datasets/JoeYing/ReTool-SFT) |
| **Rollout Configuration** | N/A                                                                                    |
| **Reward Function** | Rule-based accuracy reward              |
| **Policy Optimization** | Standard PPO                              |
| **Benchmark**    | AIME 2024, AIME 2025                                                           |
| **Core Insights** | 1. The model's average response length decreased by 40% after RL training, suggesting improved efficiency in reasoning token utilization. <br/>2. The model's proficiency in code utilization improved during RL training, with the average code ratio increasing to nearly 98% and the average code lines increasing fivefold |

#### <div id="WebThinker">2025.0430, WebThinker</div>

| Project or Paper | [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/pdf/2504.21776) |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| **GitHub**       | https://github.com/RUC-NLPIR/WebThinker                                                                                          |
| **Backbone Model** | QwQ-32B,  DeepSeek-R1-Distilled-Qwen-7B,  DeepSeek-R1-Distilled-Qwen-14B,  DeepSeek-R1-Distilled-Qwen-32B                                                                                |
| **RL Algorithm** | Online DPO                                                        |
| **Training Dataset** | 3k example sampled from SuperGPQA, WebWalkerQA, OpenThoughts, Natural Reasoning, Numina Math |
| **Rollout Configuration** | N/A                                                                                    |
| **Reward Function** | Correctness, Tool Efficiency, and Thinking Conciseness               |
| **Policy Optimization** | Iterative Online DPO                              |
| **Benchmark**    | GPQA; GAIA; WebWalkerQA; HLE; Glaive                                                           |
| **Core Insights** | The framework's Deep Web Explorer and Autonomous Think-Search-and-Draft strategy enable LRMs to autonomously explore the web and produce comprehensive outputs. WebThinker outperforms existing methods and strong proprietary systems in complex reasoning benchmarks and scientific report generation tasks, enhancing LRM reliability and applicability in complex scenarios. |
