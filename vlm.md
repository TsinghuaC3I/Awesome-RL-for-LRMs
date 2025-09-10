### Multimodal Models

- [Multimodal Models](#multimodal-models)
  - [2025.0128, open-r1-multimodal](#20250128-open-r1-multimodal)
  - [2025.0202, R1-V](#20250202-r1-v)
  - [2025.0215, VLM-R1](#20250215-vlm-r1)
  - [2025.0303, Visual-RFT](#20250303-visual-rft)
  - [2025.0306, r1-vlm](#20250306-r1-vlm)
  - [2025.0310, VisualThinker-R1-Zero](#20250310-visualthinker-r1-zero)
  - [2025.0310, MM-Eureka](#20250310-mm-eureka)
  - [2025.0310, Curr\_ReFT](#20250310-curr_reft)
  - [2025.0311, MMR1](#20250311-mmr1)
  - [2025.0315, MetaSpatial](#20250315-metaspatial)
  - [2025.0327, Reason-RFT](#20250327-reason-rft)
  - [2025.0409, Kimi-VL-Thinking](#20250409-kimi-vl-thinking)
  - [2025.0409, VideoChat-R1](#20250409-videochat-r1)
  - [2025.0410, VL-Rethinker](#20250410-vl-rethinker)



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
| GitHub                | [Deep-Agent/R1-V](https://github.com/Deep-Agent/R1-V)       |
| Backbone Model        | Visual Counting/Complex Visual Reasoning: Qwen2-VL-2B-Instruct; <br />Geometry Reasoning: Qwen2.5-VL-7B-Instruct; |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | Visual Counting: [Clevr CoGenT-A](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train), 70K;<br /> Geometry Reasoning: [GeoQA-Train](https://huggingface.co/datasets/leonardPKU/GEOQA_R1V_Train_8K), 8K;<br /> Complex Visual Reasoning: [Clevr_CoGenT_TrainA_R1](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_R1), 37.8K; |
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
| Training Dataset      | [COCO](https://cocodataset.org/#download), 83K, no improvement for both STF and RL model;<br /> [Description Detection Dataset](https://github.com/shikras/d-cube), 24K; |
| Rollout Configuration | 8 (prompts + images) * 8 responses; Temperature=0.9;         |
| Reward Function       | Rule-based Rewards (IoU, Format)                             |
| Policy Optimization   | PPO Loss + KL Loss (default 0.04)                            |
| Benchmark             | OVDEval: 6.55%, 4.51% higher than base and SFT models in NMS-AP; <br />COCO(filter out images with more than 10 bbox): 6.1%, 2.6% higher than base and SFT models in MAP; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


#### <div id="vlm-r1">2025.0303, Visual-RFT</div>

| Project or Paper      | [Visual-RFT: Visual Reinforcement Fine-Tuning ](https://arxiv.org/pdf/2503.01785) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [Liuziyu77/Visual-RFT](https://github.com/Liuziyu77/Visual-RFT)      |
| Backbone Model        | Qwen2-VL-2/7B                                                |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [ViRFT Datasets](https://huggingface.co/collections/laolao77/virft-datasets-67bc271b6f2833eccc0651df), 6K/6k/32/408/400/784/148; |
| Rollout Configuration | 2 * 8 responses; Temperature=1.0;                            |
| Reward Function       | Rule-based Rewards (Accuracy, IoU, Format)；                  |
| Policy Optimization   | PPO Loss                         |
| Benchmark             | Qwen2-VL-2B:<br />Fine-grained classification, Avg. Acc.（Flower102, Pets37, FGVC-Aircraft, Car196）:<br />1-shot +24.3%, +28.6%, 2-shot +27.5%, +24.7, 4-shot +25.9% , +26.3%, 8-shot +29.1%, +24.8%, 16-shot +29.3%, +21.3%, compared to base and SFT;<br />Object Detection (COCO), mAP:<br />1-shot +14.0%, +14.1%, 2-shot +21.9%, +20.5%, 4-shot +21.0%, +15.4%, 8-shot +27.8%, 17.2%, 16-shot +27.2%, +15.5%,  compared to base and SFT;<br />Rare Object Detection (LVIS), mAP:<br />10-shot, +15.4%, +9.4%,  compared to base and SFT;<br />Open Vocabulary Object Detection, mAP:<br />COCO: +21.5%, +17.7%, compared to base and SFT;<br />LVIS: +18.0%, +13.1%, compared to base and SFT;<br />Reasoning Grounding (LISA), mIoU:<br />+10.7%, +9.3%,  compared to base and SFT;<br /><br />Qwen2-VL-7B:<br />Object Detection (COCO), mAP:<br />4-shot +11.3%, +10.2%,  compared to base and SFT;<br />Rare Object Detection (LVIS), mAP:<br />10-shot, +18.4%, +6.2%,  compared to base and SFT;<br />Open Vocabulary Object Detection, mAP:<br />COCO: +9.5%, +10.1%, compared to base and SFT;<br />LVIS: +14.7%, +6.4%, compared to base and SFT;<br />Reasoning Grounding (LISA), mIoU:<br />+3.5%, +4.8%,  compared to base and SFT; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


#### <div id="r1-vlm">2025.0306, r1-vlm</div>

| Project or Paper      | [GRPO for vision - Teaching an LLM to reason about images](https://www.groundlight.ai/blog/visual-reasoning-models) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [groundlight/r1_vlm](https://github.com/groundlight/r1_vlm)  |
| Backbone Model        | Qwen2.5-VL-3B-Instruct                                       |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | Message Decoding: [message-decoding-words-and-sequences-r1](https://huggingface.co/datasets/sunildkumar/message-decoding-words-and-sequences-r1), 27K;<br /> Message Decoding-Single Word [message-decoding-words-r1](https://huggingface.co/datasets/sunildkumar/message-decoding-words-r1), 10K;<br />Digit Recognition: [digit-recognition-r1 ](https://huggingface.co/datasets/sunildkumar/digit-recognition-r1), 2K; |
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
| Additional Notes      |      |

#### <div id="curr-reft">2025.0310, Curr_ReFT</div>

| Project or Paper      | [Boosting the Generalization and Reasoning of Vision Language Models with Curriculum Reinforcement Learning](https://arxiv.org/pdf/2503.07065) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [ding523/Curr_REFT](https://github.com/ding523/Curr_REFT) |
| Backbone Model        | Qwen2.5-VL-3/7B                                                 |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [RefCOCO](https://github.com/lichengunc/refer), 3K for training;<br />[Math360K](https://huggingface.co/datasets/Zhiqiang007/MathV360K), [Geo170K](https://huggingface.co/datasets/Luckyjhg/Geo170K), 3K for training;     |
| Rollout Configuration | 1 * 4 responses; Temperature=1.0;         |
| Reward Function       | Difficulty-aware Rule-based Rewards (Accuracy, IoU, Format)；         |
| Policy Optimization   | PPO Loss                            |
| Benchmark             | ID (In-Distribution), compared to base and SFT:<br/>Math (Math360K+Geo170K, 1K for testing): -3B +11.0%, +8.8%, -7B +11.2%, +4.4%;<br/>Detection (RefCOCO, 1K for testing): -3B +58.0%, +14.6%, -7B +51.3%, +2.5%;<br/>Classification: (RefCOCO, 1K for testing) -3B +31.9%, +21.3%, -7B +44.6%, +4.2%;<br/><br/>OOD(Out-of-Distribution), compared to base and SFT: <br/>Math (CLEVER-70K, 0.5K for testing): -3B +55.9%, +42.9%, -7B +46.6%, +21.6%;<br/>Detection (Refgta, 1K for testing): -3B +43.3%, +13.3% -7B +41.7%, +28.3%;<br/>Classification: (Pascal-VOC, 1K for testing) -3B +15.4%, +18.0%, -7B +12.5%, +6.5%; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


#### <div id="curr-reft">2025.0311, MMR1</div>

| Project or Paper      | [MMR1: Advancing the Frontiers of Multimodal Reasoning](https://github.com/LengSicong/MMR1) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [LengSicong/MMR1](https://github.com/LengSicong/MMR1) |
| Backbone Model        | Qwen2.5-VL-7B                                                 |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | 6K high-quality samples from public training datasets    |
| Rollout Configuration |          |
| Reward Function       |        |
| Policy Optimization   |                        |
| Benchmark             | Compared to base model: +3.5%, +4.6%, +4.0%, +2.6%, +2.9% on MathVista, MathVision, LogicVista, MathVerse, MathVerse_V;<br />Compared to sft-cot: +16.3%, +6.8%, +17.0%, +21.4%, +24.1% on MathVista, MathVision, LogicVista, MathVerse, MathVerse_V; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


#### <div id="metaspatial">2025.0315, MetaSpatial</div>

| Project or Paper      | [MetaSpatial: Reinforcing 3D Spatial Reasoning in VLMs for the Metaverse](https://arxiv.org/abs/2503.18470) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [PzySeere/MetaSpatial](https://github.com/PzySeere/MetaSpatial) |
| Backbone Model        | Qwen2.5-VL-7B                                                 |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | [3D-Reasoning-Dataset](https://huggingface.co/datasets/zhenyupan/3d_layout_reasoning), 50;      |
| Rollout Configuration | 16 (prompts + images) * 4 responses; Temperature=1.0;         |
| Reward Function       | Rule-based Rewards (Format Reward, Physics Reward, Rendering-based Reward);         |
| Policy Optimization   | PPO Loss                            |
| Benchmark             | +74% compared with base model on test-set of 3d-reasoning dataset; |
| Core Insights         |   Injecting physics reward and gpt-4o-based rendering evaluation reward.                                                    |
| Additional Notes      |   


#### <div id="curr-reft">2025.0327, Reason-RFT</div>

| Project or Paper      | [Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning](https://arxiv.org/pdf/2503.20752) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [tanhuajie/Reason-RFT](https://github.com/tanhuajie/Reason-RFT) |
| Backbone Model        | Qwen2-VL-2/7B                                                 |
| RL Algorithm          | GRPO                                                         |
| Training Dataset      | Visual Counting: [CLEVR-Math](https://huggingface.co/datasets/dali-does/clevr-math), 35K for training;<br />Structure Perception: [Geo170K](https://huggingface.co/datasets/Luckyjhg/Geo170K), 4.5K for training;<br />Spatial Transformation: [TRANCE](https://github.com/hughplay/TVR), 60K for training;     |
| Rollout Configuration | 2 * 8 responses; Temperature=1.0;         |
| Reward Function       | Rule-based Rewards (Accuracy, Format)；         |
| Policy Optimization   | PPO Loss                            |
| Benchmark             | ID (In-Distribution), compared to base and CoT-SFT(stage-1):<br />Visual Counting: -2B +14.40%, +11.30%, -7B -3.00%, +12.30%;<br />Structure Perception: -2B +23.17%, +5.98%, -7B +15.97%, +7.93%;<br />Spatial Transformation: -2B +70.83%, +9.76%, -7B +66.44%, -1.34%;<br/><br/>OOD (Out-of-Distribution), compared to base and CoT-SFT(stage-1):<br/>Visual Counting: -2B +19.20%, +4.70%, -7B +8.90%, +8.60%;<br/>Structure Perception: -2B +12.50%, +7.88%, -7B +5.37%, +16.25%;<br/>Spatial Transformation: -2B +59.45%, +20.86%, -7B +46.64%, +11.46%;|
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


#### <div id="curr-reft">2025.0409, Kimi-VL-Thinking</div>

| Project or Paper      | [KIMI-VL TECHNICAL REPORT](https://github.com/MoonshotAI/Kimi-VL/blob/main/Kimi-VL.pdf) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [MoonshotAI/Kimi-VL](https://github.com/MoonshotAI/Kimi-VL) |
| Backbone Model        | Kimi-VL-A3B                                                |
| RL Algorithm          | Online Policy Mirror Decent/Length Penalty Reward/Curriculum Sampling/Prioritized Sampling/Chain-of-Thought RM/Long2short RL                  |
| Training Dataset      | Long-CoT data about mathematical problem-solving and domain-specific VQA     |
| Rollout Configuration | None        |
| Reward Function       | Correctness, Length；         |
| Policy Optimization   | Online Policy Mirror Decent                     |
| Benchmark             | Comparable with Qwen2.5-VL-32/72B on MathVision (Pass@1)     |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


#### <div id="curr-reft">2025.0409, VideoChat-R1</div>

| Project or Paper      | [VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning](https://arxiv.org/pdf/2504.06958) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [OpenGVLab/VideoChat-R1](https://github.com/OpenGVLab/VideoChat-R1) |
| Backbone Model        | Qwen2.5-VL-7B                                               |
| RL Algorithm          | GRPO    |
| Training Dataset      | Joint training on three tasks: Temporal Grounding: Charade-STA, 5K; Object Tracking: GoT, 10K; Grounding QA: NExTGQA, 3K; total of 18K;   |  
| Rollout Configuration | 1 * 8 responses; Temperature=1.0;       |
| Reward Function       | Rule-based Rewards (Accuracy, Format, IoU)；         |
| Policy Optimization   | PPO Loss                    |
| Benchmark             | mIoU(Overlap on OT) compared with base and SFT:<br />Temporal Grounding: +31.8%, +14.5%;<br />Object Tracking: +31.2%, +2.0%;<br />Grounding QA: +17.0%, +4.2%;     |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |


#### <div id="curr-reft">2025.0410, VL-Rethinker</div>

| Project or Paper      | [VL-Rethinker: Incentivizing Self-Reflection of Vision-Language Models with Reinforcement Learning](https://arxiv.org/pdf/2504.08837) |
| --------------------- | ------------------------------------------------------------ |
| GitHub                | [TIGER-AI-Lab/VL-Rethinker](https://github.com/TIGER-AI-Lab/VL-Rethinker) |
| Backbone Model        | Qwen2.5-VL-7/72B                                                 |
| RL Algorithm          | GRPO + Selective Sample Replay                                                         |
| Training Dataset      | A 16K query set from Virgo, R1-OneVision, and MM-Eureka;    |
| Rollout Configuration |          |
| Reward Function       |        |
| Policy Optimization   |  PPO loss + KL loss;                      |
| Benchmark             | 7B compared to base model: +4.7%, +6.3%, +2.4% on MathVista, MathVerse, MathVision, +4.4%, +0.7%, +3.1% on MMMU-Pro, MMMU, EMMA,  +2.2% on MEGA;<br />72B compared to base model: +5.5%, +4.5%, +5.5% on MathVista, MathVerse, MathVision, +1.7%, -1.4%, +5.8% on MMMU-Pro, MMMU, EMMA, +2.3% on MEGA; |
| Core Insights         |                                                              |
| Additional Notes      |                                                              |

