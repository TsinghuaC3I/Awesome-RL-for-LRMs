# Awesome-RRT [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)


Collection of Reasoning RL Tricks (RRT).

> While RL has significantly advanced LLM reasoning, its training remains highly unstable and is influenced by various factors, such as hyperparameters and data.
>
> The **Awesome-RRT Collection** aims to provide resources and empirical insights on reinforcement learning (RL) training for large language model (LLM) reasoning.

In this repository, we systematically analyze and compare recent efforts to reproduce DeepSeek-R1, focusing on training details to provide insights that facilitate the efficient implementation of RL training. Additionally, we track the latest advancements in this field and curate relevant resources such as datasets and frameworks.


## Table of Contents

- [Awesome-RRT ](#awesome-rrt-)
  - [Table of Contents](#table-of-contents)
  - [Updates](#updates)
  - [Projects](#projects)
    - [Overview](#overview)
    - [LLM](#llm)
      - [DAPO](#dapo)
      - [TinyZero](#tinyzero)
    - [VLM](#vlm)
      - [R1-VL](#r1-vl)
    - [Application](#application)
  - [Acknowledgment](#acknowledgment)
  - [Contributing](#contributing)
  - [License](#license)

## Updates

- :fire: **[2025.03.20]** Add [DAPO](https://dapo-sia.github.io) - an Open-Source LLM Reinforcement Learning System at Scale.

## Projects

### Overview


| Date                                               |                         | Backbone                                                  | Tasks                     | Training                                  | Findings                          | Details                      | Links                                                        |
| ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------ | --------------------------------------------------- | --------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2025/03/18                                        | DAPO          | Qwen2.5-32B                                       | Mathematical  Reasoning | -                                                   |                                                    | [[DAPO]](#dapo)                                    | [[Code]](https://github.com/BytedTsinghua-SIA/DAPO)<br/>[[Paper]](https://arxiv.org/pdf/2503.14476) |
| 2025/01/24                             | TinyZero                | Qwen2.5-3B-Instruct                | Countdown               | 4 A800s                                             |                                              | [[TinyZero]](#tinyzero) | [[Code]](https://github.com/Jiayi-Pan/TinyZero)<br/>[[Experiment Log]](https://wandb.ai/jiayipan/TinyZero) |

---

### LLM

#### <mark>DAPO</mark>

- Paper: https://arxiv.org/pdf/2503.14476v1
- Code: https://github.com/BytedTsinghua-SIA/DAPO

| Name              | Value                                                        |
| :---------------- | :----------------------------------------------------------- |
| Backbone Model    | [[Qwen2.5-32B]](https://huggingface.co/Qwen/Qwen2.5-32B)     |
| Training Data     | [[Data Link]](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)<br/>size: 17k<br/>source: AoPS website |
| Hyperparams       | train_batch_size:<br/>rollout_batch_size:<br/>n_samples_per_prompt:<br/>episode:<br/>epoch:<br/>learning_rate:<br/>rl_advantage:<br/>gpus (hours): |
| RL-Curve          | <img src="figs/dapo_curve.png" alt="RL-Curve" style="zoom: 15%;" /> |
| Tricks & Findings |                                                              |

#### <mark>TinyZero</mark>

- Code: https://github.com/Jiayi-Pan/TinyZero
- Experiment log: https://wandb.ai/jiayipan/TinyZero

---

### VLM

#### <mark>R1-VL</mark>



### Application

- Medical & Search



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

## License

This project is licensed under the [MIT License](https://github.com/TsinghuaC3I/Awesome-RRT/blob/main/LICENSE).
