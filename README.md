# 📚 DistilRL: A Condensed Introduction

**A visual, bilingual 12-chapter path from Sutton & Barto to PPO and GRPO.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Website](https://img.shields.io/badge/Read-Online-2563eb?logo=readthedocs&logoColor=white)](https://datawhalechina.github.io/distil-rl-introduction/)
[![English](https://img.shields.io/badge/Language-English-334155)](README.md)
[![中文](https://img.shields.io/badge/语言-中文-c2410c)](README_zh.md)
[![GitHub Stars](https://img.shields.io/github/stars/datawhalechina/distil-rl-introduction?style=social)](https://github.com/datawhalechina/distil-rl-introduction)

<span>[ English | <a href="README_zh.md">中文</a> ]</span>

---

<img src="docs/_static/img/logo.png" align="right" width="34%" alt="DistilRL reinforcement learning guide logo"/>

## Start Reading

DistilRL reorganizes selected ideas from Sutton and Barto's *Reinforcement Learning: An Introduction* and the [University of Alberta Reinforcement Learning Specialization](https://www.coursera.org/specializations/reinforcement-learning) into a shorter conceptual route. The 12 chapters connect multi-armed bandits, MDPs, dynamic programming, Monte Carlo methods, temporal-difference learning, function approximation, and policy gradients before reaching PPO and GRPO.

- **New to reinforcement learning?** [Start with Chapter 0 and follow the full path](https://datawhalechina.github.io/distil-rl-introduction/Contents/0_prelude.html).
- **Learning RL for PPO, GRPO, or LLM post-training?** [Start with the MDP foundation in Chapter 3](https://datawhalechina.github.io/distil-rl-introduction/Contents/3_markov_decision_process.html), then continue through Chapters 6, 8, 10, and 11.
- **Prefer Chinese?** [Read the complete Chinese edition](https://datawhalechina.github.io/distil-rl-introduction/index_zh.html).

**[Read the tutorial online](https://datawhalechina.github.io/distil-rl-introduction/)** · **[Star DistilRL on GitHub](https://github.com/datawhalechina/distil-rl-introduction)**

<br clear="right"/>

## Who It Is For

- Learners who want the main conceptual thread before going deeper into the original textbook.
- ML and LLM engineers who want the reinforcement-learning foundations behind policy gradients, PPO, and GRPO.
- English and Chinese readers who want to move between two complete editions of the same 12-chapter path.

DistilRL is a reading guide and conceptual companion. It is not a code-first course, and it does not aim to replace the complete Sutton and Barto textbook.

## The 12-Chapter Learning Path

> **Simple decisions → sequential decisions → planning with a model → learning from samples → temporal-difference learning → function approximation → direct policy optimization → PPO and GRPO.**

Chapter 0 explains why the guide exists and how to use it. Optional material and visual summaries help you choose the depth that fits your goal.

---

## 📋 Catalog

### 🌟 Introduction

- [Chapter 0: Prelude](docs/Contents/0_prelude.md)
- [Chapter 1: Introduction to RL](docs/Contents/1_intro.md)

### 🧮 Tabular Solution Methods

#### Fundamentals of Reinforcement Learning

- [Chapter 2: Multi-armed Bandits](docs/Contents/2_multi_armed_bandits.md)
- [Chapter 3: Markov Decision Process](docs/Contents/3_markov_decision_process.md)
- [Chapter 4: Dynamic Programming](docs/Contents/4_dynamic_programming.md)

#### Sample-based Learning Methods

- [Chapter 5: Monte Carlo Methods](docs/Contents/5_monte_carlo_methods.md)
- [Chapter 6: Temporal Difference Learning](docs/Contents/6_temporal_difference_learning.md)
- [Chapter 7: Planning, Learning, Acting](docs/Contents/7_planning_learning_acting.md)

### 🤖 Approximate Solution Methods

#### Value Function Approximation

- [Chapter 8: On-policy Prediction with Approximation](docs/Contents/8_on_policy_prediction_with_approximation.md)
- [Chapter 9: On-policy Control with Approximation](docs/Contents/9_on_policy_control_with_approximation.md)

#### Policy Approximation

- [Chapter 10: Policy Gradient Methods](docs/Contents/10_policy_gradient_methods.md)
- [Chapter 11: Modern Policy Gradient Methods](docs/Contents/11_modern_policy_gradient_methods.md)

---

## 🤝 Contributing

Both the English and Chinese editions are complete. Contributions that make either edition clearer or easier to navigate are welcome:

- **Report an issue:** Flag a typo, broken link, or unclear explanation.
- **Suggest an improvement:** Open a pull request with a focused correction.
- **Help more learners find it:** Share the chapter that was most useful to you.

If DistilRL saved you time, consider [starring the repository](https://github.com/datawhalechina/distil-rl-introduction) so you can find it again and help the next learner discover it.

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).
