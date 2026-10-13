# 📚 蒸馏版强化学习：简明导论

**从 Sutton & Barto 到 PPO 与 GRPO 的双语、可视化 12 章强化学习精简路径。**

[![许可证](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![在线阅读](https://img.shields.io/badge/在线阅读-进入网站-2563eb?logo=readthedocs&logoColor=white)](https://datawhalechina.github.io/distil-rl-introduction/index_zh.html)
[![English](https://img.shields.io/badge/Language-English-334155)](README.md)
[![中文](https://img.shields.io/badge/语言-中文-c2410c)](README_zh.md)
[![GitHub 星标](https://img.shields.io/github/stars/datawhalechina/distil-rl-introduction?style=social)](https://github.com/datawhalechina/distil-rl-introduction)

<span>[ <a href="README.md">English</a> | 中文 ]</span>

---

<img src="docs/_static/img/logo.png" align="right" width="34%" alt="DistilRL 强化学习导论项目图标"/>

## 开始阅读

DistilRL 将 Sutton 与 Barto 的《强化学习导论》和 [University of Alberta / Coursera 强化学习专项课程](https://www.coursera.org/specializations/reinforcement-learning)中的部分核心内容重新筛选、组织为一条更精简的概念路径。12 章内容从多臂老虎机、MDP、动态规划、蒙特卡洛方法、时序差分和函数近似，逐步过渡到策略梯度、PPO 与 GRPO。

- **刚开始学强化学习？** [从第 0 章开始，按完整路径阅读](https://datawhalechina.github.io/distil-rl-introduction/Contents_zh/0_%E5%89%8D%E8%A8%80.html)。
- **为 PPO、GRPO 或大模型后训练补强化学习基础？** [从第 3 章的 MDP 基础开始](https://datawhalechina.github.io/distil-rl-introduction/Contents_zh/3_%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B.html)，再依次阅读第 6、8、10、11 章。
- **更习惯英文？** [阅读完整英文版](https://datawhalechina.github.io/distil-rl-introduction/)。

**[在线阅读中文教程](https://datawhalechina.github.io/distil-rl-introduction/index_zh.html)** · **[在 GitHub 上为 DistilRL 加星](https://github.com/datawhalechina/distil-rl-introduction)**

<br clear="right"/>

## 适合谁阅读

- 希望先理清强化学习主线，再深入阅读原教材的学习者。
- 希望补齐策略梯度、PPO 与 GRPO 概念基础的 ML 和 LLM 工程师。
- 希望在同一条 12 章学习路径中对照中英文内容的读者。

DistilRL 是一份阅读指南和概念学习伴侣，不是代码实战课，也不以完整替代 Sutton 与 Barto 原教材为目标。

## 12 章学习路径

> **简单决策 → 序列决策 → 已知模型求解 → 从样本中学习 → 时序差分 → 函数近似 → 直接策略优化 → PPO 与 GRPO。**

第 0 章介绍项目的由来和阅读方法。你可以借助可选内容标记和可视化总结，按自己的目标选择阅读深度。

---

## 📋 目录

### 🌟 介绍

- [第 0 章：序言](docs/Contents_zh/0_前言.md)
- [第 1 章：强化学习简介](docs/Contents_zh/1_强化学习简介.md)

### 🧮 表格解决方法

#### 强化学习基础

- [第 2 章：多臂老虎机](docs/Contents_zh/2_多臂老虎机问题.md)
- [第 3 章：马尔可夫决策过程](docs/Contents_zh/3_马尔可夫决策过程.md)
- [第 4 章：动态规划](docs/Contents_zh/4_动态规划.md)

#### 基于采样的学习方法

- [第 5 章：蒙特卡洛方法](docs/Contents_zh/5_蒙特卡洛方法.md)
- [第 6 章：时序差分学习](docs/Contents_zh/6_时序差分学习.md)
- [第 7 章：规划、学习与执行](docs/Contents_zh/7_规划_学习_执行.md)

### 🤖 近似解决方法

#### 价值函数近似

- [第 8 章：使用近似方法进行同策略预测](docs/Contents_zh/8_使用近似方法进行同策略预测.md)
- [第 9 章：基于函数近似的同策略控制](docs/Contents_zh/9_基于函数近似的同策略控制.md)

#### 策略近似

- [第 10 章：策略梯度方法](docs/Contents_zh/10_策略梯度方法.md)
- [第 11 章：现代策略梯度方法](docs/Contents_zh/11_现代策略梯度方法.md)

---

## 🤝 参与贡献

英文版和中文版已完成。欢迎通过以下方式让任何一个版本更清晰、更易于阅读：

- **报告问题：** 提交拼写错误、失效链接或表述不清之处。
- **建议改进：** 通过 Pull Request 提交范围明确的修正。
- **帮助更多学习者发现它：** 分享对你最有帮助的章节。

如果 DistilRL 帮你节省了学习时间，欢迎[为仓库加星](https://github.com/datawhalechina/distil-rl-introduction)：方便以后找回来，也能帮助下一位学习者发现它。

## 📜 许可证

本项目使用 [Apache License 2.0](LICENSE)。
