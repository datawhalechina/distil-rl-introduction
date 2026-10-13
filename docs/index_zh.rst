:orphan:

DistilRL 强化学习简明导论：从 Sutton & Barto 到 PPO 与 GRPO
====================================================================

.. raw:: html

   <p class="language-switch"><a href="index.html" hreflang="en" lang="en">English</a> | <strong>中文</strong></p>
   <p class="distilrl-hero-cta"><a href="Contents_zh/0_%E5%89%8D%E8%A8%80.html" class="my-button" data-seo-event="learning_path_start" data-seo-location="homepage"><strong>从基础开始</strong></a> <a href="Contents_zh/3_%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B.html" class="my-button" data-seo-event="learning_path_accelerated" data-seo-location="homepage"><strong>逐步学到 PPO 与 GRPO</strong></a></p>

双语、可视化的 12 章学习路径
------------------------------------

**DistilRL 将 Sutton 与 Barto 的《强化学习导论》和 University of Alberta 强化学习课程中的部分核心内容重新筛选、组织成一条更精简的概念路径。** 它从多臂老虎机和 MDP 出发，经过时序差分学习与函数近似，最后将这些基础连接到策略梯度、PPO 与 GRPO。

本教程已提供完整的中英文版。它是一份阅读指南和概念学习伴侣，不是代码实战课，也不以完整替代原教材为目标。

选择适合你的阅读入口
------------------------

* **刚开始学强化学习？** :doc:`从第 0 章开始 <Contents_zh/0_前言>`，按顺序阅读全部 12 章。
* **为 PPO、GRPO 或大模型后训练补强化学习基础？** :doc:`从第 3 章的 MDP 基础开始 <Contents_zh/3_马尔可夫决策过程>`，再依次阅读第 6、8、10、11 章。

适合谁阅读
------------

* 希望先理清强化学习主线，再深入阅读原教材的学习者。
* 希望补齐策略梯度、PPO 与 GRPO 概念基础的 ML 和 LLM 工程师。
* 希望在同一条学习路径中对照中英文内容的读者。

12 章学习路径
----------------

**简单决策 → 序列决策 → 已知模型求解 → 从样本中学习 → 时序差分 → 函数近似 → 直接策略优化 → PPO 与 GRPO。**

I. 引言
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: I. 引言

   Contents_zh/0_前言
   Contents_zh/1_强化学习简介


II. 表格方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: II.1. 强化学习基础

   Contents_zh/2_多臂老虎机问题.md
   Contents_zh/3_马尔可夫决策过程.md
   Contents_zh/4_动态规划.md

.. toctree::
   :maxdepth: 1
   :caption: II.2. 基于采样的学习方法

   Contents_zh/5_蒙特卡洛方法.md
   Contents_zh/6_时序差分学习.md
   Contents_zh/7_规划_学习_执行.md


III. 近似方法
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: III.1. 价值函数近似

   Contents_zh/8_使用近似方法进行同策略预测.md
   Contents_zh/9_基于函数近似的同策略控制.md


.. toctree::
   :maxdepth: 1
   :caption: III.2. 策略近似

   Contents_zh/10_策略梯度方法.md
   Contents_zh/11_现代策略梯度方法.md

随时找回 DistilRL
------------------

如果这份教程帮你建立了更清晰的强化学习知识地图，欢迎在 GitHub 上 `为 DistilRL 加星 <https://github.com/datawhalechina/distil-rl-introduction>`_。这样既方便以后找回来，也能帮助下一位学习者发现它。

`阅读英文版 <index.html>`_ | `查看源码并参与贡献 <https://github.com/datawhalechina/distil-rl-introduction>`_
