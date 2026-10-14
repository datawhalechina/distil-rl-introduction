DistilRL: Reinforcement Learning from Sutton & Barto to PPO and GRPO
====================================================================

.. raw:: html

   <p class="language-switch"><strong>English</strong> | <a href="index_zh.html" hreflang="zh-CN" lang="zh-CN">中文</a></p>
   <p class="distilrl-hero-cta"><a href="Contents/0_prelude.html" class="my-button" data-seo-event="learning_path_start" data-seo-location="homepage"><strong>Start from the foundations</strong></a> <a href="Contents/3_markov_decision_process.html" class="my-button" data-seo-event="learning_path_accelerated" data-seo-location="homepage"><strong>Build toward PPO &amp; GRPO</strong></a></p>

A Visual, Bilingual 12-Chapter Learning Path
---------------------------------------------

**DistilRL reorganizes selected ideas from Sutton and Barto's** *Reinforcement Learning: An Introduction* **and the University of Alberta reinforcement-learning curriculum into a shorter conceptual route.** It moves from multi-armed bandits and MDPs through temporal-difference learning and function approximation, then connects those foundations to policy gradients, PPO, and GRPO.

The guide is available in complete English and Chinese editions. It is a reading guide and conceptual companion, not a code-first course or a replacement for the complete textbook.

Start with the Route That Fits You
----------------------------------

* **New to reinforcement learning?** :doc:`Start with Chapter 0 <Contents/0_prelude>` and follow all 12 chapters in order.
* **Learning RL for PPO, GRPO, or LLM post-training?** :doc:`Start with the MDP foundation in Chapter 3 <Contents/3_markov_decision_process>`, then continue through Chapters 6, 8, 10, and 11.

Who This Guide Is For
---------------------

* Learners who want the main conceptual thread before going deeper into the original textbook.
* ML and LLM engineers who want the reinforcement-learning foundations behind policy gradients, PPO, and GRPO.
* English and Chinese readers who want to compare two editions of the same learning path.

The 12-Chapter Route
--------------------

**Simple decisions → sequential decisions → planning with a model → learning from samples → temporal-difference learning → function approximation → direct policy optimization → PPO and GRPO.**

I. Introduction
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: I. Introduction

   Contents/0_prelude
   Contents/1_intro


II. Tabular Solution Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: II.1. Fundamentals of Reinforcement Learning

   Contents/2_multi_armed_bandits
   Contents/3_markov_decision_process
   Contents/4_dynamic_programming

.. toctree::
   :maxdepth: 1
   :caption: II.2. Sample-based Learning Methods

   Contents/5_monte_carlo_methods
   Contents/6_temporal_difference_learning
   Contents/7_planning_learning_acting


III. Approximate Solution Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :caption: III.1. Value Function Approximation

   Contents/8_on_policy_prediction_with_approximation
   Contents/9_on_policy_control_with_approximation


.. toctree::
   :maxdepth: 1
   :caption: III.2. Policy Approximation

   Contents/10_policy_gradient_methods
   Contents/11_modern_policy_gradient_methods

Keep DistilRL Within Reach
--------------------------

If this guide helps you build a clearer mental map of reinforcement learning, `star DistilRL on GitHub <https://github.com/datawhalechina/distil-rl-introduction>`_. It makes the guide easier to find again and helps the next learner discover it.

`Read the Chinese edition <index_zh.html>`_ | `View the source and contribute <https://github.com/datawhalechina/distil-rl-introduction>`_
