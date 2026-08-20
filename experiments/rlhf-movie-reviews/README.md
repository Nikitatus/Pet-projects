# RLHF for Movie Reviews

An experiment in aligning GPT-2 to generate negative movie reviews using RLHF techniques.

The workflow has two stages:

1. Train a reward model on synthetic preference pairs.
2. Fine-tune the language model with PPO to maximize the learned reward.
