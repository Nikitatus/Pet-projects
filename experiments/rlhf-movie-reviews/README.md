# RLHF for Movie Reviews

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nikitatus/Pet-projects/blob/main/experiments/rlhf-movie-reviews/rlhf_movie_reviews.ipynb)

An experiment in aligning GPT-2 to generate negative movie reviews using RLHF techniques.

The workflow has two stages:

1. Train a reward model on synthetic preference pairs.
2. Fine-tune the language model with PPO to maximize the learned reward.
