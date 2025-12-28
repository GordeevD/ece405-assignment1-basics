# UHM ECE 405 Spring 2026 Assignment 1: Basics

This asignment is created from Assignment 1 of [CS336 at Stanford taught in Spring 2025](https://stanford-cs336.github.io/spring2025/). 
For the full description of the original assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

Check out useful [lectures from CS336 at Stanford](https://github.com/stanford-cs336/spring2025-lectures).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix. Any improvements of the existing codebase
(including adaptations from Stanford to UHM workflows, modifications of PDF, etc) will be rewarded with extra points.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### [Click here](https://colab.research.google.com/drive/10qZA9UZwdm52ecPP1UrZmYIKHqpJoH51?usp=sharing) for an example setup at Colab



Caution! The free GPU runtimes are very limited! Make sure to disconnect and delete your runtime when you spend time writing code or switch to another task. Using colab GPU runtimes for too long might result in losing access to them  (inceased wait times and/or short session durations).

If any of this happens to you, please consult with the professor.

## ECE405 Assignment instructions

Follow along the [CS336@Stanford handout](./cs336_spring2025_assignment1_basics.pdf) with small deviations:
1. What the code looks like: clone https://github.com/igormolybog/ece405-assignment1-basics.git
2. What you can use: Implementation from scratch is preferred, but experiments are essential. If you are stuck with some implementation, just use the Huggingface/Pytorch implementation and proceed to the experiments
    - Submit the report reflecting your attempts at implementation for partial credit
3. How to submit: You will submit the report on the assignment to [Assignment Submission Form](https://docs.google.com/forms/d/e/1FAIpQLScJg_QkwjKux3xKeM-EOmZyvA6zlbVIrf_lxN_qoCFoxdqTrg/viewform?usp=sharing&ouid=111841773839267096112). The code does not have to be attached as long as you include links to the main GitHub branch where your code lives and links to all of the Colab notebooks if applicable.
    - You don't need to submit to leaderboard.
4. Problems (learning_rate, batch_size_experiment, parallel_layers, layer_norm_ablation, pre_norm_ablation, main_experiment):
    - get a free T4 GPU at Colab
    - reduce the number of total tokens processed down to 33,000,000 or even lower for faster iteration. Keep the number of tokens consistant across your experiments.
5. Problem (learning_rate):
    - validation loss can be anything
6. Skip Problem (leaderboard) from Section 7.5

