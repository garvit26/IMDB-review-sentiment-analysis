# Sentiment Analysis with a Transformer Network

This project implements a Transformer-based neural network from scratch to perform sentiment analysis on the IMDB movie review dataset. The goal is to classify movie reviews as either "positive" or "negative."

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Project Description

This notebook demonstrates how to build and train a Transformer network for a natural language processing (NLP) task. Unlike traditional recurrent neural networks (RNNs) or LSTMs, the Transformer architecture relies entirely on self-attention mechanisms to process sequential data. This approach allows for significant parallelization and has become the foundation for most state-of-the-art NLP models.

The key steps in this project are:
1.  Loading and preprocessing the IMDB movie review dataset.
2.  Creating custom Keras layers for token and positional embeddings.
3.  Building a Transformer encoder block, which includes Multi-Head Self-Attention and a Feed-Forward Network.
4.  Assembling the full classifier model.
5.  Training the model and evaluating its performance on the test set.

## Dataset

The project uses the **IMDB movie review dataset**, which is a standard benchmark for sentiment analysis. It consists of 50,000 highly polarized movie reviews that are split into 25,000 for training and 25,000 for testing. Each review is labeled as either positive (1) or negative (0).

A vocabulary size of 20,000 unique words is used, and reviews are padded to a maximum length of 200 tokens.

## Model Architecture

The core of this project is the Transformer encoder architecture, which is composed of the following custom layers:

1.  **Token and Position Embedding Layer (`TokenAndPositionEmbedding`)**:
    -   **Token Embedding**: Converts integer-based token inputs into dense vector representations (embeddings).
    -   **Positional Embedding**: Since the model does not process data sequentially, it injects information about the position of each token in the sequence. These positional embeddings are added to the token embeddings.

2.  **Transformer Encoder Block (`TransformerBlock`)**: This is the main building block of the network.
    -   **Multi-Head Self-Attention (`MultiHeadSelfAttention`)**: Allows the model to weigh the importance of different words in the input sequence when processing a specific word. The "multi-head" mechanism allows it to focus on different parts of the sequence simultaneously.
    -   **Feed-Forward Network**: A simple, fully connected feed-forward network applied to each position separately.
    -   **Layer Normalization & Dropout**: Applied after both the attention and feed-forward sub-layers for regularization and to stabilize training.

The final classifier model stacks the embedding layer, a Transformer block, and a few dense layers to produce the final binary classification output.

## Installation

To run this notebook, you need Python and the following libraries. You can install them using `pip`:

```bash
pip install tensorflow numpy
