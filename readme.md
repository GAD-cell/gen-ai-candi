# CANDI: Hybrid Discrete-Continuous Diffusion Implementation

This repository contains a PyTorch implementation of the CANDI (Continuous ANd DIscrete diffusion) framework applied to a Llama-based architecture. This project aims to replicate and validate the effectiveness of hybrid diffusion models for text generation, specifically trained on the CodeParrot dataset.

## Project Overview

Standard continuous diffusion models often fail when applied directly to discrete domains like text due to the "temporal dissonance" between discrete token identifiability and continuous geometric learning. CANDI addresses this by decoupling the discrete corruption (masking) from the continuous corruption (Gaussian noise).

This implementation adapts a standard Llama model to function as a non-autoregressive diffusion backbone by removing the causal masking mechanism and implementing the specific forward and reverse diffusion processes described in the CANDI paper.

## Repository Structure

The project is organized as follows:

- **src/model/model.py**: Contains the `CandiLlama` class. This is the core logic that wraps the Llama backbone, implements the hybrid forward process (noising), and the reverse diffusion sampling loop (denoising).
- **src/train/pretrain.py**: The main training script. It handles data loading (CodeParrot), the training loop, loss computation, and checkpointing using Weights & Biases (WandB) for logging.
- **src/model_perplexity.py**: Evaluation script that calculates the Generative Perplexity of the model using a pre-trained GPT-2 model as a judge.
- **src/test_model.py**: A utility script to verify the forward pass and perform simple inference tests to ensure the model architecture is functioning correctly.

## Installation

To set up the environment, ensure you have Python installed. You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
make env
```

## Training 
```bash
make pretraining
```
