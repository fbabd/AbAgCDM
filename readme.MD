# AbAgCDM (Antibody-Antigen Contrastive-Discriminative Model)  

AbAgCDM is a sequence-based framework for modeling antibody–antigen binding under antigen sequence variation. The model is designed to capture mutation-induced changes in binding by jointly learning from binding labels and contrastive comparisons across antigen variants for the same antibody.

The framework is motivated by the observation that antibody therapeutics often fail when antigens acquire mutations that disrupt binding. AbAgCDM treats antigen variants as systematic sequence perturbations and learns representations that reflect functional binding differences rather than sequence similarity alone.

---

## Key Features

- Joint encoding of antibody and antigen sequences
- Supervised binding classification (bind / no-bind)
- Variant-aware contrastive learning across antigen mutations
- Evaluation protocols for unseen antibodies and unseen antigen variants
- Sequence-level analysis for identifying mutation-driven binding changes
- Structure-free and computationally efficient

---

## Problem Setting

Given:
- A set of antibody sequences
- Multiple sequence variants of a shared antigen
- Binary binding labels for antibody–antigen pairs

The goal is to:
1. Predict whether an antibody binds a given antigen variant
2. Rank antigen variants by binding probability for a fixed antibody
3. Identify mutations associated with loss of binding (escape)

AbAgCDM formulates this task as a mutation-driven perturbation problem, where antigen variants act as controlled sequence changes applied to a shared interaction system.

---

## Model Overview

The input to the model is a concatenated sequence of the form: `[CLS] Antibody [EOS] Antigen [EOS]`

A pretrained protein language model encoder (ESM2) processes the joint sequence. Two training objectives are applied:

1. **Binding classification loss**  
   A supervised objective for predicting bind or no-bind labels.

2. **Within-antibody contrastive loss**  
   A contrastive objective that compares binding and non-binding antigen variants for the same antibody, encouraging representations to separate by binding outcome rather than sequence identity.

The final training objective is a weighted combination of these two losses.

---

## Evaluation Protocols

AbAgCDM is evaluated under two biologically motivated generalization settings:

- **Unseen antibody generalization**  
  Antibodies in the test set are not observed during training.

- **Unseen antigen variant generalization**  
  Antigen variants are held out using a leave-one-variant-out strategy.

In addition to binary classification metrics, the model is evaluated on its ability to rank antigen variants by binding probability for individual antibodies.

---

## Sequence-Level Analysis

To study mutation-driven binding changes, the framework supports sequence-level analysis based on model attention patterns and prediction shifts across variants. These analyses are intended to highlight regions associated with binding loss or tolerance, rather than provide causal explanations.

---

## Installation

```bash
git clone https://github.com/fbabd/AbAgCDM.git
cd AbAgCDM/AbAgCDM 
pip install -r requirements.txt 
```

Example command for training the model:

```bash
python train.py --config configs.json 
``` 

## Reproducibility

All experiments reported in the paper are run using fixed random seeds and predefined data splits. 

To use the trained model, download the contents of "checkpoint" folder from here https://drive.google.com/drive/folders/1_UGkG5kVkslyhQelLAJ_SJto4moUiIYq?usp=sharing 
and copy the folder inside `AbAgCDM`. 

