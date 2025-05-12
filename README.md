# Generative Model-based Collective Variable Learning with Metastable State Identification

This repository contains code, data, and experiments related to our proposed generative model-based framework for learning physically interpretable collective variables (CVs) in molecular dynamics (MD) systems.

## 🔬 Overview

We introduce a **generative modeling approach** to construct collective variables that:
- Faithfully capture the metastable states in high-dimensional MD data
- Eliminate the need for hand-crafted feature engineering
- Ensure one-to-one correspondence between latent prior modes and MD metastable states

Our framework leverages:
- A prior distribution with **predefined local maxima**
- An invertible neural mapping to reconstruct the full-dimensional PDF
- A hybrid loss function to enforce **mode alignment** and **kinetic fidelity**

We demonstrate the method on **alanine dipeptide in aqueous solution**, where the learned CVs reproduce key metastable states and correctly reflect kinetic properties such as mean square displacement behavior.

# Project Structure: Generative CV Learning for MD + Müller–Brown

```
generative-metastable/
├── muller_model/                # Müller–Brown potential (toy model)
│   ├── train_model.py
│   ├── Muller.ipynb             # Loading model and plotting some results
│   └── checkpoints/
│
├── md_model/                     # Alanine dipeptide (real MD)
│   ├── train_md_model.py               
│   └── checkpoints/           
│
├── notebooks/                   # Jupyter notebooks (optional)
│   ├── md_analysis.ipynb
│   └── muller_visualization.ipynb
│
├── requirements.txt             # Minimal: tensorflow, tfp, numpy
├── runtime.txt                  # Binder/Colab Python version
├── .gitignore                   # Ignore checkpoints, data, etc.
└── README.md
```