# Genetic Algorithm Feature Selection for Trial Completion Time

This project applies a **Genetic Algorithm (GA)** to a large multi-sensor dataset to identify which features have the greatest impact on **task completion time**. The goal is to improve model performance and interpretability by selecting the most informative subset of features from a high-dimensional dataset.

---

## 🚀 Project Overview

Modern sensor datasets often include **hundreds of potential features**, many of which may be irrelevant or redundant. This project:

- Uses a **Genetic Algorithm** to evolve subsets of features over generations
- Trains a predictive model to estimate completion time
- Scores each chromosome based on model error (fitness function)
- Identifies **top-ranking features** that drive performance
- Produces interpretable results for downstream modeling

---

## 📁 Repository Structure

```plaintext
Genetic-Algo/
│
├── src/
│   ├── data_cleaning/        # Scripts for preprocessing and feature engineering
│   ├── feature_selection/    # GA implementation and fitness evaluation
│   ├── models/               # Machine learning models for evaluation
│   └── utils/                # Helper functions
│
├── requirements.txt          # Dependencies to install
├── .gitignore                # Excludes large data + build files
└── README.md                 # Project documentation
