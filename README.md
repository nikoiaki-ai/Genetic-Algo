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
│   ├── features/                  # Feature extraction from eye, IMU, and shimmer sensors
│   ├── parsers/                   # Parses raw data from trials and sensor recordings
│   ├── utils/                     # Common helper functions for loading and formatting data
│   │
│   ├── append_eye_features.py     # Generates eye-tracking derived predictors
│   ├── append_imu_features.py     # Generates motion-based predictors from IMU data
│   ├── append_shimmer_features.py # Generates physiological predictors from Shimmer sensors
│   ├── build_table.py             # Builds the final training dataset from all features
│   ├── fix_task_stage.py          # Cleans and normalizes trial stage labels
│   └── ga_rf_select.py            # Genetic Algorithm + Random Forest feature selection
│
├── requirements.txt          # Dependencies to install
├── .gitignore                # Excludes large data + build files
└── README.md                 # Project documentation

> The dataset and output folders are excluded via `.gitignore` due to size and sensitivity.

---

## 🧪 Method Summary

✔ Parse multi-sensor data  
✔ Extract meaningful biomechanical and cognitive workload features  
✔ Train a model to predict trial completion time  
✔ Use a Genetic Algorithm to select high-value feature combinations  
✔ Rank features by contribution to performance

This helps reveal which physiological or behavioral signals **drive task efficiency**.
