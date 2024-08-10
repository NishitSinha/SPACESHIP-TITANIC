# SPACESHIP-TITANIC

This repository contains a machine learning project focused on predicting passenger survival in a fictional interstellar incident aboard the Spaceship Titanic, using the dataset provided by the Kaggle competition.

## Overview

The project involves:

- **Data preprocessing:** Handling missing values, feature engineering, and encoding categorical variables.
- **Model training and testing:** Using models like Random Forest, XGBoost, and Neural Networks.
- **Evaluation:** Assessing model performance using accuracy, precision, and AUC-ROC metrics.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NishitSinha/SPACESHIP-TITANIC.git
   cd SPACESHIP-TITANIC
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place the `train.csv` and `test.csv` files in the repository directory.
2. Run the `spaceship-titanic.ipynb` notebook to preprocess data, train models, and generate predictions.

## Results

Model predictions are saved in the `subm.csv` file.

## Dependencies

- pandas
- numpy
- scikit-learn
- xgboost
- tensorflow
- matplotlib

## Acknowledgements

- [Kaggle Spaceship Titanic Competition](https://www.kaggle.com/c/spaceship-titanic)
