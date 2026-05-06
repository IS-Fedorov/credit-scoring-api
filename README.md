# Credit Scoring API

This is a machine learning project for credit risk prediction.

The project includes data preprocessing, feature engineering, model training, evaluation, saving the pipeline, and using it through a FastAPI endpoint.

## Project structure

- `notebook/` — main notebook with data processing, experiments, model training, and evaluation
- `api/` — FastAPI application for making predictions
- `models/` — saved trained pipeline
- `src/` — custom transformer classes used inside the saved pipeline
- `data/` — expected local data folder, not uploaded to GitHub

## What I used

- Python
- pandas
- numpy
- scikit-learn
- LightGBM
- Optuna
- FastAPI
- joblib

## Data

The full dataset is not included in this repository because of its size.

To run the full notebook, place files into:
- data/raw/train_data.zip
- data/raw/train_target.csv