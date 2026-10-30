# Fairness Over Time

Does a model that is fair when deployed stay fair as data changes over time?
Using Folktables ACSIncome (California, 2014–2018), I train models on 2014,
monitor accuracy and fairness through 2018, diagnose why fairness changes,
and compare maintenance strategies.

## Setup

conda env create -f environment.yml
conda activate fairness
pip install -e .

## Structure

- `src/`: reusable code (data, preprocessing, models, fairness, drift, explanations)
- `notebooks/`: one notebook per phase (01_baseline … 04_fix)
- `artifacts/`: saved models and results read by the app
- `app/`: Streamlit app (never trains, only reads artifacts)
