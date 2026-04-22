# W Boson Event Classification Using Machine Learning

Masters Project — [Panjab University,Chandigarh] — [2026]

## Overview
This repository contains all code for my Masters dissertation:
"Identification of W Boson Events in CMS Collision Data 
Using Machine Learning"

## Results
- AUC-ROC: 0.9849
- Signal efficiency: 96.0%
- Background rejection: 93.9%

## Repository Structure

## Requirements
```bash
pip install uproot awkward numpy pandas scikit-learn 
           matplotlib seaborn scipy joblib
```

## How to Run
```bash
# Step 1: Convert ROOT files to CSV
python3 data_processing/simulated_data.py
python3 data_processing/z_background.py
python3 data_processing/merge_files.py

# Step 2: Train the classifier
python3 training/train_simulation.py

# Step 3: Evaluate performance
python3 evaluation/feature_removal.py
python3 evaluation/crossval_graph.py
python3 evaluation/learning_curve.py

# Step 4: Apply to real data
python3 results/graph6_real_data.py
```

## Data
Signal and background MC samples from CERN Open Data Portal:
https://opendata.cern.ch

## Supervisor
Dr. Vipin Bhatnagar, Department of Physics, Panjab University, Chandigarh


