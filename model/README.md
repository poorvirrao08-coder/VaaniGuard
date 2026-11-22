# VaaniGuard model and data

This folder contains the model-training, evaluation, and audit scripts, their datasets,
and the saved model files used by the backend.

## Install model dependencies

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r model\requirements.txt
```

## Prepare and train the Kannada-English model

```powershell
.\.venv\Scripts\python.exe model\prepare_kannada_data.py
.\.venv\Scripts\python.exe model\train_kannada_mixed.py
```

The training script writes `vaaniguard_kannada_baseline.pkl` in this folder. The backend
loads that model when it starts.

## Train the English baselines

```powershell
.\.venv\Scripts\python.exe model\train_baseline.py
.\.venv\Scripts\python.exe model\train_grouped_baseline.py
```

Other `audit_*.py`, `inspect_*.py`, and `evaluate_*.py` scripts operate on the datasets
in this folder. Script paths are resolved relative to this folder, so commands can be
run from the project root.
