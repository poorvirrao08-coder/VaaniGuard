# VaaniGuard

VaaniGuard is a prototype for detecting scams in payment-related messages. It has a
React frontend, a FastAPI backend, and Python scripts for training and evaluating
the machine-learning models.

The backend uses an experimental Kannada-English model. Its predictions are not a
guarantee that a message is safe and should not be used as the sole security or
financial decision-maker.

## Project layout

```text
VaaniGuard/
|-- backend/    FastAPI application and backend requirements
|-- frontend/   React/Vite user interface
|-- model/      Training scripts, datasets, and saved model files
|-- .venv/      Local Python virtual environment (created during setup)
`-- README.md
```

See the folder READMEs for details: [backend](backend/README.md),
[frontend](frontend/README.md), and [model](model/README.md).

## Requirements

- Python 3.10 or newer
- Node.js and npm
- Windows PowerShell examples below assume the repository root is the current folder

## First-time setup

Open PowerShell in the `VaaniGuard` project root. Create a Python environment and
install the backend and model dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r model\requirements.txt
```

Install the frontend dependencies once:

```powershell
cd frontend
npm.cmd ci
cd ..
```

## Run the application

Start the backend and frontend in **two separate terminals**.

### Terminal 1: API

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

The backend loads `model\vaaniguard_kannada_baseline.pkl` at startup.

### Terminal 2: frontend

```powershell
cd frontend
npm.cmd run dev -- --host 127.0.0.1
```

Open the frontend URL printed by Vite, normally
`http://127.0.0.1:5173`. Keep both terminals open while using the app. Press
`Ctrl+C` in each terminal to stop its server.

Useful backend URLs:

- Health check: `http://127.0.0.1:8010/health`
- Interactive API documentation: `http://127.0.0.1:8010/docs`
- Message analysis: `POST http://127.0.0.1:8010/analyze`

## Train or test models

Run these commands from the project root. Prepare and train the Kannada-English
model with:

```powershell
.\.venv\Scripts\python.exe model\prepare_kannada_data.py
.\.venv\Scripts\python.exe model\train_kannada_mixed.py
```

Training writes the model to `model\vaaniguard_kannada_baseline.pkl`; restart the
backend after retraining so it loads the new file. To test it interactively:

```powershell
.\.venv\Scripts\python.exe model\test_kannada_model.py
```

English baseline training commands:

```powershell
.\.venv\Scripts\python.exe model\train_baseline.py
.\.venv\Scripts\python.exe model\train_grouped_baseline.py
```

The other scripts in `model/` inspect datasets, audit splits, and evaluate model
behavior. See [model/README.md](model/README.md) for more.

## Troubleshooting

- **Frontend cannot connect:** confirm the API is running on port `8010`, then
  refresh the frontend.
- **Model file not found:** check that `model\vaaniguard_kannada_baseline.pkl`
  exists; train it using the commands above if needed.
- **Missing Python packages:** rerun the `pip install -r ...` commands from the
  project root.
- **Frontend packages missing:** run `npm.cmd ci` from the `frontend/` folder.

## Data and limitations

- The English/Sentinel dataset contains 14,000 records.
- The Kannada/English pilot contains 180 records, including synthetic training
  samples. Native Kannada speakers should review these samples before production
  use.
- Accuracy measured on the small pilot may not generalize to real-world messages.
- This prototype has no database or deployment pipeline.
