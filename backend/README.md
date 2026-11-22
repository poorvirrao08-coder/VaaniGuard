# VaaniGuard backend

The FastAPI application loads the Kannada-English model from `..\model`.

## Run from the project root

Install the backend dependencies into the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
```

Check `http://127.0.0.1:8010/health` or open the interactive API docs at
`http://127.0.0.1:8010/docs`.
