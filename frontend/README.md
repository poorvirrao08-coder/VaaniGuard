# VaaniGuard frontend

This React/Vite application sends messages to the FastAPI backend at
`http://127.0.0.1:8010`.

## Run locally

From this folder, install dependencies once and start the development server:

```powershell
npm.cmd ci
npm.cmd run dev -- --host 127.0.0.1
```

Open the local URL printed by Vite. Start the backend separately; see
[`../backend/README.md`](../backend/README.md).

## Validate

```powershell
npm.cmd run lint
npm.cmd run build
```
