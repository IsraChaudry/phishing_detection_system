# AI Phishing Detection — Demo

This project provides a Streamlit frontend for a DistilBERT-based phishing detector.

Quick demo (recommended for presentation / deadline):

1. Activate the venv (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

2. Start the demo UI (demo mode uses a lightweight heuristic and does not require heavy ML libs):

```powershell
streamlit run app_working.py
```

3. In the sidebar, keep "Demo mode (no model)" checked to run without the model. Uncheck it to load the real model (requires PyTorch + Transformers installed and a valid `phishing_model_deployment` folder).

To install the CPU PyTorch and Transformers (optional, for real model):

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\venv\Scripts\python.exe -m pip install transformers
```

Notes
- The app includes a copy-to-clipboard button next to the recommended action for quick demo usage.
- If Torch fails to load due to DLL issues on Windows, keep Demo mode enabled and proceed with the frontend.
