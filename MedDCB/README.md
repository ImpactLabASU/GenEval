# DCB WACV Demo (Public Web Version)

Public, GitHub-ready web demo for Domain Conformal Bounds (DCB).

## Included Files

- `app.py`: Flask backend for upload -> theta extraction -> SDCD analysis
- `template/index.html`: UI
- `dcb_core.py`: Mahalanobis bounds and coverage logic
- `run_app.py`: optional non-debug launcher
- `requirements.txt`: dependencies

## Setup

```bash
cd version-latest/final-code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 app.py
```

Open: `http://localhost:8080`

## Dataset Upload Format

Upload two dataset folders in the UI (Dataset A and Dataset B).  
Each folder must contain:

- Image files (nested subfolders are supported)
- Exactly one CSV file

CSV must include an image identifier column such as:

- `image_id`
- `image`
- `filename`
- `path`

Identifier values should match uploaded image filenames or relative paths.

## Notes

- No hardcoded absolute system paths are required.
- Temporary files are created under `dcb_tmp/`.
