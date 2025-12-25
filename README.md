# Smart Scan — PDF Tool

Lightweight Flask app for scanning and processing PDFs/images in a browser.

## Project structure

- `app.py` — Flask application entrypoint (runs the web server).
- `scanner.py` — scanning / processing helper logic used by the app.
- `requirements.txt` — Python dependencies.
- `templates/` — HTML templates (includes `index.html`).
- `static/` — static assets (JS/CSS).

## Prerequisites

- Python 3.8 or newer
- `pip` (Python package installer)

## Install

1. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Start the application with:

```bash
python app.py
```

By default the app runs on `http://127.0.0.1:5000` — open that URL in your browser.

## Usage

- The web UI is served from `templates/index.html`.
- The front-end uses assets in `static/` (e.g., `fabric.min.js`).
- Scanning and processing logic lives in `scanner.py` and is invoked by `app.py`.

## Troubleshooting

- If the port is already in use, change the port in `app.py` or stop the other service.
- If dependencies fail to install, ensure your `pip` is up-to-date:

```bash
python -m pip install --upgrade pip
```

