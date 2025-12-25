#!/usr/bin/env bash
gunicorn smart_scan.app:app --bind 0.0.0.0:$PORT
