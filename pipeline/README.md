# Smart Bin – Software for the Bin

This repository contains the bin-side software for the Smart Waste Segregation system.

## Features
- Image preprocessing with quality checks
- AI classification (stub)
- Fail-safe fallback to mixed waste
- Fill level tracking
- Event-based backend integration
- API communication aligned with mobile app backend

## How to run (local test)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline_test.py


Note:
- API sending is disabled by default (change SEND_API_EVENTS to True)
- Uses simulated fill levels
- AI model is currently a stub
