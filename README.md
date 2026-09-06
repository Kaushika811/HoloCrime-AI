# HoloCrime AI

## Member 1 – AI Surveillance and Event Detection

This module performs automated CCTV surveillance and detects unauthorized entry into a restricted zone.

## Features

- Video input using OpenCV
- Person detection using YOLO
- Person tracking using ByteTrack
- Restricted zone monitoring
- Unauthorized entry detection
- Evidence screenshot capture
- Event video recording
- CSV event logging

## System Workflow

Video Input
↓
Person Detection
↓
Person Tracking
↓
Restricted Zone Detection
↓
Unauthorized Entry Detection
↓
Evidence Screenshot
↓
Event Video Clip
↓
CSV Event Log

## Installation

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Run

python scripts/event_logger_csv.py

## Output

The system generates:

- Event screenshots
- Event video clips
- CSV event logs

## Member 2 Handover

Member 2 should use the generated event video as input for 3D reconstruction and evidence mapping.

Generated event videos are stored in:

output/event_videos/