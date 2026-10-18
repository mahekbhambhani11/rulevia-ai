# Rulevia AI

**Natural-language industrial safety intelligence using Gemini and computer vision.**

Rulevia AI is a prototype designed to make industrial safety monitoring more flexible. It allows operators to describe safety requirements in natural language, generate structured rules with Gemini, analyze video footage using computer vision, and review available findings and evidence.

## Live Demo

**Application:** https://rulevia-ai.web.app

**Backend API:** https://rulevia-backend-905825489285.asia-south1.run.app

**API Documentation:** https://rulevia-backend-905825489285.asia-south1.run.app/docs

## Problem

Industrial environments generate large amounts of CCTV footage. Manually reviewing this footage and adapting monitoring rules can be time-consuming and technically demanding.

Rulevia AI explores a workflow that connects natural-language safety instructions with video analysis and reviewable results.

## Key Features

- **Natural-language rule generation:** Use Gemini to convert safety instructions into structured rule fields.
- **Video analysis:** Extract video frames and apply computer-vision object detection.
- **Safety review workflow:** Inspect available findings and supporting evidence.
- **Cloud-hosted application:** Access the web interface and backend through deployed Google Cloud services.

## Technology Stack

- **AI:** Gemini API
- **Computer Vision:** YOLO11n, OpenCV
- **Backend:** Python, FastAPI
- **Frontend:** React, TypeScript, Vite
- **Google Cloud:** Cloud Run, Secret Manager, Cloud Build, Artifact Registry
- **Hosting:** Firebase Hosting

## Architecture

1. The operator describes a safety requirement.
2. Gemini helps generate a structured rule.
3. The backend processes video frames and performs object detection.
4. The application presents available results for review.

## Future Scope

- Real-time CCTV stream monitoring
- Configurable restricted-zone entry detection
- Helmet and personal protective equipment compliance detection
- Real-time alerts and notifications
- Improved evidence storage and auditability

These are planned extensions and should not be interpreted as fully implemented capabilities.

## Getting Started

### Backend

Create a Python virtual environment, install the dependencies from `backend/requirements.txt`, configure the required environment variables using `backend/.env.example`, and start the FastAPI application.

### Frontend

Install the frontend dependencies and run the Vite development server from the `frontend` directory.

Refer to the backend and frontend configuration files for the exact setup requirements.

## Security

Never commit API keys, `.env` files containing secrets, credentials, or confidential video footage. Use Google Cloud Secret Manager for deployed secrets.

## Project Status

Rulevia AI is a hackathon prototype. Detection accuracy and safety decisions require validation before use in real industrial environments.

