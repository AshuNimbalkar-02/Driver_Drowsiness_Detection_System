# Driver Drowsiness Detection System



## Project Overview

Driver Drowsiness Detection System is a full-stack safety monitoring application built to detect driver fatigue and log safety events in real time. The project combines a React-based frontend for camera-driven attention tracking with an Express backend API that stores event logs and simulates fleet sync behavior.

## Features

- Real-time drowsiness and alertness monitoring using webcam input
- Safety event logging for driver status, EAR/MAR values, and alerts
- Backend log storage in a lightweight JSON database
- Fleet synchronization endpoint for simulated remote safety monitoring
- Smooth, responsive UI built with React and modern frontend libraries

## Technology Used

- Frontend
  - React
  - Vite
  - React Webcam
  - MediaPipe Face Mesh
  - Recharts
  - Framer Motion
  - Howler
  - Lucide Icons
  - Tailwind Merge

- Backend
  - Node.js
  - Express
  - CORS
  - Morgan

- Root / tooling
  - concurrently
  - npm scripts for multi-service startup

## Project Structure

- `frontend/` - React app and UI source code
- `backend/` - Express server, API routes, and logs storage
- `backend/data/logs.json` - persistent safety log database file
- `package.json` - root scripts for install and running both services

## Requirements

- Node.js 18+ (or latest stable Node)
- npm

## Local Setup

1. Open a terminal at the repository root.
2. Install both frontend and backend dependencies:

```bash
npm run install:all
```

## Development Run

Start both the frontend and backend together:

```bash
npm run dev
```

This launches:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:5000`

If you want to run services separately:

```bash
npm run dev:frontend
npm run dev:backend
```

## Backend API Endpoints

- `GET /api/logs` - fetch all logged events
- `GET /api/logs/:fleetId` - fetch logs filtered by fleet ID
- `POST /api/logs` - create a new safety log event
- `POST /api/sync` - simulate fleet sync with server

## Build for Production

To build the frontend app:

```bash
cd frontend
npm run build
```

## Notes

- `backend/data/logs.json` stores safety logs locally.
- The backend defaults to port `5000`.

Enjoy exploring the Driver Drowsiness Detection System and customizing the model and UI for your fleet safety use case.

# Driver_Drowsiness_Detection_System
