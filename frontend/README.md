# VYOM Lunar Image Registration — Mission Control Frontend

Modern, high-performance telemetry dashboard and interactive visualizer for the **VYOM** lunar image registration pipeline.

Built with **React 19**, **Vite**, **TypeScript**, and **Tailwind CSS**.

---

## Features

- **Interactive Telemetry Dashboard:** Live visualization of Chandrayaan-2 and reference orbiter imagery.
- **Dual Visual Modes:** Side-by-side comparison, alpha blending, difference map, and split view.
- **Keypoint Telemetry Radar:** Real-time multi-dimensional radar chart showing spatial distribution, inlier consensus, and reprojection fidelity.
- **Algorithm Comparison Suite:** Run SIFT, AKAZE, and RIFT2-style phase congruency simultaneously on identical input pairs with comparative metrics tables.
- **DEMO Dataset Browser:** Instant evaluation of real Chandrayaan-2 OHRC vs NASA LROC NAC lunar crater pairs and synthetic benchmark sets.

---

## Getting Started

### Prerequisites
- Node.js (v18.0 or higher)
- npm (v9.0 or higher)

### Installation
```bash
# From the frontend directory
npm install
```

### Environment Configuration
The frontend communicates with the FastAPI registration backend (`http://localhost:8000` by default).

To configure custom endpoints, copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Ensure `VITE_API_BASE_URL` points to your active backend service:
```env
VITE_API_BASE_URL=http://localhost:8000
```

### Development Server
```bash
npm run dev
```
The interface will be accessible at [http://localhost:3000](http://localhost:3000) (or the port specified by Vite).

### Production Build
```bash
npm run build
npm run preview
```
