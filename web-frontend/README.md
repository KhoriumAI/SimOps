# Web Frontend

The primary web-based user interface for Khorium MeshGen, built with React, Vite, and Tailwind CSS.

## Technology Stack

- **Framework**: [React](https://reactjs.org/)
- **Build Tool**: [Vite](https://vitejs.dev/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **3D Rendering**: [Three.js](https://threejs.org/) with `@react-three/fiber`
- **State Management**: [Zustand](https://github.com/pmndrs/zustand) (if applicable) or React Context

## Getting Started

### Installation
```bash
cd web-frontend
npm install
```

### Development
Runs the app in development mode with hot-reloading:
```bash
npm run dev
```

### Build
Builds the app for production to the `dist` folder:
```bash
npm run build
```

## Structure
- `src/components`: Reusable UI components.
- `src/components/viewer`: 3D visualization logic and mesh rendering.
- `src/api`: Axios/Fetch logic to communicate with the Python backend.
- `src/store`: Global state management.

## Environment Variables
Copy `.env.production.example` to `.env.local` and configure your backend API URL:
```env
VITE_API_URL=http://localhost:8000
```
