NPC Chat Studio (Frontend)
===========================

React-based web UI for interacting with NPC dialogue models.

Overview
--------

A Vite + React + TailwindCSS application providing:

- Multi-NPC chat interface
- Real-time latent state visualization
- Model selection and evaluation metrics
- Episodic memory inspection

Location:: ``slm_training/npc_frontend/``

Quick Start
-----------

Install dependencies::

   cd slm_training/npc_frontend
   pnpm install

Development server::

   pnpm dev          # Vite dev server on http://localhost:5173

Production build::

   pnpm build        # Output to dist/
   pnpm preview      # Preview production build

Architecture
------------

Components
~~~~~~~~~~

App.jsx
   Main application shell. Manages global state (NPCs, messages, current state,
   model catalog), health polling, and layout.

NPCPanel
   Left sidebar. Lists available NPCs, handles add/remove/reset, shows turn counts.

ChatWindow
   Center panel. Displays conversation history, message input, handles sending
   messages to backend API.

StatePanel
   Right sidebar. Visualizes predicted latent state (C_t, A_t, M_t, R_t, N_t, D_t),
   episodic memories, and evaluation metrics.

AddNPCModal
   Modal dialog for creating new NPCs from scenario files.

api.js
   Axios-based API client. Wraps all backend endpoints.

State Management
~~~~~~~~~~~~~~~~

React useState hooks (no Redux)::

   npcs           // List of available NPCs
   activeNpc      // Currently selected NPC ID
   messages       // {npc_id: [message, ...]} chat history
   currentState   // Latest predicted latent state
   memories       // Retrieved episodic memories
   evalData       // Evaluation metrics from backend
   models         // Available trained models
   catalog        // Model catalog from MLflow

API Integration
---------------

The frontend expects a backend API running on ``http://localhost:8765``
(see ``scripts/serve.py``).

Endpoints Used
~~~~~~~~~~~~~~

GET /health
   Polls every 10s for connection status.

GET /v1/models
   Lists available models for selector dropdown.

GET /api/npcs
   Lists NPCs loaded from scenario files.

POST /api/npcs/{npc_id}/chat
   Sends player message, receives NPC response + latent state.

GET /api/eval/summary
   Fetches evaluation metrics for display.

GET /api/catalog
   Loads model catalog from MLflow.

Tech Stack
----------

============== =========== ================================================
Dependency     Version     Purpose
============== =========== ================================================
react          ^18.3.1     UI framework
react-dom      ^18.3.1     DOM renderer
recharts       ^2.12.7     Latent state visualization charts
lucide-react   ^0.441.0    Icons (Wifi, Cpu, ChevronDown, etc.)
axios          ^1.7.7      HTTP client
vite           ^5.4.8      Build tool + dev server
tailwindcss    ^3.4.13     Utility CSS framework
============== =========== ================================================

Configuration
-------------

Vite config (vite.config.js)::

   export default {
     plugins: [react()],
     server: { port: 5173 }
   }

Tailwind config (tailwind.config.js)::

   theme: {
     extend: {
       colors: {
         surface: { 950: '#0f172a', 900: '#1e293b', 800: '#334155' }
       }
     }
   }

Styling
-------

- **Primary accent**: amber-400 (NPC name, highlights)
- **Success state**: emerald-400 / emerald-500 (ready, saved)
- **Warning state**: yellow-400 (encoders only)
- **Error state**: red-400 (offline)
- **Surface colors**: slate/surface scale for dark theme

Responsive Design
~~~~~~~~~~~~~~~~~

- Layout: Flexbox with ``flex-col`` header + ``flex`` main content
- Mobile: Not fully optimized (desktop-first design)
- Panels: Fixed-width sidebars (NPCPanel, StatePanel), fluid ChatWindow

Features
--------

Model Selector Dropdown
   Header button showing trained models with metrics (MSE, PPL).
   Click to expand dropdown, shows "saved" vs "no weights" status.

Connection Status
   Live indicator in header:
   - Red + WifiOff: Offline (can't reach backend)
   - Yellow + Wifi: Encoders only (partial service)
   - Green + Wifi: Ready (full service)

Real-time State Updates
   After each chat turn, StatePanel refreshes with:
   - Predicted latent state components
   - Retrieved episodic memories
   - Response latency (elapsed_ms)

NPC Management
   - Add NPCs from YAML scenario files
   - Remove NPCs from active list
   - Reset conversation history per NPC

Development Notes
-----------------

- Message IDs use global counter (``msgCounter``) for uniqueness
- Health polling runs every 10s, cleans up on unmount
- API errors are caught and logged to console (user feedback minimal)
- Modal state managed by App.jsx, rendered conditionally
