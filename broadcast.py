# broadcast.py
# Simple in-memory WebSocket broadcast manager for development.
# For production replace with Redis pub/sub (notes below).

import asyncio
from typing import Set
from fastapi import WebSocket

class BroadcastManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast_json(self, message: dict):
        """
        Broadcast a JSON-serializable dict to all connected websockets.
        Keeps errors isolated per connection.
        """
        to_remove = []
        async with self._lock:
            conns = list(self.active_connections)
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                # mark for removal
                to_remove.append(ws)
        if to_remove:
            async with self._lock:
                for ws in to_remove:
                    if ws in self.active_connections:
                        self.active_connections.remove(ws)

# single global instance imported by main.py and crud.py
broadcast_manager = BroadcastManager()
