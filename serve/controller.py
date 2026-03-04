"""
Emomni Controller
Manages distributed model workers and routes requests.
Based on FastChat controller architecture.
"""

import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import time
from typing import List, Dict, Optional
import threading

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import numpy as np
import requests
import uvicorn

from .constants import (
    CONTROLLER_HEART_BEAT_EXPIRATION,
    DEFAULT_CONTROLLER_PORT,
    SERVER_ERROR_MSG
)
from .utils import build_logger

logger = build_logger("controller", "controller.log")


class DispatchMethod(Enum):
    """Methods for dispatching requests to workers."""
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name: str):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method: {name}")


@dataclasses.dataclass
class WorkerInfo:
    """Information about a registered worker."""
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: float
    worker_address: str = ""


def heart_beat_controller(controller: "Controller"):
    """Background thread to check worker heartbeats."""
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stale_workers_by_expiration()


class Controller:
    """
    Controller that manages distributed model workers.
    
    Features:
    - Worker registration and heartbeat monitoring
    - Load balancing via lottery or shortest queue
    - Streaming response forwarding
    """

    def __init__(self, dispatch_method: str):
        self.worker_info: Dict[str, WorkerInfo] = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        # Start heartbeat monitoring thread
        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, 
            args=(self,), 
            daemon=True
        )
        self.heart_beat_thread.start()

        logger.info(f"Controller initialized with dispatch method: {dispatch_method}")

    def register_worker(
        self, 
        worker_name: str, 
        check_heart_beat: bool,
        worker_status: Optional[dict]
    ) -> bool:
        """Register a worker with the controller."""
        if worker_name not in self.worker_info:
            logger.info(f"Registering new worker: {worker_name}")
        else:
            logger.info(f"Re-registering existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            logger.error(f"Failed to get status for worker: {worker_name}")
            return False

        self.worker_info[worker_name] = WorkerInfo(
            model_names=worker_status["model_names"],
            speed=worker_status.get("speed", 1),
            queue_length=worker_status.get("queue_length", 0),
            check_heart_beat=check_heart_beat,
            last_heart_beat=time.time(),
            worker_address=worker_name
        )

        logger.info(f"Worker registered: {worker_name}, models: {worker_status['model_names']}")
        return True

    def get_worker_status(self, worker_name: str) -> Optional[dict]:
        """Get status from a worker via HTTP."""
        try:
            r = requests.post(
                worker_name + "/worker_get_status", 
                timeout=5
            )
            if r.status_code == 200:
                return r.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get status from {worker_name}: {e}")
        return None

    def remove_worker(self, worker_name: str):
        """Remove a worker from the registry."""
        if worker_name in self.worker_info:
            del self.worker_info[worker_name]
            logger.info(f"Removed worker: {worker_name}")

    def refresh_all_workers(self):
        """Refresh status of all registered workers."""
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Removed stale worker: {w_name}")

    def list_models(self) -> List[str]:
        """List all available models across workers."""
        model_names = set()
        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)
        return sorted(list(model_names))

    def get_worker_address(self, model_name: str) -> str:
        """Get a worker address for the specified model."""
        if self.dispatch_method == DispatchMethod.LOTTERY:
            return self._lottery_dispatch(model_name)
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            return self._shortest_queue_dispatch(model_name)
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def _lottery_dispatch(self, model_name: str) -> str:
        """Dispatch using weighted random selection."""
        worker_names = []
        worker_speeds = []
        
        for w_name, w_info in self.worker_info.items():
            if model_name in w_info.model_names:
                worker_names.append(w_name)
                worker_speeds.append(w_info.speed)
        
        if not worker_names:
            return ""
        
        worker_speeds = np.array(worker_speeds, dtype=np.float32)
        norm = np.sum(worker_speeds)
        if norm < 1e-4:
            return ""
        
        worker_speeds = worker_speeds / norm
        idx = np.random.choice(len(worker_names), p=worker_speeds)
        return worker_names[idx]

    def _shortest_queue_dispatch(self, model_name: str) -> str:
        """Dispatch to worker with shortest queue."""
        worker_names = []
        worker_qlen = []
        
        for w_name, w_info in self.worker_info.items():
            if model_name in w_info.model_names:
                worker_names.append(w_name)
                effective_qlen = w_info.queue_length / max(w_info.speed, 1)
                worker_qlen.append(effective_qlen)
        
        if not worker_names:
            return ""
        
        min_idx = int(np.argmin(worker_qlen))
        w_name = worker_names[min_idx]
        self.worker_info[w_name].queue_length += 1
        
        logger.debug(f"Dispatched to {w_name} (queue: {worker_qlen[min_idx]:.2f})")
        return w_name

    def receive_heart_beat(self, worker_name: str, queue_length: int) -> bool:
        """Receive and process heartbeat from a worker."""
        if worker_name not in self.worker_info:
            logger.warning(f"Heartbeat from unknown worker: {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.debug(f"Heartbeat received: {worker_name}")
        return True

    def remove_stale_workers_by_expiration(self):
        """Remove workers that haven't sent heartbeat recently."""
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)
            logger.info(f"Removed expired worker: {worker_name}")

    def worker_api_generate_stream(self, params: dict):
        """Forward streaming generation request to a worker."""
        worker_addr = self.get_worker_address(params["model"])
        
        if not worker_addr:
            logger.warning(f"No worker available for model: {params['model']}")
            ret = {
                "text": SERVER_ERROR_MSG,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"
            return

        try:
            response = requests.post(
                worker_addr + "/worker_generate_stream",
                json=params, 
                stream=True, 
                timeout=120
            )
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            logger.error(f"Worker request failed: {worker_addr}, {e}")
            ret = {
                "text": SERVER_ERROR_MSG,
                "error_code": 3,
            }
            yield json.dumps(ret).encode() + b"\0"

    def get_aggregated_status(self) -> dict:
        """Get aggregated status from all workers."""
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)
            speed += w_info.speed
            queue_length += w_info.queue_length

        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
            "worker_count": len(self.worker_info)
        }


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI()
controller: Controller = None


@app.post("/register_worker")
async def register_worker(request: Request):
    """Register a new worker."""
    data = await request.json()
    success = controller.register_worker(
        data["worker_name"], 
        data.get("check_heart_beat", True),
        data.get("worker_status", None)
    )
    return {"success": success}


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    """Refresh all worker statuses."""
    controller.refresh_all_workers()
    return {"success": True}


@app.post("/list_models")
async def list_models():
    """List all available models."""
    models = controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    """Get a worker address for a model."""
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    """Receive heartbeat from a worker."""
    data = await request.json()
    exist = controller.receive_heart_beat(
        data["worker_name"], 
        data.get("queue_length", 0)
    )
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    """Forward streaming generation request."""
    params = await request.json()
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)


@app.post("/worker_get_status")
async def worker_api_get_status():
    """Get aggregated worker status."""
    return controller.get_aggregated_status()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "worker_count": len(controller.worker_info),
        "models": controller.list_models()
    }


def create_controller_app(dispatch_method: str = "shortest_queue") -> FastAPI:
    """Create and configure the controller app."""
    global controller
    controller = Controller(dispatch_method)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emomni Controller")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_CONTROLLER_PORT)
    parser.add_argument(
        "--dispatch-method", 
        type=str, 
        choices=["lottery", "shortest_queue"], 
        default="shortest_queue"
    )
    args = parser.parse_args()
    
    logger.info(f"Starting controller on {args.host}:{args.port}")
    logger.info(f"Dispatch method: {args.dispatch_method}")
    
    controller = Controller(args.dispatch_method)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
