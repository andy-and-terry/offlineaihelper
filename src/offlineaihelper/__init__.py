__version__ = "0.1.0"

# Sync (task-routing) public API — usable without starting the FastAPI server
from offlineaihelper.sync_app import ResponseEnvelope, SyncOfflineAIHelper

__all__ = ["ResponseEnvelope", "SyncOfflineAIHelper"]
