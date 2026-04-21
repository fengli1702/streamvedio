import ray
import threading

@ray.remote # Runs on CPU by default
class SharedWeightsActor:
    """
    Stores and manages LORA weights for sharing between actors.
    """
    def __init__(self):
        """
        Initializes the actor.
        - self.weights: Stores the actual LoRA weights (e.g., state_dict) or a pointer/path.
        - self.update_available: Flag indicating if new weights are available.
        - self._lock: Threading lock for safe concurrent access to shared state.
        """
        # Stores LoRA weights ("weights": state_dict) and training step ("step": training_step)
        self.weights: dict = None
        # Flag indicating if new weights are available since the last get_weights call.
        self.update_available = False
        # Lock for thread-safe access to weights and flag
        self._lock = threading.Lock()
        print("SharedWeightsActor initialized.")

    def get_weights(self):
        """
        Returns the latest weights if an update is available, otherwise None.
        Sets update_available to False upon retrieval.
        """
        with self._lock:
            if self.update_available:
                self.update_available = False # Reset flag after retrieval
                return self.weights
            else:
                return None

    def set_weights(self, new_weights: tuple):
        """
        Updates the stored weights and sets the update flag to True.
        """
        with self._lock:
            self.weights = new_weights
            self.update_available = True

    def check_update_status(self):
        """Checks if an update is available without changing the flag."""
        with self._lock:
            return self.update_available