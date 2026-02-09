from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any
import logging

class Step(ABC):
    def __init__(self, name: str, config: Any):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"export.steps.{name}")

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step.
        :param context: Dictionary containing artifacts and state from previous steps.
        :return: Updated context.
        """
        pass

    def log_start(self):
        self.logger.info(f"--- Starting Step: {self.name} ---")

    def log_end(self):
        self.logger.info(f"--- Finished Step: {self.name} ---")
