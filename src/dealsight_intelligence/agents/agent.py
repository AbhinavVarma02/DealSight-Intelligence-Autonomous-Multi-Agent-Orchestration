"""Base class for every agent in the system.

Provides a coloured logging helper so the run log is easy to follow when
multiple agents are working at the same time.
"""

from __future__ import annotations

import logging


class Agent:
    """Tiny base class — gives each agent a name, a colour, and a log() method."""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    name = "Agent"
    color = WHITE

    def log(self, message: str) -> None:
        logging.info("%s[%s] %s%s", self.color, self.name, message, self.RESET)
