"""Messaging agent.

Formats a deal alert and either prints it (dry-run) or posts it to Pushover.
Push is opt-in via DEALSIGHT_INTELLIGENCE_DO_PUSH so nothing leaks during local
development.
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request

from dealsight_intelligence import config
from dealsight_intelligence.agents.agent import Agent
from dealsight_intelligence.agents.deals import Opportunity


class MessagingAgent(Agent):
    name = "Messaging Agent"
    color = Agent.RED


    def __init__(self, do_push: bool | None = None) -> None:
        self.do_push = config.bool_env("DEALSIGHT_INTELLIGENCE_DO_PUSH", False)
        if do_push is not None:
            self.do_push = do_push

    def alert(self, opportunity: Opportunity) -> str:
        # Build a single human-readable line describing the deal and either
        # send it via Pushover or just log it locally.
        message = (
            f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
            f"Estimate=${opportunity.estimate:.2f}, "
            f"Discount=${opportunity.discount:.2f}: "
            f"{opportunity.deal.product_description} {opportunity.deal.url}"
        )
        if self.do_push:
            self._send_pushover(message)
            self.log("sent Pushover alert")
        else:
            self.log(f"dry-run alert: {message}")
        return message

    def _send_pushover(self, message: str) -> None:
        user = os.getenv("PUSHOVER_USER")
        token = os.getenv("PUSHOVER_TOKEN")
        if not user or not token:
            raise RuntimeError("PUSHOVER_USER and PUSHOVER_TOKEN are required when push is enabled")
        payload = urllib.parse.urlencode({"user": user, "token": token, "message": message}).encode()
        request = urllib.request.Request("https://api.pushover.net/1/messages.json", data=payload)
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status >= 400:
                raise RuntimeError(f"Pushover failed with status {response.status}")
