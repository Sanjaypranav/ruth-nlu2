import json
import logging
from unittest.mock import patch

import ruth
from ruth.core.channels import TelegramInput
from ruth.core.channels.telegram import TelegramOutput
from ruth.core.agent import Agent

logger = logging.getLogger(__name__)


def noop(*args, **kwargs):
    """Just do nothing."""
    pass


def mock_get_me(self):
    self.username = "YOUR_TELEGRAM_BOT"
    return self


@patch.object(TelegramOutput, "set_webhook", noop)
@patch.object(TelegramOutput, "get_me", mock_get_me)
def test_telegram_edit_message():
    telegram_test_edited_message = {
        "update_id": 280069275,
        "edited_message": {
            "message_id": 591,
            "from": {
                "id": 1760450482,
                "is_bot": "False",
                "first_name": "Martin",
                "last_name": "Man",
                "language_code": "en",
            },
            "chat": {
                "id": 1760450482,
                "first_name": "Martin",
                "last_name": "Man",
                "type": "private",
            },
            "date": 1621577771,
            "edit_date": 1621580124,
            "text": "Hello!",
        },
    }

    input_channel = TelegramInput(
        # you get this when setting up a bot
        access_token="123:YOUR_ACCESS_TOKEN",
        # this is your bots username
        verify="YOUR_TELEGRAM_BOT",
        # the url your bot should listen for messages
        webhook_url="YOUR_WEBHOOK_URL",
    )

    app = ruth.core.run.configure_app([input_channel], port=5004)
    app.agent = Agent()
    _, res = app.test_client.post(
        "/webhooks/telegram/webhook", json=json.dumps(telegram_test_edited_message)
    )

    assert res.status_code == 200
