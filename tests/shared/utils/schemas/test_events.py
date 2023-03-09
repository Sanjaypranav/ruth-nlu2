from typing import Type

import pytest
import jsonschema

from ruth.core.actions.action import RemoteAction
from ruth.shared.core.events import Event
import ruth.shared.utils.common


@pytest.mark.parametrize("event_class", ruth.shared.utils.common.all_subclasses(Event))
def test_remote_action_validate_all_event_subclasses(event_class: Type[Event]):

    if event_class.type_name == "slot":
        response = {
            "events": [{"event": "slot", "name": "test", "value": "example"}],
            "responses": [],
        }
    elif event_class.type_name == "entities":
        response = {"events": [{"event": "entities", "entities": []}], "responses": []}
    else:
        response = {"events": [{"event": event_class.type_name}], "responses": []}

    # ignore the below events since these are not sent or received outside Rasa
    if event_class.type_name not in [
        "wrong_utterance",
        "wrong_action",
        "warning_predicted",
    ]:
        jsonschema.validate(response, RemoteAction.action_response_format_spec())
