from typing import List, Any, Text

import pytest
from ruth.core.actions.loops import LoopAction
from ruth.core.channels import CollectingOutputChannel
from ruth.shared.core.domain import Domain
from ruth.shared.core.events import (
    Event,
    ActionExecutionRejected,
    ActionExecuted,
    ActiveLoop,
    SlotSet,
)
from ruth.core.nlg import TemplatedNaturalLanguageGenerator
from ruth.shared.core.trackers import DialogueStateTracker


async def test_whole_loop():
    expected_activation_events = [
        ActionExecutionRejected("tada"),
        ActionExecuted("test"),
    ]

    expected_do_events = [ActionExecuted("do")]
    expected_deactivation_events = [SlotSet("deactivated")]

    form_name = "my form"

    class MyLoop(LoopAction):
        def name(self) -> Text:
            return form_name

        async def activate(self, *args: Any) -> List[Event]:
            return expected_activation_events

        async def do(self, *args: Any) -> List[Event]:
            events_so_far = args[-1]
            assert events_so_far == [ActiveLoop(form_name), *expected_activation_events]

            return expected_do_events

        async def deactivate(self, *args) -> List[Event]:
            events_so_far = args[-1]
            assert events_so_far == [
                ActiveLoop(form_name),
                *expected_activation_events,
                *expected_do_events,
                ActiveLoop(None),
            ]

            return expected_deactivation_events

        async def is_done(self, *args) -> bool:
            events_so_far = args[-1]
            return events_so_far == [
                ActiveLoop(form_name),
                *expected_activation_events,
                *expected_do_events,
            ]

    tracker = DialogueStateTracker.from_events("some sender", [])
    domain = Domain.empty()

    action = MyLoop()
    actual = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert actual == [
        ActiveLoop(form_name),
        *expected_activation_events,
        *expected_do_events,
        ActiveLoop(None),
        *expected_deactivation_events,
    ]


async def test_loop_without_deactivate():
    expected_activation_events = [
        ActionExecutionRejected("tada"),
        ActionExecuted("test"),
    ]

    expected_do_events = [ActionExecuted("do")]
    form_name = "my form"

    class MyLoop(LoopAction):
        def name(self) -> Text:
            return form_name

        async def activate(self, *args: Any) -> List[Event]:
            return expected_activation_events

        async def do(self, *args: Any) -> List[Event]:
            return expected_do_events

        async def deactivate(self, *args) -> List[Event]:
            raise ValueError("this shouldn't be called")

        async def is_done(self, *args) -> bool:
            return False

    tracker = DialogueStateTracker.from_events("some sender", [])
    domain = Domain.empty()

    action = MyLoop()
    actual = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert actual == [
        ActiveLoop(form_name),
        *expected_activation_events,
        *expected_do_events,
    ]


async def test_loop_without_activate_and_without_deactivate():
    expected_do_events = [ActionExecuted("do")]
    form_name = "my form"

    class MyLoop(LoopAction):
        def name(self) -> Text:
            return form_name

        async def activate(self, *args: Any) -> List[Event]:
            raise ValueError("this shouldn't be called")

        async def do(self, *args: Any) -> List[Event]:
            return expected_do_events

        async def deactivate(self, *args) -> List[Event]:
            return [SlotSet("deactivated")]

        async def is_activated(self, *args: Any) -> bool:
            return True

        async def is_done(self, *args) -> bool:
            return False

    tracker = DialogueStateTracker.from_events("some sender", [])
    domain = Domain.empty()

    action = MyLoop()
    actual = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert actual == [*expected_do_events]


async def test_raise_not_implemented_error():
    loop = LoopAction()
    with pytest.raises(NotImplementedError):
        await loop.do(None, None, None, None, [])

    with pytest.raises(NotImplementedError):
        await loop.is_done(None, None, None, None, [])
