import ruth.core
from ruth.shared.core.constants import ACTION_SESSION_START_NAME
from ruth.shared.core.domain import Domain
from ruth.shared.core.events import (
    SessionStarted,
    SlotSet,
    UserUttered,
    ActionExecuted,
    DefinePrevUserUtteredFeaturization,
)
from ruth.shared.core.trackers import DialogueStateTracker
from ruth.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from ruth.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
from ruth.shared.core.training_data.structures import Story
from ruth.shared.nlu.constants import INTENT_NAME_KEY

domain = Domain.load("data/test_moodbot/domain.yml")


def test_session_start_is_not_serialised(domain: Domain):
    tracker = DialogueStateTracker("default", domain.slots)
    # the retrieved tracker should be empty
    assert len(tracker.events) == 0

    # add SlotSet event
    tracker.update(SlotSet("slot", "value"))

    # add the two SessionStarted events and a user event
    tracker.update(ActionExecuted(ACTION_SESSION_START_NAME))
    tracker.update(SessionStarted())
    tracker.update(
        UserUttered("say something", intent={INTENT_NAME_KEY: "some_intent"})
    )
    tracker.update(DefinePrevUserUtteredFeaturization(False))

    YAMLStoryWriter().dumps(
        Story.from_events(tracker.events, "some-story01").story_steps
    )

    expected = """version: "2.0"
stories:
- story: some-story01
  steps:
  - slot_was_set:
    - slot: value
  - intent: some_intent
"""

    actual = YAMLStoryWriter().dumps(
        Story.from_events(tracker.events, "some-story01").story_steps
    )
    assert actual == expected


def test_as_story_string_or_statement():
    import ruth.shared.utils.io

    stories = """
    stories:
    - story: hello world
      steps:
      - or:
        - intent: intent1
        - intent: intent2
        - intent: intent3
      - action: some_action
    """

    reader = YAMLStoryReader(is_used_for_training=False)
    yaml_content = ruth.shared.utils.io.read_yaml(stories)

    steps = reader.read_from_parsed_yaml(yaml_content)

    assert len(steps) == 1

    assert (
        steps[0].as_story_string()
        == """
## hello world
* intent1 OR intent2 OR intent3
    - some_action
"""
    )


def test_cap_length():
    assert (
            ruth.shared.core.training_data.structures._cap_length("mystring", 6) == "mys..."
    )


def test_cap_length_without_ellipsis():
    assert (
            ruth.shared.core.training_data.structures._cap_length(
            "mystring", 3, append_ellipsis=False
        )
            == "mys"
    )


def test_cap_length_with_short_string():
    assert ruth.shared.core.training_data.structures._cap_length("my", 3) == "my"
