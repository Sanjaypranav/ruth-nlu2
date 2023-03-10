import json

import jsonpickle
import pytest

import ruth.shared.utils.io
import ruth.utils.io
from ruth.shared.core.conversation import Dialogue
from ruth.shared.core.domain import Domain
from ruth.core.tracker_store import InMemoryTrackerStore
from tests.conftest import (
    TEST_DIALOGUES,
    EXAMPLE_DOMAINS,
)
from tests.core.utilities import tracker_from_dialogue_file


@pytest.mark.parametrize("filename", TEST_DIALOGUES)
def test_dialogue_serialisation(filename, domain: Domain):
    dialogue_json = ruth.shared.utils.io.read_file(filename)
    restored = json.loads(dialogue_json)
    tracker = tracker_from_dialogue_file(filename, domain)
    en_de_coded = json.loads(jsonpickle.encode(tracker.as_dialogue()))
    assert restored == en_de_coded


@pytest.mark.parametrize("pair", zip(TEST_DIALOGUES, EXAMPLE_DOMAINS))
def test_inmemory_tracker_store(pair):
    filename, domainpath = pair
    domain = Domain.load(domainpath)
    tracker = tracker_from_dialogue_file(filename, domain)
    tracker_store = InMemoryTrackerStore(domain)
    tracker_store.save(tracker)
    restored = tracker_store.retrieve(tracker.sender_id)
    assert restored == tracker


def test_tracker_default(domain: Domain):
    filename = "data/test_dialogues/default.json"
    tracker = tracker_from_dialogue_file(filename, domain)
    assert tracker.get_slot("name") == "Peter"
    assert tracker.get_slot("price") is None  # slot doesn't exist!


def test_dialogue_from_parameters(domain: Domain):
    filename = "data/test_dialogues/default.json"
    tracker = tracker_from_dialogue_file(filename, domain)
    serialised_dialogue = InMemoryTrackerStore.serialise_tracker(tracker)
    deserialised_dialogue = Dialogue.from_parameters(json.loads(serialised_dialogue))
    assert tracker.as_dialogue().as_dict() == deserialised_dialogue.as_dict()
