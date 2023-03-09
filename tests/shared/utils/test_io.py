import os
import string
import textwrap
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Text, List, Set, Any, Dict

import pytest

import ruth.shared
from ruth.shared.exceptions import FileIOException, FileNotFoundException, RasaException
import ruth.shared.utils.io
import ruth.shared.utils.validation
from ruth.shared.constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS
from ruth.utils import io as io_utils

os.environ["USER_NAME"] = "user"
os.environ["PASS"] = "pass"


def test_raise_user_warning():
    with pytest.warns(UserWarning) as record:
        ruth.shared.utils.io.raise_warning("My warning.")

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."


def test_raise_future_warning():
    with pytest.warns(FutureWarning) as record:
        ruth.shared.utils.io.raise_warning("My future warning.", FutureWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My future warning."


def test_raise_deprecation():
    with pytest.warns(DeprecationWarning) as record:
        ruth.shared.utils.io.raise_warning("My warning.", DeprecationWarning)

    assert len(record) == 1
    assert record[0].message.args[0] == "My warning."
    assert isinstance(record[0].message, DeprecationWarning)


def test_read_file_with_not_existing_path():
    with pytest.raises(FileNotFoundException):
        ruth.shared.utils.io.read_file("some path")


@pytest.mark.parametrize(
    "list_function, expected",
    [
        (
                ruth.shared.utils.io.list_directory,
                {"subdirectory", "subdirectory/sub_file.txt", "file.txt"},
        ),
        (ruth.shared.utils.io.list_files, {"subdirectory/sub_file.txt", "file.txt"}),
        (ruth.shared.utils.io.list_subdirectories, {"subdirectory"}),
    ],
)
def test_list_directory(
    tmpdir: Path, list_function: Callable[[Text], List[Text]], expected: Set[Text]
):
    subdirectory = tmpdir / "subdirectory"
    subdirectory.mkdir()

    sub_sub_directory = subdirectory / "subdirectory"
    sub_sub_directory.mkdir()

    sub_sub_file = sub_sub_directory / "sub_file.txt"
    sub_sub_file.write_text("", encoding=ruth.shared.utils.io.DEFAULT_ENCODING)

    file1 = subdirectory / "file.txt"
    file1.write_text("", encoding="utf-8")

    hidden_directory = subdirectory / ".hidden"
    hidden_directory.mkdir()

    hidden_file = subdirectory / ".test.text"
    hidden_file.write_text("", encoding="utf-8")

    expected = {str(subdirectory / entry) for entry in expected}

    assert set(list_function(str(subdirectory))) == expected


def test_read_yaml_string():
    config_without_env_var = """
    user: user
    password: pass
    """
    content = ruth.shared.utils.io.read_yaml(config_without_env_var)
    assert content["user"] == "user" and content["password"] == "pass"


def test_read_yaml_string_with_env_var():
    config_with_env_var = """
    user: ${USER_NAME}
    password: ${PASS}
    """
    content = ruth.shared.utils.io.read_yaml(config_with_env_var)
    assert content["user"] == "user" and content["password"] == "pass"


def test_read_yaml_string_with_multiple_env_vars_per_line():
    config_with_env_var = """
    user: ${USER_NAME} ${PASS}
    password: ${PASS}
    """
    content = ruth.shared.utils.io.read_yaml(config_with_env_var)
    assert content["user"] == "user pass" and content["password"] == "pass"


def test_read_yaml_string_with_env_var_prefix():
    config_with_env_var_prefix = """
    user: db_${USER_NAME}
    password: db_${PASS}
    """
    content = ruth.shared.utils.io.read_yaml(config_with_env_var_prefix)
    assert content["user"] == "db_user" and content["password"] == "db_pass"


def test_read_yaml_string_with_env_var_postfix():
    config_with_env_var_postfix = """
    user: ${USER_NAME}_admin
    password: ${PASS}_admin
    """
    content = ruth.shared.utils.io.read_yaml(config_with_env_var_postfix)
    assert content["user"] == "user_admin" and content["password"] == "pass_admin"


def test_read_yaml_string_with_env_var_infix():
    config_with_env_var_infix = """
    user: db_${USER_NAME}_admin
    password: db_${PASS}_admin
    """
    content = ruth.shared.utils.io.read_yaml(config_with_env_var_infix)
    assert content["user"] == "db_user_admin" and content["password"] == "db_pass_admin"


def test_read_yaml_string_with_env_var_not_exist():
    config_with_env_var_not_exist = """
    user: ${USER_NAME}
    password: ${PASSWORD}
    """
    with pytest.raises(RasaException):
        ruth.shared.utils.io.read_yaml(config_with_env_var_not_exist)


def test_environment_variable_not_existing():
    content = "model: \n  test: ${variable}"
    with pytest.raises(RasaException):
        ruth.shared.utils.io.read_yaml(content)


def test_environment_variable_dict_without_prefix_and_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: ${variable}"

    content = ruth.shared.utils.io.read_yaml(content)

    assert content["model"]["test"] == "test"


def test_environment_variable_in_list():
    os.environ["variable"] = "test"
    content = "model: \n  - value\n  - ${variable}"

    content = ruth.shared.utils.io.read_yaml(content)

    assert content["model"][1] == "test"


def test_environment_variable_dict_with_prefix():
    os.environ["variable"] = "test"
    content = "model: \n  test: dir/${variable}"

    content = ruth.shared.utils.io.read_yaml(content)

    assert content["model"]["test"] == "dir/test"


def test_environment_variable_dict_with_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: ${variable}/dir"

    content = ruth.shared.utils.io.read_yaml(content)

    assert content["model"]["test"] == "test/dir"


def test_environment_variable_dict_with_prefix_and_with_postfix():
    os.environ["variable"] = "test"
    content = "model: \n  test: dir/${variable}/dir"

    content = ruth.shared.utils.io.read_yaml(content)

    assert content["model"]["test"] == "dir/test/dir"


def test_environment_variable_with_dollar_char():
    os.environ["variable1"] = "$test1"
    os.environ["variable2"] = "test2"
    content = "model: \n  test1: ${variable1}\n  test2: ${variable2}"

    content = ruth.shared.utils.io.read_yaml(content)

    assert content["model"]["test1"] == "$test1"
    assert content["model"]["test2"] == "test2"


def test_environment_variable_with_dollar_char_in_the_middle():
    os.environ["variable1"] = "test$123"
    content = "model: \n  test1: ${variable1}"

    content = ruth.shared.utils.io.read_yaml(content)

    assert content["model"]["test1"] == "test$123"


def test_emojis_in_yaml():
    test_data = """
    data:
        - one 😁💯 👩🏿‍💻👨🏿‍💻
        - two £ (?u)\\b\\w+\\b f\u00fcr
    """
    content = ruth.shared.utils.io.read_yaml(test_data)

    assert content["data"][0] == "one 😁💯 👩🏿‍💻👨🏿‍💻"
    assert content["data"][1] == "two £ (?u)\\b\\w+\\b für"


def test_emojis_in_tmp_file():
    test_data = """
        data:
            - one 😁💯 👩🏿‍💻👨🏿‍💻
            - two £ (?u)\\b\\w+\\b f\u00fcr
        """
    test_file = io_utils.create_temporary_file(test_data)
    content = ruth.shared.utils.io.read_yaml_file(test_file)

    assert content["data"][0] == "one 😁💯 👩🏿‍💻👨🏿‍💻"
    assert content["data"][1] == "two £ (?u)\\b\\w+\\b für"


def test_read_emojis_from_json():
    import json

    d = {"text": "hey 😁💯 👩🏿‍💻👨🏿‍💻🧜‍♂️(?u)\\b\\w+\\b} f\u00fcr"}
    json_string = json.dumps(d, indent=2)

    content = ruth.shared.utils.io.read_yaml(json_string)

    expected = "hey 😁💯 👩🏿‍💻👨🏿‍💻🧜‍♂️(?u)\\b\\w+\\b} für"
    assert content.get("text") == expected


def test_bool_str():
    test_data = """
    one: "yes"
    two: "true"
    three: "True"
    """

    content = ruth.shared.utils.io.read_yaml(test_data)

    assert content["one"] == "yes"
    assert content["two"] == "true"
    assert content["three"] == "True"


@pytest.mark.parametrize(
    "should_preserve_key_order, expected_keys",
    [(True, list(reversed(string.ascii_lowercase)))],
)
def test_dump_yaml_key_order(
    tmp_path: Path, should_preserve_key_order: bool, expected_keys: List[Text]
):
    file = tmp_path / "test.yml"

    # create YAML file with keys in reverse-alphabetical order
    content = ""
    for i in reversed(string.ascii_lowercase):
        content += f"{i}: {uuid.uuid4().hex}\n"

    file.write_text(content)

    # load this file and ensure keys are in correct reverse-alphabetical order
    data = ruth.shared.utils.io.read_yaml_file(file)
    assert list(data.keys()) == list(reversed(string.ascii_lowercase))

    # dumping `data` will result in alphabetical or reverse-alphabetical list of keys,
    # depending on the value of `should_preserve_key_order`
    ruth.shared.utils.io.write_yaml(
        data, file, should_preserve_key_order=should_preserve_key_order
    )
    with file.open() as f:
        keys = [line.split(":")[0] for line in f.readlines()]

    assert keys == expected_keys


@pytest.mark.parametrize(
    "source, target",
    [
        # ordinary dictionary
        ({"b": "b", "a": "a"}, OrderedDict({"b": "b", "a": "a"})),
        # dict with list
        ({"b": [1, 2, 3]}, OrderedDict({"b": [1, 2, 3]})),
        # nested dict
        ({"b": {"c": "d"}}, OrderedDict({"b": OrderedDict({"c": "d"})})),
        # doubly-nested dict
        (
            {"b": {"c": {"d": "e"}}},
            OrderedDict({"b": OrderedDict({"c": OrderedDict({"d": "e"})})}),
        ),
        # a list is not affected
        ([1, 2, 3], [1, 2, 3]),
        # a string also isn't
        ("hello", "hello"),
    ],
)
def test_convert_to_ordered_dict(source: Any, target: Any):
    assert ruth.shared.utils.io.convert_to_ordered_dict(source) == target

    def _recursively_check_dict_is_ordered_dict(d):
        if isinstance(d, dict):
            assert isinstance(d, OrderedDict)
            for value in d.values():
                _recursively_check_dict_is_ordered_dict(value)

    # ensure nested dicts are converted correctly
    _recursively_check_dict_is_ordered_dict(target)


def test_create_directory_for_file(tmp_path: Path):
    file = str(tmp_path / "dir" / "test.txt")

    ruth.shared.utils.io.create_directory_for_file(str(file))
    assert not os.path.exists(file)
    assert os.path.exists(os.path.dirname(file))


def test_write_json_file(tmp_path: Path):
    expected = {"abc": "dasds", "list": [1, 2, 3, 4], "nested": {"a": "b"}}
    file_path = str(tmp_path / "abc.txt")

    ruth.shared.utils.io.dump_obj_as_json_to_file(file_path, expected)
    assert ruth.shared.utils.io.read_json_file(file_path) == expected


def test_create_directory_if_new(tmp_path: Path):
    directory = str(tmp_path / "a" / "b")
    ruth.shared.utils.io.create_directory(directory)

    assert os.path.exists(directory)


def test_create_directory_if_already_exists(tmp_path: Path):
    # This should not throw an exception
    ruth.shared.utils.io.create_directory(str(tmp_path))
    assert True


def test_raise_deprecation_warning():
    with pytest.warns(FutureWarning) as record:
        ruth.shared.utils.io.raise_deprecation_warning(
            "This feature is deprecated.", warn_until_version="3.0.0"
        )

    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "This feature is deprecated. (will be removed in 3.0.0)"
    )


def test_raise_deprecation_warning_version_already_in_message():
    with pytest.warns(FutureWarning) as record:
        ruth.shared.utils.io.raise_deprecation_warning(
            "This feature is deprecated and will be removed in 3.0.0!",
            warn_until_version="3.0.0",
        )

    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "This feature is deprecated and will be removed in 3.0.0!"
    )


def test_raise_deprecation_warning_default():
    with pytest.warns(FutureWarning) as record:
        ruth.shared.utils.io.raise_deprecation_warning("This feature is deprecated.")

    assert len(record) == 1
    assert record[0].message.args[0] == (
        f"This feature is deprecated. "
        f"(will be removed in {NEXT_MAJOR_VERSION_FOR_DEPRECATIONS})"
    )


def test_read_file_with_wrong_encoding(tmp_path: Path):
    file = tmp_path / "myfile.txt"
    file.write_text("ä", encoding="latin-1")
    with pytest.raises(FileIOException):
        ruth.shared.utils.io.read_file(file)


@pytest.mark.parametrize("config_file", Path("data", "configs_for_docs").glob("*.yml"))
def test_validate_config_file(config_file: Path):
    # does not raise
    ruth.shared.utils.io.read_model_configuration(config_file)


def test_validate_config_file_with_extra_keys(tmp_path: Path):
    content = textwrap.dedent(
        """
        language: en
        pipeline:
        policies:

        importers:
        - RasaFileImporter
        """
    )
    config_file = tmp_path / "config.yml"
    config_file.write_text(content)

    ruth.shared.utils.io.read_model_configuration(config_file)


@pytest.mark.parametrize(
    "config",
    [
        # Pre 2.x pipeline templates are invalid
        textwrap.dedent(
            """
            pipeline: supervised_embeddings
            """
        ),
        # Each list item needs the `name` property
        textwrap.dedent(
            """
            pipeline:
            - DIETClassier
            policies:
            """
        ),
        # Name property is missing
        textwrap.dedent(
            """
            pipeline:
            policies:
            - some_attribute: "lala"
            """
        ),
        # Name property is not a string
        textwrap.dedent(
            """
            pipeline:
            policies:
            - name: 1234
            """
        ),
        # Invalid training data version
        textwrap.dedent(
            """
            version: 2.0
            policies:
            pipeline:
            """
        ),
        # Language has wrong type
        textwrap.dedent(
            """
            language: []
            policies:
            pipeline:
            """
        ),
    ],
)
def test_invalid_config_files(config: Text, tmp_path: Path):
    config_file = tmp_path / "config.yml"
    config_file.write_text(config)
    with pytest.raises(ruth.shared.utils.validation.YamlValidationException):
        ruth.shared.utils.io.read_model_configuration(config_file)


@pytest.mark.parametrize(
    "content, expected",
    [
        ("rest:", {"rest": None}),
        (
            textwrap.dedent(
                """
                tracker_store:
                    password: test
                """
            ),
            {"tracker_store": {"password": "test"}},
        ),
    ],
)
def test_read_config_file(tmp_path: Path, content: Text, expected: Dict):
    config_file = tmp_path / "file.yml"
    config_file.write_text(content)

    assert ruth.shared.utils.io.read_config_file(config_file) == expected


@pytest.mark.parametrize(
    "content",
    [
        "text",
        textwrap.dedent(
            """
            - item1
            - item2
            """
        ),
    ],
)
def test_read_invalid_config_file(tmp_path: Path, content: Text):
    config_file = tmp_path / "file.yml"
    config_file.write_text(content)

    with pytest.raises(ruth.shared.utils.validation.YamlValidationException):
        ruth.shared.utils.io.read_model_configuration(config_file)


@pytest.mark.parametrize(
    "file,keys,expected_result",
    [
        ("data/test_yaml_stories/stories.yml", ["stories"], True),
        ("data/test_yaml_stories/stories.yml", ["something_else"], False),
        ("data/test_yaml_stories/stories.yml", ["stories", "something_else"], True),
        (
            "data/test_domains/default_retrieval_intents.yml",
            ["intents", "responses"],
            True,
        ),
        ("data/test_yaml_stories/rules_without_stories.yml", ["rules"], True),
        ("data/test_yaml_stories/rules_without_stories.yml", ["stories"], False),
        ("data/test_stories/stories.md", ["something"], False),
    ],
)
async def test_is_key_in_yaml(file: Text, keys: List[Text], expected_result: bool):
    assert ruth.shared.utils.io.is_key_in_yaml(file, *keys) == expected_result


async def test_is_key_in_yaml_with_unicode_files():
    # This shouldn't raise
    assert ruth.shared.utils.io.is_key_in_yaml(
        "./data/test_nlu_no_responses/nlu_with_unicode.yml", "nlu"
    )