import argparse
import uuid

import ruth.cli.arguments.default_arguments
import ruth.cli.arguments.train
import ruth.cli.arguments.run


def set_interactive_arguments(parser: argparse.ArgumentParser) -> None:
    """Specifies arguments for `ruth interactive`."""
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Save story files in e2e format. In this format user messages "
        "will be included in the stories.",
    )
    ruth.cli.arguments.run.add_port_argument(parser)

    ruth.cli.arguments.default_arguments.add_model_param(parser, default=None)
    ruth.cli.arguments.train.add_data_param(parser)

    _add_common_params(parser)
    train_arguments = _add_training_arguments(parser)

    ruth.cli.arguments.train.add_force_param(train_arguments)
    ruth.cli.arguments.train.add_persist_nlu_data_param(train_arguments)


def set_interactive_core_arguments(parser: argparse.ArgumentParser) -> None:
    """Specifies arguments for `ruth interactive core`."""
    ruth.cli.arguments.default_arguments.add_model_param(
        parser, model_name="Rasa Core", default=None
    )
    ruth.cli.arguments.default_arguments.add_stories_param(parser)

    _add_common_params(parser)
    _add_training_arguments(parser)
    ruth.cli.arguments.run.add_port_argument(parser)


def _add_common_params(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-visualization",
        default=False,
        action="store_true",
        help="Disable plotting the visualization during interactive learning.",
    )

    parser.add_argument(
        "--conversation-id",
        default=uuid.uuid4().hex,
        help="Specify the id of the conversation the messages are in. Defaults to a "
        "UUID that will be randomly generated.",
    )

    ruth.cli.arguments.default_arguments.add_endpoint_param(
        parser,
        help_text="Configuration file for the model server "
        "and the connectors as a yml file.",
    )


# noinspection PyProtectedMember
def _add_training_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    train_arguments = parser.add_argument_group("Train Arguments")
    ruth.cli.arguments.train.add_config_param(train_arguments)
    ruth.cli.arguments.default_arguments.add_domain_param(train_arguments)
    ruth.cli.arguments.train.add_out_param(
        train_arguments, help_text="Directory where your models should be stored."
    )
    ruth.cli.arguments.train.add_augmentation_param(train_arguments)
    ruth.cli.arguments.train.add_debug_plots_param(train_arguments)
    ruth.cli.arguments.train.add_finetune_params(train_arguments)
    return train_arguments
