import argparse
import logging
import uuid

from typing import List

from ruth import telemetry
from ruth.cli import SubParsersAction
from ruth.cli.arguments import shell as arguments
from ruth.shared.utils.cli import print_error
from ruth.exceptions import ModelNotFound

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all shell parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    shell_parser = subparsers.add_parser(
        "shell",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=(
            "Loads your trained model and lets you talk to your "
            "assistant on the command line."
        ),
    )
    shell_parser.set_defaults(func=shell)

    shell_parser.add_argument(
        "--conversation-id",
        default=uuid.uuid4().hex,
        required=False,
        help="Set the conversation ID.",
    )

    run_subparsers = shell_parser.add_subparsers()

    shell_nlu_subparser = run_subparsers.add_parser(
        "nlu",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Interprets messages on the command line using your NLU model.",
    )

    shell_nlu_subparser.set_defaults(func=shell_nlu)

    arguments.set_shell_arguments(shell_parser)
    arguments.set_shell_nlu_arguments(shell_nlu_subparser)


def shell_nlu(args: argparse.Namespace) -> None:
    from ruth.cli.utils import get_validated_path
    from ruth.shared.constants import DEFAULT_MODELS_PATH
    from ruth.model import get_model, get_model_subdirectories
    import ruth.nlu.run

    args.connector = "cmdline"

    model = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

    try:
        model_path = get_model(model)
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `ruth train nlu`."
        )
        return

    _, nlu_model = get_model_subdirectories(model_path)

    if not nlu_model:
        print_error(
            "No NLU model found. Train a model before running the "
            "server using `ruth train nlu`."
        )
        return

    telemetry.track_shell_started("nlu")
    ruth.nlu.run.run_cmdline(nlu_model)


def shell(args: argparse.Namespace) -> None:
    from ruth.cli.utils import get_validated_path
    from ruth.shared.constants import DEFAULT_MODELS_PATH
    from ruth.model import get_model, get_model_subdirectories

    args.connector = "cmdline"

    model = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

    try:
        model_path = get_model(model)
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `ruth train`."
        )
        return

    core_model, nlu_model = get_model_subdirectories(model_path)

    if not core_model:
        import ruth.nlu.run

        telemetry.track_shell_started("nlu")

        ruth.nlu.run.run_cmdline(nlu_model)
    else:
        import ruth.cli.run

        telemetry.track_shell_started("ruth")

        ruth.cli.run.run(args)
