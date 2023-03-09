import argparse

from ruth.cli.arguments.default_arguments import add_model_param
from ruth.cli.arguments.run import add_server_arguments


def set_shell_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_param(parser)
    add_server_arguments(parser)


def set_shell_nlu_arguments(parser: argparse.ArgumentParser) -> None:
    add_model_param(parser)
