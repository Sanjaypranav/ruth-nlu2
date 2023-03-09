import pytest

import ruth.shared.utils.cli


def test_print_error_and_exit():
    with pytest.raises(SystemExit):
        ruth.shared.utils.cli.print_error_and_exit("")
