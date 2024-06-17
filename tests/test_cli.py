from shell import shell

from test_data import example_dataset


def test_cli_parser():
    result = shell("trackastra")
    assert result.code == 0


def test_cli_tracking():
    example_dataset()
    result = shell(
        "trackastra track -i test_data/img -m test_data/TRA --model-pretrained general_2d"  # noqa: RUF100
    )
    assert result.code == 0
