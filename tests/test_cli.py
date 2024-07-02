import os
from pathlib import Path

from test_data import example_dataset


def test_cli_parser():
    result = os.system("trackastra")
    assert result == 0


def test_cli_tracking_from_folder():
    example_dataset()
    cmd = "trackastra track -i test_data/img -m test_data/TRA --output-ctc test_data/tracked --output-edge-table test_data/tracked.csv --model-pretrained general_2d --device cpu"  # noqa: RUF100
    print(cmd)
    result = os.system(cmd)
    assert Path("test_data/tracked").exists()
    assert Path("test_data/tracked.csv").exists()
    assert result == 0


def test_cli_tracking_from_file():
    root = Path(__file__).parent.parent / "trackastra" / "data" / "resources"
    print(root)
    assert root.exists()
    output_ctc = Path("test_data") / "tracked_bacteria"
    output_edge_table = Path("test_data") / "tracked_bacteria.csv"

    cmd = f"trackastra track -i {root / 'trpL_150310-11_img.tif'} -m {root / 'trpL_150310-11_mask.tif'} --output-ctc {output_ctc} --output-edge-table {output_edge_table} --model-pretrained general_2d --device cpu"  # noqa: RUF100
    print(cmd)
    result = os.system(cmd)

    assert output_ctc.exists()
    assert output_edge_table.exists()
    assert result == 0
