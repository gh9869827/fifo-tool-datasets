import tempfile
from typing import cast
import pytest
from unittest.mock import patch
from fifo_tool_datasets.sdk.hf_dataset_adapters.common import StructuredConversationRecord
from fifo_tool_datasets.sdk.hf_dataset_adapters.sqna import (
    SQNAAdapter
)
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
    DatasetDict
)
import pathlib

def test_from_dat_to_wide_dataset():
    adapter = SQNAAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "sqna_01.dat"
    dataset = adapter.from_dat_to_wide_dataset(str(path))

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 2
    assert dataset[0]["in"] == "question 1"
    assert dataset[0]["out"] == "answer 1"
    assert dataset[1]["in"] == "question 2"
    assert dataset[1]["out"] == "answer 2"

@pytest.mark.parametrize("filename,expected_error", [
    ("sqna_broken_01.dat", "Invalid syntax \\(missing answer\\)"),
    ("sqna_broken_02.dat", "Invalid syntax \\(missing question\\)"),
    ("sqna_broken_03.dat", "Invalid syntax \\(missing question\\)"),
    ("sqna_broken_04.dat", "Invalid syntax \\(missing answer\\)"),
    ("sqna_broken_05.dat", "Invalid syntax \\(line does not start with > or <\\)"),
])
def test_from_dat_to_wide_dataset_broken(filename: str, expected_error: str) -> None:
    adapter = SQNAAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / filename

    with pytest.raises(SyntaxError, match=expected_error):
        adapter.from_dat_to_wide_dataset(str(path))

def normalize_dat(content: str) -> str:
    # Normalize line endings and strip trailing whitespace
    return "\n".join(line.strip() for line in content.strip().splitlines())

@pytest.mark.parametrize("filename", [
    "sqna_01.dat",
])
def test_roundtrip_wide_to_dat(filename: str) -> None:
    adapter = SQNAAdapter()
    fixtures_dir = pathlib.Path(__file__).parent / "fixtures"
    original_path = fixtures_dir / filename

    # Load .dat → wide
    wide_dataset = adapter.from_dat_to_wide_dataset(str(original_path))

    # Write wide → .dat
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".dat") as tmp:
        tmp_path = pathlib.Path(tmp.name)
        adapter.from_wide_dataset_to_dat(wide_dataset, str(tmp_path))

    # Compare normalized content
    with open(original_path, "r", encoding="utf-8") as f1, open(tmp_path, "r", encoding="utf-8") as f2:
        original = normalize_dat(f1.read())
        roundtrip = normalize_dat(f2.read())

    assert roundtrip == original

@pytest.mark.parametrize("filename,expected", [
    (
        "sqna_01.dat",
        [
            {"messages": [
                {"role": "user", "content": "question 1"},
                {"role": "assistant", "content": "answer 1"}
            ]},
            {"messages": [
                {"role": "user", "content": "question 2"},
                {"role": "assistant", "content": "answer 2"}
            ]}
        ]
    )
])
def test_from_wide_dataset_to_json(
    filename: str,
    expected: list[StructuredConversationRecord]
) -> None:
    adapter = SQNAAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / filename

    wide = adapter.from_dat_to_wide_dataset(str(path))
    json_records = adapter.from_wide_dataset_to_json(wide)

    assert isinstance(json_records, list)
    assert json_records == expected
    assert all("role" in m and "content" in m for r in json_records for m in r["messages"])

def test_from_dataset_to_wide_dataset_sqna() -> None:
    adapter = SQNAAdapter()

    structured_data = [
        {"messages": [
            {"role": "user", "content": "question 1"},
            {"role": "assistant", "content": "answer 1"}
        ]},
        {"messages": [
            {"role": "user", "content": "question 2"},
            {"role": "assistant", "content": "answer 2"}
        ]}
    ]
    # Pylance: Type of from_list() is partially unknown
    structured_dataset = Dataset.from_list(  # type: ignore[reportUnknownMemberType]
        structured_data
    )
    wide_dataset = adapter.from_dataset_to_wide_dataset(structured_dataset)

    assert wide_dataset.column_names == ["in", "out"]
    assert len(wide_dataset) == 2
    assert wide_dataset[0]["in"] == "question 1"
    assert wide_dataset[0]["out"] == "answer 1"
    assert wide_dataset[1]["in"] == "question 2"
    assert wide_dataset[1]["out"] == "answer 2"

def test_from_hub_to_dataset_wide_dict_success():
    adapter = SQNAAdapter()

    # Simulate a valid split DatasetDict
    # Pylance: Type of from_dict() is partially unknown
    dummy_split = Dataset.from_dict(  # type: ignore[reportUnknownMemberType]
        {"in": ["question?"], "out": ["answer."]}
    )

    fake_dataset_dict = DatasetDict({
        "train": dummy_split,
        "validation": dummy_split,
        "test": dummy_split
    })

    with patch(
        "fifo_tool_datasets.sdk.hf_dataset_adapters.sqna.load_dataset",
        return_value=fake_dataset_dict
    ):
        result = adapter.from_hub_to_dataset_wide_dict("username/dataset")
        assert isinstance(result, DatasetDict)
        assert set(cast(list[str], result.keys())) == {"train", "validation", "test"}
