import tempfile
from typing import Any, cast
from unittest.mock import patch
import pathlib
import pytest
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
    DatasetDict
)
from fifo_tool_datasets.sdk.hf_dataset_adapters.common import (
    StructuredConversationRecord
)
from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import (
    DSLAdapter
)

EXPECTED_DSL_01_WIDE: list[dict[str, str]] = [
    {"system": "System prompt #1", "in": "in #1", "out": "out #1"}
]

EXPECTED_DSL_02_WIDE: list[dict[str, str]] = [
    {"system": "System prompt #1", "in": "in #1", "out": "out #1"},
    {"system": "System prompt #2", "in": "in #2", "out": "out #2"}
]

EXPECTED_DSL_01_STRUCTURED = [
    {"messages": [
        {"role": "system",     "content": "System prompt #1"},
        {"role": "user",       "content": "in #1"},
        {"role": "assistant",  "content": "out #1"}
    ]}
]

EXPECTED_DSL_02_STRUCTURED = [
    {"messages": [
        {"role": "system",     "content": "System prompt #1"},
        {"role": "user",       "content": "in #1"},
        {"role": "assistant",  "content": "out #1"}
    ]},
    {"messages": [
        {"role": "system",     "content": "System prompt #2"},
        {"role": "user",       "content": "in #2"},
        {"role": "assistant",  "content": "out #2"}
    ]}
]


@pytest.mark.parametrize("filename,expected", [
    ("dsl_01.dat", EXPECTED_DSL_01_WIDE),
    ("dsl_02.dat", EXPECTED_DSL_02_WIDE)
])
def test_from_dat_to_wide_dataset(filename: str, expected: list[dict[str, str]]):
    adapter = DSLAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / filename
    dataset = adapter.from_dat_to_wide_dataset(str(path))

    assert isinstance(dataset, Dataset)

    # Pylance: Type of to_list() is partially unknown
    assert dataset.to_list() == expected   # type: ignore[reportUnknownMemberType]


@pytest.mark.parametrize("filename,expected_error", [
    ("dsl_broken_01.dat", r"The file must start with '---'."),
    ("dsl_broken_02.dat", r"Each DSL sample must span 4 lines \(including ---\)."),
    ("dsl_broken_03.dat", r"Missing '---' block delimiter at line 5."),
    ("dsl_broken_04.dat", r"Expected '<' at start of output line in block at line 4."),
    ("dsl_broken_05.dat", r"Expected '>' at start of input line in block at line 3."),
    ("dsl_broken_06.dat", r"Expected '\$' at start of system line in block at line 2."),
    ("dsl_broken_07.dat", r"Expected '\$' at start of system line in block at line 2."),
])
def test_from_dat_to_wide_dataset_broken(filename: str, expected_error: str) -> None:
    adapter = DSLAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / filename

    with pytest.raises(SyntaxError, match=expected_error):
        adapter.from_dat_to_wide_dataset(str(path))

def normalize_dat(content: str) -> str:
    # Normalize line endings and strip trailing whitespace
    return "\n".join(line.strip() for line in content.strip().splitlines())


@pytest.mark.parametrize("filename", [
    "dsl_01.dat",
    "dsl_02.dat",
])
def test_roundtrip_wide_to_dat(filename: str) -> None:
    adapter = DSLAdapter()
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
    ("dsl_01.dat", EXPECTED_DSL_01_STRUCTURED),
    ("dsl_02.dat", EXPECTED_DSL_02_STRUCTURED),
])
def test_from_wide_dataset_to_json_dsl(
    filename: str,
    expected: list[StructuredConversationRecord]
) -> None:
    adapter = DSLAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / filename

    wide = adapter.from_dat_to_wide_dataset(str(path))
    json_records = adapter.from_wide_dataset_to_json(wide)

    assert isinstance(json_records, list)
    assert json_records == expected
    assert all("role" in m and "content" in m for r in json_records for m in r["messages"])


@pytest.mark.parametrize(
    "structured, expected_wide",
    [
        (EXPECTED_DSL_01_STRUCTURED, EXPECTED_DSL_01_WIDE),
        (EXPECTED_DSL_02_STRUCTURED, EXPECTED_DSL_02_WIDE),
    ]
)
def test_from_dataset_to_wide_dataset_dsl(
    structured: list[StructuredConversationRecord],
    expected_wide: list[dict[str, str]]
) -> None:
    adapter = DSLAdapter()

    # Pylance: Type of from_list() is partially unknown
    structured_dataset = Dataset.from_list(  # type: ignore[reportUnknownMemberType]
        cast(list[dict[str, Any]], structured)
    )

    wide_dataset = adapter.from_dataset_to_wide_dataset(structured_dataset)

    assert wide_dataset.column_names == ["system", "in", "out"]
    assert len(wide_dataset) == len(expected_wide)
    for i, expected in enumerate(expected_wide):
        assert wide_dataset[i] == expected


def test_from_hub_to_dataset_wide_dict_success() -> None:
    adapter = DSLAdapter()

    # Simulate a valid split DatasetDict with DSL-style fields
    dummy_split = Dataset.from_dict(  # type: ignore[reportUnknownMemberType]
        {
            "system": ["Sys prompt"],
            "in": ["Input text"],
            "out": ["Output DSL"]
        }
    )

    fake_dataset_dict = DatasetDict({
        "train": dummy_split,
        "validation": dummy_split,
        "test": dummy_split
    })

    with patch(
        "fifo_tool_datasets.sdk.hf_dataset_adapters.dsl.load_dataset",
        return_value=fake_dataset_dict
    ):
        result = adapter.from_hub_to_dataset_wide_dict("username/dataset")
        assert isinstance(result, DatasetDict)
        assert set(cast(list[str], result.keys())) == {"train", "validation", "test"}
