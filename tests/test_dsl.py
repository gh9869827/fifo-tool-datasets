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
    {"system": "System prompt #1", "in": "in #1", "reasoning": "", "out": "out #1"}
]

EXPECTED_DSL_02_WIDE: list[dict[str, str]] = [
    {"system": "System prompt #1", "in": "in #1", "reasoning": "", "out": "out #1"},
    {"system": "System prompt #2", "in": "in #2", "reasoning": "", "out": "out #2"}
]

EXPECTED_DSL_03_WIDE: list[dict[str, str]] = [
    {
        "system": (
            "only one line. Note that after the $ there is a blank added for readability. "
            "It is optional when reading a file. But when generating a dat file, always add it."
        ),
        "in": "only one line. Same note as above about the blank.",
        "reasoning": "",
        "out": "only one line. Same note as above about the blank.",
    },
    {
        "system": "when multi lines\nare needed it is formatted\nlike that",
        "in": "only one line. Same note as above about the blank.",
        "reasoning": "",
        "out": "only one line. Same note as above about the blank.",
    },
    {
        "system": "still one line.",
        "in": "one line\nand\nanother here",
        "reasoning": "",
        "out": "but only one here is supported, i.e. one record with single vs multi lines.",
    },
    {
        "system": "line 1\nline 2",
        "in": "user input 1\nuser input 2",
        "reasoning": "",
        "out": "dsl output 1\ndsl output 2",
    },
]

EXPECTED_DSL_04_WIDE: list[dict[str, str]] = [
    {"system": "Sys #1", "in": "in #1", "reasoning": "", "out": "out #1"},
    {"system": "Sys #1", "in": "in #2", "reasoning": "", "out": "out #2"},
    {"system": "Sys multi\nline", "in": "in #3", "reasoning": "", "out": "out #3"},
    {"system": "Sys multi\nline", "in": "in #4", "reasoning": "", "out": "out #4"},
    {"system": "Sys multi\nline", "in": "in #5", "reasoning": "", "out": "out #5"},
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

EXPECTED_DSL_03_STRUCTURED = [
    {
        "messages": [
            {
                "role": "system",
                "content": (
                    "only one line. Note that after the $ there is a blank added for readability. "
                    "It is optional when reading a file. But when generating a dat file, always add it."
                ),
            },
            {"role": "user", "content": "only one line. Same note as above about the blank."},
            {"role": "assistant", "content": "only one line. Same note as above about the blank."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "when multi lines\nare needed it is formatted\nlike that"},
            {"role": "user", "content": "only one line. Same note as above about the blank."},
            {"role": "assistant", "content": "only one line. Same note as above about the blank."},
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "still one line."},
            {"role": "user", "content": "one line\nand\nanother here"},
            {
                "role": "assistant",
                "content": "but only one here is supported, i.e. one record with single vs multi lines.",
            },
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "line 1\nline 2"},
            {"role": "user", "content": "user input 1\nuser input 2"},
            {"role": "assistant", "content": "dsl output 1\ndsl output 2"},
        ]
    },
]

EXPECTED_DSL_04_STRUCTURED = [
    {"messages": [
        {"role": "system", "content": "Sys #1"},
        {"role": "user", "content": "in #1"},
        {"role": "assistant", "content": "out #1"},
    ]},
    {"messages": [
        {"role": "system", "content": "Sys #1"},
        {"role": "user", "content": "in #2"},
        {"role": "assistant", "content": "out #2"},
    ]},
    {"messages": [
        {"role": "system", "content": "Sys multi\nline"},
        {"role": "user", "content": "in #3"},
        {"role": "assistant", "content": "out #3"},
    ]},
    {"messages": [
        {"role": "system", "content": "Sys multi\nline"},
        {"role": "user", "content": "in #4"},
        {"role": "assistant", "content": "out #4"},
    ]},
    {"messages": [
        {"role": "system", "content": "Sys multi\nline"},
        {"role": "user", "content": "in #5"},
        {"role": "assistant", "content": "out #5"},
    ]},
]


@pytest.mark.parametrize("filename,expected", [
    ("dsl_01.dat", EXPECTED_DSL_01_WIDE),
    ("dsl_02.dat", EXPECTED_DSL_02_WIDE),
    ("dsl_03.dat", EXPECTED_DSL_03_WIDE),
    ("dsl_04.dat", EXPECTED_DSL_04_WIDE)
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
    ("dsl_broken_02.dat", r"DSL sample is not closed properly, last line 8"),
    ("dsl_broken_03.dat", r"Expected closing '---' but got '\$System prompt #2' at line 5."),
    ("dsl_broken_04.dat", r"Expected '<' at start of line 5."),
    ("dsl_broken_05.dat", r"Expected '>' but got '<' at line 4."),
    ("dsl_broken_06.dat", r"Expected '\$' but got '>' at line 2."),
    ("dsl_broken_07.dat", r"Expected '\$' at start of line 2."),
    ("dsl_broken_08.dat", r"Expected '>' but got '<' at line 7."),
    ("dsl_broken_09.dat", r"Expected closing '---' but got '\$ System prompt #2' at line 5."),
    ("dsl_broken_10.dat", r"Expected '<' at start of line 4."),
    ("dsl_broken_11.dat", r"The file is empty."),
    ("dsl_broken_12.dat", r"Empty tag '>' detected at line 3."),
    ("dsl_broken_13.dat", r"Empty tag '<' detected at line 6."),
    ("dsl_broken_14.dat", r"System prompt placeholder '\.\.\.' without preceding system at line 2."),
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
    "dsl_03.dat",
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


def test_wide_to_dat_collapses_system_prompt() -> None:
    adapter = DSLAdapter()
    dataset = Dataset.from_dict(  # type: ignore[reportUnknownMemberType]
        {
            "system": ["Repeat", "Repeat", "Other"],
            "in": ["q1", "q2", "q3"],
            "out": ["a1", "a2", "a3"],
        }
    )

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".dat") as tmp:
        tmp_path = pathlib.Path(tmp.name)
        adapter.from_wide_dataset_to_dat(dataset, str(tmp_path))

    with open(tmp_path, "r", encoding="utf-8") as f:
        content = normalize_dat(f.read())

    assert content.splitlines() == [
        "---",
        "$ Repeat",
        "> q1",
        "< a1",
        "---",
        "$ ...",
        "> q2",
        "< a2",
        "---",
        "$ Other",
        "> q3",
        "< a3",
        "---",
    ]


@pytest.mark.parametrize("filename,expected", [
    ("dsl_01.dat", EXPECTED_DSL_01_STRUCTURED),
    ("dsl_02.dat", EXPECTED_DSL_02_STRUCTURED),
    ("dsl_03.dat", EXPECTED_DSL_03_STRUCTURED),
    ("dsl_04.dat", EXPECTED_DSL_04_STRUCTURED),
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
        (EXPECTED_DSL_03_STRUCTURED, EXPECTED_DSL_03_WIDE),
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

    assert wide_dataset.column_names == ["system", "in", "reasoning", "out"]
    assert len(wide_dataset) == len(expected_wide)
    for i, expected in enumerate(expected_wide):
        # Build expected dict with correct field order
        expected_with_reasoning = {
            "system": expected["system"],
            "in": expected["in"],
            "reasoning": "",
            "out": expected["out"]
        }
        assert wide_dataset[i] == expected_with_reasoning


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


def test_sort_dat_file(tmp_path: pathlib.Path) -> None:
    adapter = DSLAdapter()
    dat_path = tmp_path / "unsorted.dat"
    dat_path.write_text(
        """---
$ b
> q2
< a2
---
$ a
> q1
< a1
---
$ b
> q1
< a0
---
""",
        encoding="utf-8",
    )

    adapter.sort_dat_file(str(dat_path))

    sorted_ds = adapter.from_dat_to_wide_dataset(str(dat_path))
    # Pylance: Type of to_list() is partially unknown
    assert sorted_ds.to_list() == [  # type: ignore[reportUnknownMemberType]
        {"system": "a", "in": "q1", "reasoning": "", "out": "a1"},
        {"system": "b", "in": "q1", "reasoning": "", "out": "a0"},
        {"system": "b", "in": "q2", "reasoning": "", "out": "a2"},
    ]


# Tests for reasoning support

def test_from_dat_to_wide_dataset_with_reasoning() -> None:
    """Test parsing .dat file with reasoning section."""
    adapter = DSLAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "dsl_reasoning_01.dat"
    dataset = adapter.from_dat_to_wide_dataset(str(path))

    assert isinstance(dataset, Dataset)
    # Pylance: Type of to_list() is partially unknown
    result = dataset.to_list()  # type: ignore[reportUnknownMemberType]
    assert len(result) == 1
    assert result[0] == {
        "system": "You are a precise DSL parser.",
        "in": "today at 5:30PM",
        "reasoning": "base=TODAY\ntime.hour=17\ntime.minute=30",
        "out": "SET_TIME(TODAY, 17, 30)"
    }


def test_from_dat_to_wide_dataset_mixed_reasoning() -> None:
    """Test parsing .dat file with mixed reasoning (some records with, some without)."""
    adapter = DSLAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "dsl_reasoning_02.dat"
    dataset = adapter.from_dat_to_wide_dataset(str(path))

    assert isinstance(dataset, Dataset)
    # Pylance: Type of to_list() is partially unknown
    result = dataset.to_list()  # type: ignore[reportUnknownMemberType]
    assert len(result) == 2
    assert result[0] == {
        "system": "You are a precise DSL parser.",
        "in": "today at 5:30PM",
        "reasoning": "base=TODAY\ntime.hour=17\ntime.minute=30",
        "out": "SET_TIME(TODAY, 17, 30)"
    }
    assert result[1] == {
        "system": "You are a precise DSL parser.",
        "in": "set alarm for 8am",
        "reasoning": "",
        "out": "SET_ALARM(TODAY, 8, 0)"
    }


def test_from_dat_to_wide_dataset_single_line_reasoning() -> None:
    """Test parsing .dat file with single-line reasoning."""
    adapter = DSLAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "dsl_reasoning_03.dat"
    dataset = adapter.from_dat_to_wide_dataset(str(path))

    assert isinstance(dataset, Dataset)
    # Pylance: Type of to_list() is partially unknown
    result = dataset.to_list()  # type: ignore[reportUnknownMemberType]
    assert len(result) == 1
    assert result[0] == {
        "system": "You are a precise DSL parser.",
        "in": "set alarm for 8am",
        "reasoning": "Single line reasoning",
        "out": "SET_ALARM(TODAY, 8, 0)"
    }


def test_from_wide_dataset_to_json_with_reasoning() -> None:
    """Test conversion to JSON format with reasoning."""
    adapter = DSLAdapter()
    dataset = Dataset.from_dict({  # type: ignore[reportUnknownMemberType]
        "system": ["You are a precise DSL parser."],
        "in": ["today at 5:30PM"],
        "reasoning": ["base=TODAY\ntime.hour=17\ntime.minute=30"],
        "out": ["SET_TIME(TODAY, 17, 30)"]
    })

    json_records = adapter.from_wide_dataset_to_json(dataset)

    assert isinstance(json_records, list)
    assert len(json_records) == 1
    assert json_records[0] == {
        "messages": [
            {"role": "system", "content": "You are a precise DSL parser."},
            {"role": "user", "content": "today at 5:30PM"},
            {
                "role": "assistant",
                "content": "SET_TIME(TODAY, 17, 30)",
                "metadata": {
                    "reasoning": "base=TODAY\ntime.hour=17\ntime.minute=30"
                }
            }
        ]
    }


def test_from_wide_dataset_to_json_without_reasoning() -> None:
    """Test conversion to JSON format without reasoning (backward compatibility)."""
    adapter = DSLAdapter()
    dataset = Dataset.from_dict({  # type: ignore[reportUnknownMemberType]
        "system": ["You are a precise DSL parser."],
        "in": ["set alarm for 8am"],
        "reasoning": [""],
        "out": ["SET_ALARM(TODAY, 8, 0)"]
    })

    json_records = adapter.from_wide_dataset_to_json(dataset)

    assert isinstance(json_records, list)
    assert len(json_records) == 1
    assert json_records[0] == {
        "messages": [
            {"role": "system", "content": "You are a precise DSL parser."},
            {"role": "user", "content": "set alarm for 8am"},
            {"role": "assistant", "content": "SET_ALARM(TODAY, 8, 0)"}
        ]
    }


def test_from_wide_dataset_to_dat_with_reasoning() -> None:
    """Test writing .dat file with reasoning."""
    adapter = DSLAdapter()
    dataset = Dataset.from_dict({  # type: ignore[reportUnknownMemberType]
        "system": ["You are a precise DSL parser."],
        "in": ["today at 5:30PM"],
        "reasoning": ["base=TODAY\ntime.hour=17\ntime.minute=30"],
        "out": ["SET_TIME(TODAY, 17, 30)"]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir) / "test.dat"
        adapter.from_wide_dataset_to_dat(dataset, str(tmp_path))
        content = tmp_path.read_text(encoding="utf-8")

    expected = """---
$ You are a precise DSL parser.
> today at 5:30PM
?
base=TODAY
time.hour=17
time.minute=30
< SET_TIME(TODAY, 17, 30)
---
"""
    assert content == expected


def test_from_wide_dataset_to_dat_without_reasoning() -> None:
    """Test writing .dat file without reasoning (backward compatibility)."""
    adapter = DSLAdapter()
    dataset = Dataset.from_dict({  # type: ignore[reportUnknownMemberType]
        "system": ["You are a precise DSL parser."],
        "in": ["set alarm for 8am"],
        "reasoning": [""],
        "out": ["SET_ALARM(TODAY, 8, 0)"]
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir) / "test.dat"
        adapter.from_wide_dataset_to_dat(dataset, str(tmp_path))
        content = tmp_path.read_text(encoding="utf-8")

    expected = """---
$ You are a precise DSL parser.
> set alarm for 8am
< SET_ALARM(TODAY, 8, 0)
---
"""
    assert content == expected


def test_roundtrip_with_reasoning() -> None:
    """Test roundtrip conversion: .dat -> wide -> .dat with reasoning."""
    adapter = DSLAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "dsl_reasoning_01.dat"

    # Load .dat → wide
    wide_dataset = adapter.from_dat_to_wide_dataset(str(path))

    # Write wide → .dat
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir) / "test.dat"
        adapter.from_wide_dataset_to_dat(wide_dataset, str(tmp_path))

        # Load again
        wide_dataset2 = adapter.from_dat_to_wide_dataset(str(tmp_path))

    # Compare
    # Pylance: Type of to_list() is partially unknown
    assert wide_dataset.to_list() == wide_dataset2.to_list()  # type: ignore[reportUnknownMemberType]


def test_from_dataset_to_wide_dataset_with_reasoning() -> None:
    """Test conversion from structured format with reasoning to wide format."""
    adapter = DSLAdapter()

    structured_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a precise DSL parser."},
                {"role": "user", "content": "today at 5:30PM"},
                {
                    "role": "assistant",
                    "content": "SET_TIME(TODAY, 17, 30)",
                    "metadata": {
                        "reasoning": "base=TODAY\ntime.hour=17\ntime.minute=30"
                    }
                }
            ]
        }
    ]

    # Pylance: Type of from_list() is partially unknown
    structured_dataset = Dataset.from_list(  # type: ignore[reportUnknownMemberType]
        cast(list[dict[str, Any]], structured_data)
    )

    wide_dataset = adapter.from_dataset_to_wide_dataset(structured_dataset)

    assert wide_dataset.column_names == ["system", "in", "reasoning", "out"]
    # Pylance: Type of to_list() is partially unknown
    result = wide_dataset.to_list()  # type: ignore[reportUnknownMemberType]
    assert len(result) == 1
    assert result[0] == {
        "system": "You are a precise DSL parser.",
        "in": "today at 5:30PM",
        "reasoning": "base=TODAY\ntime.hour=17\ntime.minute=30",
        "out": "SET_TIME(TODAY, 17, 30)"
    }


def test_multiline_with_content_on_same_line() -> None:
    """Test parsing multi-line sections where content starts on the same line as the tag."""
    adapter = DSLAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "dsl_multiline_sameline.dat"
    
    dataset = adapter.from_dat_to_wide_dataset(str(path))
    
    # Pylance: Type of to_list() is partially unknown
    result = dataset.to_list()  # type: ignore[reportUnknownMemberType]
    assert len(result) == 1
    assert result[0] == {
        "system": "You are a precise DSL parser.\nLine 2 of system prompt",
        "in": "today at 5:30PM\nAdditional context",
        "reasoning": "base=TODAY\ntime.hour=17\ntime.minute=30",
        "out": "SET_TIME(TODAY, 17, 30)"
    }
