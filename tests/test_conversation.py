import tempfile
from typing import Any, cast
from unittest.mock import MagicMock, patch
import pytest
from fifo_tool_datasets.sdk.hf_dataset_adapters.common import (
    StructuredConversationRecord
)
from fifo_tool_datasets.sdk.hf_dataset_adapters.conversation import (
    ConversationAdapter,
    WideConversationRecord
)
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
    DatasetDict
)
import pathlib


EXPECTED_CONVERSATION_01_WIDE: list[dict[str, Any]] = [
    {"id_conversation": 0, "id_message": 0, "role": "system",     "content": "You are a helpful assistant."},
    {"id_conversation": 0, "id_message": 1, "role": "directives", "content": "Always be concise."},
    {"id_conversation": 0, "id_message": 2, "role": "user",       "content": "Hello!"},
    {"id_conversation": 0, "id_message": 3, "role": "assistant",  "content": "Hi there. How can I help?"}
]

EXPECTED_CONVERSATION_02_WIDE: list[dict[str, Any]] = [
    {"id_conversation": 0, "id_message": 0, "role": "system",     "content": "You are a helpful assistant."},
    {"id_conversation": 0, "id_message": 1, "role": "directives", "content": "Always be concise."},
    {"id_conversation": 0, "id_message": 2, "role": "user",       "content": "Hello!"},
    {"id_conversation": 0, "id_message": 3, "role": "assistant",  "content": "Hi there. How can I help?"},
    {"id_conversation": 1, "id_message": 0, "role": "system",     "content": "You are a helpful assistant. #2"},
    {"id_conversation": 1, "id_message": 1, "role": "directives", "content": "Always be concise. #2"},
    {"id_conversation": 1, "id_message": 2, "role": "user",       "content": "Hello! #2"},
    {"id_conversation": 1, "id_message": 3, "role": "assistant",  "content": "Hi there. How can I help? #2"},
]

EXPECTED_CONVERSATION_03_WIDE: list[dict[str, Any]] = [
    {"id_conversation": 0, "id_message": 0, "role": "system",     "content": "You are a helpful assistant."},
    {"id_conversation": 0, "id_message": 1, "role": "directives", "content": "Always be concise."},
    {"id_conversation": 0, "id_message": 2, "role": "user",       "content": "Hello!"},
    {"id_conversation": 0, "id_message": 3, "role": "assistant",  "content": "Hi there. How can I help?"},
    {"id_conversation": 1, "id_message": 0, "role": "system",     "content": "You are a helpful assistant. #2"},
    {"id_conversation": 1, "id_message": 1, "role": "directives", "content": "Always be concise. #2"},
    {"id_conversation": 1, "id_message": 2, "role": "user",       "content": "Hello! #2"},
    {"id_conversation": 1, "id_message": 3, "role": "assistant",  "content": "Hi there. How can I help? #2"},
    {"id_conversation": 2, "id_message": 0, "role": "system",     "content": "You are a helpful assistant. #3"},
    {"id_conversation": 2, "id_message": 1, "role": "user",       "content": "Hello! #3"},
    {"id_conversation": 2, "id_message": 2, "role": "assistant",  "content": "Hi there. How can I help? #3"},
]

EXPECTED_CONVERSATION_04_WIDE: list[dict[str, Any]] = [
    {"id_conversation": 0, "id_message": 0, "role": "system",     "content": "You are a helpful assistant."},
    {"id_conversation": 0, "id_message": 1, "role": "directives", "content": "Always be concise."},
    {"id_conversation": 0, "id_message": 2, "role": "user",       "content": "Hello!"},
    {"id_conversation": 0, "id_message": 3, "role": "assistant",  "content": "Hi there. How can I help?"},
    {"id_conversation": 1, "id_message": 0, "role": "system",     "content": "You are a helpful assistant. #2"},
    {"id_conversation": 1, "id_message": 1, "role": "directives", "content": "Always be concise. #2"},
    {"id_conversation": 1, "id_message": 2, "role": "user",       "content": "Hello! #2"},
    {"id_conversation": 1, "id_message": 3, "role": "assistant",  "content": "Hi there. How can I help? #2"},
    {"id_conversation": 2, "id_message": 0, "role": "user",       "content": "Hello! #3"},
    {"id_conversation": 2, "id_message": 1, "role": "assistant",  "content": "Hi there. How can I help? #3"},
]

EXPECTED_CONVERSATION_01_STRUCTURED = [
    {"messages": [
        {"role": "system",     "content": "You are a helpful assistant."},
        {"role": "directives", "content": "Always be concise."},
        {"role": "user",       "content": "Hello!"},
        {"role": "assistant",  "content": "Hi there. How can I help?"}
    ]}
]

EXPECTED_CONVERSATION_02_STRUCTURED = [
    {"messages": [
        {"role": "system",     "content": "You are a helpful assistant."},
        {"role": "directives", "content": "Always be concise."},
        {"role": "user",       "content": "Hello!"},
        {"role": "assistant",  "content": "Hi there. How can I help?"}
    ]},
    {"messages": [
        {"role": "system",     "content": "You are a helpful assistant. #2"},
        {"role": "directives", "content": "Always be concise. #2"},
        {"role": "user",       "content": "Hello! #2"},
        {"role": "assistant",  "content": "Hi there. How can I help? #2"}
    ]}
]

EXPECTED_CONVERSATION_03_STRUCTURED = [
    {"messages": [
        {"role": "system",     "content": "You are a helpful assistant."},
        {"role": "directives", "content": "Always be concise."},
        {"role": "user",       "content": "Hello!"},
        {"role": "assistant",  "content": "Hi there. How can I help?"}
    ]},
    {"messages": [
        {"role": "system",     "content": "You are a helpful assistant. #2"},
        {"role": "directives", "content": "Always be concise. #2"},
        {"role": "user",       "content": "Hello! #2"},
        {"role": "assistant",  "content": "Hi there. How can I help? #2"}
    ]},
    {"messages": [
        {"role": "system",     "content": "You are a helpful assistant. #3"},
        {"role": "user",       "content": "Hello! #3"},
        {"role": "assistant",  "content": "Hi there. How can I help? #3"}
    ]}
]

EXPECTED_CONVERSATION_04_STRUCTURED = [
    {"messages": [
        {"role": "system",     "content": "You are a helpful assistant."},
        {"role": "directives", "content": "Always be concise."},
        {"role": "user",       "content": "Hello!"},
        {"role": "assistant",  "content": "Hi there. How can I help?"}
    ]},
    {"messages": [
        {"role": "system",     "content": "You are a helpful assistant. #2"},
        {"role": "directives", "content": "Always be concise. #2"},
        {"role": "user",       "content": "Hello! #2"},
        {"role": "assistant",  "content": "Hi there. How can I help? #2"}
    ]},
    {"messages": [
        {"role": "user",       "content": "Hello! #3"},
        {"role": "assistant",  "content": "Hi there. How can I help? #3"}
    ]}
]

@pytest.mark.parametrize("filename,expected", [
    ("conversation_01.dat", EXPECTED_CONVERSATION_01_WIDE),
    ("conversation_02.dat", EXPECTED_CONVERSATION_02_WIDE),
    ("conversation_03.dat", EXPECTED_CONVERSATION_03_WIDE),
    ("conversation_04.dat", EXPECTED_CONVERSATION_04_WIDE),
])
def test_from_dat_to_wide_dataset(filename: str, expected: list[WideConversationRecord]) -> None:
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / filename
    dataset = adapter.from_dat_to_wide_dataset(str(path))

    assert isinstance(dataset, Dataset)

    # Pylance: Type of to_list() is partially unknown
    assert dataset.to_list() == expected  # type: ignore[reportUnknownMemberType]

@pytest.mark.parametrize("filename,expected_error", [
    ("conversation_broken_01.dat", r"Conversation 0 is empty"),
    ("conversation_broken_02.dat", r"Two consecutive '>' tags detected at conversation 0, message 2"),
    ("conversation_broken_03.dat", r"Two consecutive '<' tags detected at conversation 0, message 3"),
    ("conversation_broken_04.dat", r"Conversation 2 is not closed properly"),
    ("conversation_broken_05.dat", r"Empty tag '>' detected at conversation 1, message 2"),
    ("conversation_broken_06.dat", r"Empty tag '\$' detected at conversation 1, message 0"),
    ("conversation_broken_07.dat", r"Error: \$ ! --- > or < followed by spaces, line 6"),
    ("conversation_broken_08.dat", r"Error: \$ ! --- > or < followed by spaces, line 2"),
])
def test_from_dat_to_wide_dataset_broken(filename: str, expected_error: str) -> None:
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / filename

    with pytest.raises(ValueError, match=expected_error):
        adapter.from_dat_to_wide_dataset(str(path))

def normalize_dat(content: str) -> str:
    # Normalize line endings and strip trailing whitespace
    return "\n".join(line.strip() for line in content.strip().splitlines())

@pytest.mark.parametrize("filename", [
    "conversation_01.dat",
    "conversation_02.dat",
    "conversation_03.dat",
    "conversation_04.dat",
])
def test_roundtrip_wide_to_dat(filename: str) -> None:
    adapter = ConversationAdapter()
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

# Parametrized test using the constants above
@pytest.mark.parametrize("filename,expected", [
    ("conversation_01.dat", EXPECTED_CONVERSATION_01_STRUCTURED),
    ("conversation_02.dat", EXPECTED_CONVERSATION_02_STRUCTURED),
    ("conversation_03.dat", EXPECTED_CONVERSATION_03_STRUCTURED),
    ("conversation_04.dat", EXPECTED_CONVERSATION_04_STRUCTURED),
])
def test_from_wide_dataset_to_json(
    filename: str,
    expected: list[StructuredConversationRecord]
) -> None:
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / filename

    wide = adapter.from_dat_to_wide_dataset(str(path))
    json_records = adapter.from_wide_dataset_to_json(wide)

    assert isinstance(json_records, list)
    assert json_records == expected

@pytest.mark.parametrize("structured_data,expected_wide", [
    (EXPECTED_CONVERSATION_01_STRUCTURED, EXPECTED_CONVERSATION_01_WIDE),
    (EXPECTED_CONVERSATION_02_STRUCTURED, EXPECTED_CONVERSATION_02_WIDE),
    (EXPECTED_CONVERSATION_03_STRUCTURED, EXPECTED_CONVERSATION_03_WIDE),
    (EXPECTED_CONVERSATION_04_STRUCTURED, EXPECTED_CONVERSATION_04_WIDE),
])
def test_from_dataset_to_wide_dataset_conversation(
    structured_data: list[StructuredConversationRecord],
    expected_wide: list[WideConversationRecord]):

    adapter = ConversationAdapter()
    # Pylance: Type of from_list() is partially unknown
    structured_dataset = Dataset.from_list(  # type: ignore[reportUnknownMemberType]
        cast(list[dict[str, Any]], structured_data)
    )
    wide_dataset = adapter.from_dataset_to_wide_dataset(structured_dataset)

    # Pylance: Type of to_list() is partially unknown
    actual = wide_dataset.to_list()  # type: ignore[reportUnknownMemberType]
    assert actual == expected_wide

@patch("datasets.DatasetDict.push_to_hub")
def test_from_dat_to_hub_mock_push(mock_push: MagicMock) -> None:
    adapter = ConversationAdapter()
    fixture_file = pathlib.Path(__file__).parent / "fixtures" / "conversation_04.dat"

    adapter.from_dat_to_hub(
        dat_filename=str(fixture_file),
        hub_dataset="dummy-user/dummy-dataset",
        commit_message="test commit message",
        split_ratios=(0.34, 0.33, 0.33)
    )

    # Assert push_to_hub was called once
    assert mock_push.called
    mock_push.assert_called_once_with(
        'dummy-user/dummy-dataset', commit_message='test commit message'
    )

@pytest.mark.parametrize(
    "directory,commit_message,should_fail,expected_exception",
    [
        ("data_dir_01", "test commit message", False, None),  # Valid directory
        ("data_dir_01", None, True, ValueError),  # Valid directory but missing commit message
        ("data_dir_broken_01", "test commit message", True, ValueError),
        ("data_dir_broken_02", "test commit message", True, ValueError),
    ]
)
@patch("datasets.DatasetDict.push_to_hub")
def test_from_dir_to_hub_parametrized(
    mock_push: MagicMock,
    directory: str,
    commit_message: str,
    should_fail: bool,
    expected_exception: type[Exception] | None,
) -> None:
    adapter = ConversationAdapter()
    fixtures_dir = pathlib.Path(__file__).parent / "fixtures" / directory

    if should_fail:
        assert expected_exception is not None
        with pytest.raises(expected_exception):
            adapter.from_dir_to_hub(
                dat_dir=str(fixtures_dir),
                hub_dataset="dummy-user/dummy-dataset",
                commit_message=commit_message
            )
        assert not mock_push.called
    else:
        adapter.from_dir_to_hub(
            dat_dir=str(fixtures_dir),
            hub_dataset="dummy-user/dummy-dataset",
            commit_message=commit_message
        )
        mock_push.assert_called_once_with(
            'dummy-user/dummy-dataset',
            commit_message=commit_message
        )

def sort_conversations(
        data: list[StructuredConversationRecord]
) -> list[StructuredConversationRecord]:
    def sort_messages(conv: StructuredConversationRecord) -> StructuredConversationRecord:
        return {
            "messages": sorted(
                conv["messages"],
                key=lambda m: m["role"] + m["content"]
            )
        }

    return sorted(
        [sort_messages(d) for d in data],
        key=str
    )

@pytest.mark.parametrize("expected", [
    EXPECTED_CONVERSATION_01_STRUCTURED,
    EXPECTED_CONVERSATION_02_STRUCTURED,
    EXPECTED_CONVERSATION_03_STRUCTURED,
    EXPECTED_CONVERSATION_04_STRUCTURED,
])
def test_roundtrip_dataset_to_dat_to_dataset(expected: list[StructuredConversationRecord]) -> None:
    adapter = ConversationAdapter()

    # Pylance: Type of from_list() is partially unknown
    dataset = Dataset.from_list(  # type: ignore[reportUnknownMemberType]
        cast(list[dict[str, Any]], expected)
    )

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".dat") as tmp:
        tmp_path = pathlib.Path(tmp.name)
        adapter.from_dataset_to_dat(dataset, str(tmp_path))

    roundtripped = adapter.from_dat_to_dataset(str(tmp_path))

    assert sort_conversations(
        # Pylance: Type of to_list() is partially unknown
        roundtripped.to_list()  # type: ignore[reportUnknownMemberType]
    ) == sort_conversations(expected)

def test_from_hub_to_dataset_wide_dict_conversation():
    adapter = ConversationAdapter()
    # Pylance: Type of from_dict() is partially unknown
    dummy_split = Dataset.from_list(  # type: ignore[reportUnknownMemberType]
        EXPECTED_CONVERSATION_04_WIDE
    )
    fake_dataset_dict = DatasetDict({
        "train": dummy_split,
        "validation": dummy_split,
        "test": dummy_split
    })

    with patch(
        "fifo_tool_datasets.sdk.hf_dataset_adapters.conversation.load_dataset",
        return_value=fake_dataset_dict
    ):
        result = adapter.from_hub_to_dataset_wide_dict("username/dataset")
        assert isinstance(result, DatasetDict)
        assert set(cast(list[str], result.keys())) == {"train", "validation", "test"}
        for _split_name, split_data in cast(dict[str, Dataset], result).items():
            assert split_data.column_names == ["id_conversation", "id_message", "role", "content"]
            assert all(isinstance(v, int) and v >= 0 for v in cast(list[int], split_data["id_conversation"]))
            assert all(isinstance(v, int) and v >= 0 for v in cast(list[int], split_data["id_message"]))

def test_from_hub_to_dataset_dict_conversation():
    adapter = ConversationAdapter()

    dummy_split = Dataset.from_list(  # type: ignore[reportUnknownMemberType]
        EXPECTED_CONVERSATION_04_WIDE
    )

    fake_dataset_dict = DatasetDict({
        "train": dummy_split,
        "validation": dummy_split,
        "test": dummy_split
    })

    with patch(
        "fifo_tool_datasets.sdk.hf_dataset_adapters.conversation.load_dataset",
        return_value=fake_dataset_dict
    ):
        result = adapter.from_hub_to_dataset_dict("username/dataset")

        assert isinstance(result, DatasetDict)
        assert set(cast(list[str], result.keys())) == {"train", "validation", "test"}

        for _split_name, split_data in cast(dict[str, Dataset], result).items():
            assert isinstance(split_data, Dataset)
            for record in cast(list[dict[str, str]], split_data):
                assert "messages" in record
                assert isinstance(record["messages"], list)
                assert all("role" in m and "content" in m for m in record["messages"])
