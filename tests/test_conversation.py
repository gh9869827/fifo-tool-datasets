import pytest
from fifo_dev_llm.dataset_adapters.common import DatasetAdapter
from fifo_dev_llm.dataset_adapters.conversation import ConversationAdapter, ConversationRecord
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
)
import pathlib

def check_dat_to_wide_dataset(
    adapter: DatasetAdapter,
    filename: str,
    expected: list[ConversationRecord],
) -> None:
    """
    Asserts that the wide-format dataset matches the expected content for a given fixture DAT file.
    """
    path = pathlib.Path(__file__).parent / "fixtures" / filename
    dataset = adapter._from_dat_to_wide_dataset(str(path))  # pylint: disable=protected-access  # type: ignore[reportPrivateUsage]

    assert isinstance(dataset, Dataset)
    assert len(dataset) == len(expected)

    for i, record in enumerate(expected):
        for key, value in record.items():
            assert dataset[i][key] == value

def test_from_dat_to_wide_dataset_01() -> None:
    adapter = ConversationAdapter()
    expected: list[ConversationRecord] = [
        {"id_conversation": 1, "id_message": 1, "role": "system",     "content": "You are a helpful assistant."},
        {"id_conversation": 1, "id_message": 2, "role": "directives", "content": "Always be concise."},
        {"id_conversation": 1, "id_message": 3, "role": "user",       "content": "Hello!"},
        {"id_conversation": 1, "id_message": 4, "role": "assistant",  "content": "Hi there. How can I help?"},
    ]
    check_dat_to_wide_dataset(adapter, "conversation_01.dat", expected)

def test_from_dat_to_wide_dataset_02() -> None:
    adapter = ConversationAdapter()
    expected: list[ConversationRecord] = [
        {"id_conversation": 1, "id_message": 1, "role": "system",     "content": "You are a helpful assistant."},
        {"id_conversation": 1, "id_message": 2, "role": "directives", "content": "Always be concise."},
        {"id_conversation": 1, "id_message": 3, "role": "user",       "content": "Hello!"},
        {"id_conversation": 1, "id_message": 4, "role": "assistant",  "content": "Hi there. How can I help?"},
        {"id_conversation": 2, "id_message": 1, "role": "system",     "content": "You are a helpful assistant. #2"},
        {"id_conversation": 2, "id_message": 2, "role": "directives", "content": "Always be concise. #2"},
        {"id_conversation": 2, "id_message": 3, "role": "user",       "content": "Hello! #2"},
        {"id_conversation": 2, "id_message": 4, "role": "assistant",  "content": "Hi there. How can I help? #2"},
    ]
    check_dat_to_wide_dataset(adapter, "conversation_02.dat", expected)

def test_from_dat_to_wide_dataset_03() -> None:
    adapter = ConversationAdapter()
    expected: list[ConversationRecord] = [
        {"id_conversation": 1, "id_message": 1, "role": "system",     "content": "You are a helpful assistant."},
        {"id_conversation": 1, "id_message": 2, "role": "directives", "content": "Always be concise."},
        {"id_conversation": 1, "id_message": 3, "role": "user",       "content": "Hello!"},
        {"id_conversation": 1, "id_message": 4, "role": "assistant",  "content": "Hi there. How can I help?"},
        {"id_conversation": 2, "id_message": 1, "role": "system",     "content": "You are a helpful assistant. #2"},
        {"id_conversation": 2, "id_message": 2, "role": "directives", "content": "Always be concise. #2"},
        {"id_conversation": 2, "id_message": 3, "role": "user",       "content": "Hello! #2"},
        {"id_conversation": 2, "id_message": 4, "role": "assistant",  "content": "Hi there. How can I help? #2"},
        {"id_conversation": 3, "id_message": 1, "role": "system",     "content": "You are a helpful assistant. #3"},
        {"id_conversation": 3, "id_message": 2, "role": "user",       "content": "Hello! #3"},
        {"id_conversation": 3, "id_message": 3, "role": "assistant",  "content": "Hi there. How can I help? #3"},
    ]
    check_dat_to_wide_dataset(adapter, "conversation_03.dat", expected)

def test_from_dat_to_wide_dataset_04() -> None:
    adapter = ConversationAdapter()
    expected: list[ConversationRecord] = [
        {"id_conversation": 1, "id_message": 1, "role": "system",     "content": "You are a helpful assistant."},
        {"id_conversation": 1, "id_message": 2, "role": "directives", "content": "Always be concise."},
        {"id_conversation": 1, "id_message": 3, "role": "user",       "content": "Hello!"},
        {"id_conversation": 1, "id_message": 4, "role": "assistant",  "content": "Hi there. How can I help?"},
        {"id_conversation": 2, "id_message": 1, "role": "system",     "content": "You are a helpful assistant. #2"},
        {"id_conversation": 2, "id_message": 2, "role": "directives", "content": "Always be concise. #2"},
        {"id_conversation": 2, "id_message": 3, "role": "user",       "content": "Hello! #2"},
        {"id_conversation": 2, "id_message": 4, "role": "assistant",  "content": "Hi there. How can I help? #2"},
        {"id_conversation": 3, "id_message": 1, "role": "user",       "content": "Hello! #3"},
        {"id_conversation": 3, "id_message": 2, "role": "assistant",  "content": "Hi there. How can I help? #3"},
    ]
    check_dat_to_wide_dataset(adapter, "conversation_04.dat", expected)

def test_from_dat_to_wide_dataset_broken_01():
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "conversation_broken_01.dat"

    with pytest.raises(ValueError, match="Conversation 1 is empty"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

def test_from_dat_to_wide_dataset_broken_02():
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "conversation_broken_02.dat"

    with pytest.raises(ValueError, match="Two consecutive '>' tags detected at conversation 1, message 3"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

def test_from_dat_to_wide_dataset_broken_03():
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "conversation_broken_03.dat"

    with pytest.raises(ValueError, match="Two consecutive '<' tags detected at conversation 1, message 4"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

def test_from_dat_to_wide_dataset_broken_04():
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "conversation_broken_04.dat"

    with pytest.raises(ValueError, match="Conversation 3 is not closed properly"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

def test_from_dat_to_wide_dataset_broken_05():
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "conversation_broken_05.dat"

    with pytest.raises(ValueError, match="Empty tag '>' detected at conversation 2, message 3"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]

def test_from_dat_to_wide_dataset_broken_06():
    adapter = ConversationAdapter()
    path = pathlib.Path(__file__).parent / "fixtures" / "conversation_broken_06.dat"

    with pytest.raises(ValueError, match="Empty tag '\\$' detected at conversation 2, message 1"):
        adapter._from_dat_to_wide_dataset(str(path)) # pylint: disable=protected-access # type: ignore[reportPrivateUsage]
