import re
from typing import Dict, Iterator, Literal, TypedDict, cast
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
    # Pylance: Type of load_dataset is partially unknown
    load_dataset  # type: ignore[reportUnknownVariableType]
)
from .common import DatasetAdapter, JsonConversation

Role = Literal["system", "directives", "user", "assistant"]

class ConversationRecord(TypedDict):
    id_conversation: int
    id_message: int
    role: Role
    content: str

class Conversation(DatasetAdapter):
    """
    Adapter for handling multi-turn conversation datasets.

    This format supports mixed roles (user, assistant, system, directives).

    This class supports:
        - Reading a conversation DAT file into a Huggingface `Dataset`
        - Uploading conversation data to the Huggingface hub
        - Downloading conversation hub datasets into DAT files
        - Converting between wide-format and JSON-format conversation data

    Expected DAT syntax:
      - Conversations are separated by `---`
      - Each message is preceded by a tag line:
          >  user
          <  assistant
          $  system
          !  directives

    Example:
        ---
        $
        You are a helpful assistant.
        !
        Always be concise.
        >
        Hello!
        <
        Hi there. How can I help?
        ---

    Corresponding dataset stored on the hub:
        [
            {
                "id_conversation": 0,
                "id_message": 1,
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "id_conversation": 0,
                "id_message": 2,
                "role": "directives",
                "content": "Always be concise."
            },
            {
                "id_conversation": 0,
                "id_message": 3,
                "role": "user",
                "content": "Convert 1200 feet into meters."
            },
            {
                "id_conversation": 0,
                "id_message": 4,
                "role": "assistant",
                "content": "1200 feet is equal to 365.76 meters."
            }
        ]

    Corresponding JSON format:
        [{"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Always be concise."},
            {"role": "user", "content": "Convert 1200 feet into meters."},
            {"role": "assistant", "content": "1200 feet is equal to 365.76 meters."}
        ]}]
    """
    def _from_dat_to_wide_dataset(self, dat_filename: str) -> Dataset:
        """
        Parses a conversation DAT file into a wide-format Hugging Face Dataset.

        Args:
            dat_filename (str):
                Path to the input DAT file.

        Returns:
            Dataset:
                A Dataset with four fields: `id_conversation`, `id_message`, `role`, and `content`.

        Raises:
            ValueError: If the file contains malformed tags or syntax violations.
        """
        class RawDataDict(TypedDict):
            id_conversation: list[int]
            id_message: list[int]
            role: list[str]
            content: list[str]

        data: RawDataDict = {
                "id_conversation": [],
                "id_message": [],
                "role" : [],
                "content": []
            }
        id_conversation, id_message = 0, 0
        role, content = None, None

        i = 0
        with open(dat_filename, "r", encoding="utf-8") as f:
            prev_tag = None
            for line in f:
                i += 1
                line = line.rstrip("\r\n")

                if re.match(r'^(---|>|<|\$|\!)\s+$', line) is not None:
                    raise ValueError(f"Error: $ ! --- > or < followed by spaces, line {i}")

                if content is not None and line in ["---", ">", "<", "$", "!"]:
                    assert role is not None
                    data["id_conversation"].append(id_conversation)
                    data["id_message"].append(id_message)
                    data["role"].append(role)
                    data["content"].append(content)

                if line == "---":
                    id_conversation += 1
                    id_message = 0
                    content = None
                    prev_tag = None
                elif line == ">":
                    assert prev_tag != ">"
                    id_message += 1
                    role = "user"
                    content = None
                    prev_tag = line
                elif line == "<":
                    assert prev_tag != "<"
                    id_message += 1
                    role = "assistant"
                    content = None
                    prev_tag = line
                elif line == "$":
                    id_message += 1
                    role = "system"
                    content = None
                    prev_tag = line
                elif line == "!":
                    id_message += 1
                    role = "directives"
                    content = None
                    prev_tag = line
                else:
                    if content is None:
                        content = line
                    else:
                        content = f"{content}\n{line}"

        # Pylance: Type of from_dict() is partially unknown
        return Dataset.from_dict(data) # type: ignore[reportUnknownMemberType]

    def _load_from_hub(self, hub_dataset: str) -> Dataset:
        """
        Loads a conversation dataset from the Hugging Face hub.

        Args:
            hub_dataset (str):
                The Hugging Face dataset identifier.

        Returns:
            Dataset:
                A dataset containing `id_conversation`, `id_message`, `role`, and `content` fields.
        """
        # Explicitly loading a split returns a Dataset, not a DatasetDict - safe to cast
        wide_dataset = cast(Dataset, load_dataset(hub_dataset, split="train"))

        # Check minimum values of 'id_conversation' and 'id_message' columns
        min_id_conversation = min(cast(list[int], wide_dataset["id_conversation"]))
        min_id_message = min(cast(list[int], wide_dataset["id_message"]))

        # Ensure the minimum values are >= 0
        if min_id_conversation < 0 or min_id_message < 0:
            raise ValueError("All 'id_conversation' and 'id_message' values must be >= 0")

        sorted_dataset = wide_dataset.sort(["id_conversation", "id_message"])

        return sorted_dataset

    def _from_wide_dataset_to_json(self, wide_dataset: Dataset) -> JsonConversation:
        """
        Converts a wide-format conversation dataset into JSON-style format.

        Args:
            wide_dataset (Dataset):
                Dataset with `id_conversation`, `id_message`, `role`, and `content` fields.

        Returns:
            JsonConversation:
                A list of dicts with `messages` containing role/content pairs for each conversation.
        """
        id_conversation = -1
        conversations: JsonConversation = []
        messages: list[Dict[str, str]] = []
        for record in self._iter_records(wide_dataset):
            if record["id_conversation"] != id_conversation:
                id_conversation = record["id_conversation"]
                if messages:
                    conversations.append({"messages" : messages})
                    messages = []

            role = record["role"]
            if role in ["assistant", "user", "system"]:
                messages.append({"role": role, "content": record["content"]})
            elif role == "directives":
                messages.append({"role": "user", "content": record["content"]})
            else:
                raise ValueError(f"Unknown role: {role}")

        if messages:
            conversations.append({"messages" : messages})

        return conversations


    def to_dat(self, wide_dataset: Dataset, dat_filename: str) -> None:
        """
        Writes a wide-format conversation dataset to a DAT file.

        Args:
            wide_dataset (Dataset):
                Dataset with `id_conversation`, `id_message`, `role`, and `content` fields.

            dat_filename (str):
                Output path for the DAT file.
        """
        id_conversation = -1
        with open(dat_filename, 'w', encoding='utf-8') as f:
            for record in self._iter_records(wide_dataset):
                if record["id_conversation"] != id_conversation:
                    id_conversation = record["id_conversation"]
                    f.write("---\n")
                role = record["role"]
                if role == "user":
                    f.write(">\n")
                elif role == "assistant":
                    f.write("<\n")
                elif role == "system":
                    f.write("$\n")
                elif role == "directives":
                    f.write("!\n")
                else:
                    raise ValueError("Unknown role")
                f.write(f"{record['content']}\n")
            f.write("---\n")

    def _iter_records(self, dataset: Dataset) -> Iterator[ConversationRecord]:
        """
        Returns an iterator over a Hugging Face Dataset with each record typed as a
        ConversationRecord.

        This helper function enables static type checking and clean field access like
        `record["id_conversation"]`, `record["role"]`, etc., which are expected fields in
        wide-format conversation datasets.

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each row contains fields
                `id_conversation`, `id_message`, `role`, and `content`.

        Returns:
            Iterator[ConversationRecord]:
                An iterator over the dataset where each item is typed as a ConversationRecord.
        """
        return cast(Iterator[ConversationRecord], iter(dataset))
