import re
from typing import Dict, Iterator, Literal, TypedDict, cast
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
    DatasetDict,
    # Pylance: Type of load_dataset is partially unknown
    load_dataset  # type: ignore[reportUnknownVariableType]
)
from .common import DatasetAdapter, JsonConversation

Role = Literal[
    "system",
    "directives",
    "user",
    "assistant"
]

TAG_TO_ROLE = {
    ">": "user",
    "<": "assistant",
    "$": "system",
    "!": "directives"
}

class WideConversationRecord(TypedDict):
    """
    A single message in wide-format, used for flattened datasets.

    Attributes:
        id_conversation (int):
            Identifier for the conversation this message belongs to.
        
        id_message (int):
            Message index within the conversation (0-based).
        
        role (Role):
            Role of the speaker (e.g., 'user', 'assistant', 'system', or 'directives').
        
        content (str):
            Text content of the message.
    """
    id_conversation: int
    id_message: int
    role: Role
    content: str

class WideDataDict(TypedDict):
    """
    Column-wise dictionary representing a wide-format dataset.

    Each list corresponds to a full column in the dataset — same length across all keys.

    Attributes:
        id_conversation (list[int]):
            List of conversation IDs for all messages.

        id_message (list[int]):
            List of message indices within each conversation.

        role (list[str]):
            List of roles (as strings) for each message.

        content (list[str]):
            List of text content for each message.
    """
    id_conversation: list[int]
    id_message: list[int]
    role: list[str]
    content: list[str]

class ConversationAdapter(DatasetAdapter):
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
                "id_message": 0,
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "id_conversation": 0,
                "id_message": 1,
                "role": "directives",
                "content": "Always be concise."
            },
            {
                "id_conversation": 0,
                "id_message": 2,
                "role": "user",
                "content": "Convert 1200 feet into meters."
            },
            {
                "id_conversation": 0,
                "id_message": 3,
                "role": "assistant",
                "content": "1200 feet is equal to 365.76 meters."
            }
        ]

    Corresponding JSON format:
        [{"messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "directives", "content": "Always be concise."},
            {"role": "user", "content": "Convert 1200 feet into meters."},
            {"role": "assistant", "content": "1200 feet is equal to 365.76 meters."}
        ]}]
    """

    def from_dat_to_wide_dataset(self, dat_filename: str) -> Dataset:
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
        flat_data: WideDataDict = {
            "id_conversation": [],
            "id_message": [],
            "role": [],
            "content": []
        }

        # we start from -1 so that the first id that is used is actually 0 since we increment before
        # use
        id_conversation, id_message = -1, -1
        role, content = None, None

        line_number = 0
        with open(dat_filename, "r", encoding="utf-8") as f:
            prev_tag = None
            for line in f:
                line_number += 1
                line = line.rstrip("\r\n")

                if re.match(r'^(---|>|<|\$|\!)\s+$', line) is not None:
                    raise ValueError(f"Error: $ ! --- > or < followed by spaces, "
                                     f"line {line_number}")

                if prev_tag is not None and line in ["---", ">", "<", "$", "!"]:
                    if content is None:
                        raise ValueError(
                            f"Empty tag '{prev_tag}' detected at conversation {id_conversation}, "
                            f"message {id_message}, line {line_number}"
                        )

                if content is not None and line in ["---", ">", "<", "$", "!"]:
                    assert role is not None
                    flat_data["id_conversation"].append(id_conversation)
                    flat_data["id_message"].append(id_message)
                    flat_data["role"].append(role)
                    flat_data["content"].append(content)

                if line in [">", "<"] and prev_tag == line:
                    raise ValueError(
                        f"Two consecutive '{line}' tags detected at conversation "
                        f"{id_conversation}, message {id_message}, line {line_number}"
                    )

                if line == "---":
                    if id_conversation >= 0 and (content is None and prev_tag is None):
                        raise ValueError(f"Conversation {id_conversation} is empty, "
                                         f"line {line_number}")
                    id_conversation += 1
                    id_message = -1
                    content = None
                    prev_tag = None
                elif line in TAG_TO_ROLE:
                    id_message += 1
                    role = TAG_TO_ROLE[line]
                    content = None
                    prev_tag = line
                else:
                    if content is None:
                        content = line
                    else:
                        content = f"{content}\n{line}"

        if prev_tag is not None:
            raise ValueError(f"Conversation {id_conversation} is "
                             f"not closed properly, last line {line_number}")

        # Pylance: Type of from_dict() is partially unknown
        return Dataset.from_dict(flat_data)  # type: ignore[reportUnknownMemberType]

    def from_dataset_to_wide_dataset(self, dataset: Dataset) -> Dataset:
        """
        Converts a structured dataset (list of conversations) to a flat wide-format Hugging Face
        Dataset.

        Args:
            structured_dataset (Dataset):
                The dataset containing a list of conversations.

        Returns:
            Dataset:
                A flat dataset with fields: id_conversation, id_message, role, and content.
        """
        flat_data: WideDataDict = {
            "id_conversation": [],
            "id_message": [],
            "role": [],
            "content": []
        }

        for convo_idx, example in enumerate(self._iter_structured_records(dataset)):
            messages = example["messages"]
            for msg_idx, msg in enumerate(messages):
                flat_data["id_conversation"].append(convo_idx)
                flat_data["id_message"].append(msg_idx)
                flat_data["role"].append(msg["role"])
                flat_data["content"].append(msg["content"])

        # Pylance: Type of from_dict() is partially unknown
        return Dataset.from_dict(flat_data)  # type: ignore[reportUnknownMemberType]

    def from_hub_to_dataset_wide_dict(
        self,
        hub_dataset: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> DatasetDict:
        """
        Loads a conversation-style dataset from the Hugging Face Hub and returns it as a split
        DatasetDict.

        Each split (`train`, `validation`, `test`) must be present and follow the wide-format schema
        expected by the adapter: one row per message, with fields like `id_conversation`,
        `id_message`, `role`, and `content`.

        All splits are validated and sorted by `id_conversation` and `id_message`.

        Args:
            hub_dataset (str):
                The Hugging Face dataset identifier (e.g., "username/dataset").

        Keyword Args:
            revision (str | None):
                Git revision to download. If `None`, the latest commit on the
                dataset's default branch is used.
            cache_dir (str | None):
                Location to store downloaded files. Uses the default HF cache if
                omitted.

        Returns:
            DatasetDict:
                A dictionary containing train, validation, and test splits in wide format, sorted by
                conversation and message order.

        Raises:
            ValueError:
                If the dataset is not split, required splits are missing, or expected fields are
                absent.
        """
        wide_dataset = load_dataset(
            hub_dataset,
            revision=revision,
            cache_dir=cache_dir
        )

        if not isinstance(wide_dataset, DatasetDict):
            raise ValueError("Expected a split DatasetDict, but got a flat Dataset.")

        required_splits = {"train", "validation", "test"}
        actual_splits = set(cast(list[str], wide_dataset.keys()))
        missing_splits = required_splits - actual_splits
        if missing_splits:
            raise ValueError(f"Missing required splits: {', '.join(sorted(missing_splits))}")

        required_columns = {"id_conversation", "id_message", "role", "content"}
        sorted_dataset = DatasetDict()

        for split_name, split_data in cast(dict[str, Dataset], wide_dataset).items():
            actual_columns = set(split_data.column_names)
            missing_columns = required_columns - actual_columns
            if missing_columns:
                raise ValueError(f"Split '{split_name}' is missing required "
                                 f"columns: {', '.join(missing_columns)}")

            min_id_conversation = min(cast(list[int], split_data["id_conversation"]))
            min_id_message = min(cast(list[int], split_data["id_message"]))

            if min_id_conversation < 0 or min_id_message < 0:
                raise ValueError(
                    f"Split '{split_name}' contains negative values "
                    "in 'id_conversation' or 'id_message'"
                )

            sorted_dataset[split_name] = split_data.sort(["id_conversation", "id_message"])

        return sorted_dataset

    def from_wide_dataset_to_json(self, wide_dataset: Dataset) -> JsonConversation:
        """
        Converts a wide-format conversation dataset into JSON-style format.

        Args:
            wide_dataset (Dataset):
                Dataset with `id_conversation`, `id_message`, `role`, and `content` fields.

        Returns:
            JsonConversation:
                A list of dicts with `messages` containing role/content pairs for each conversation.
        """
        sorted_wide_dataset = wide_dataset.sort(["id_conversation", "id_message"])

        id_conversation = -1
        conversations: JsonConversation = []
        messages: list[Dict[str, str]] = []
        for record in self._iter_wide_records(sorted_wide_dataset):
            if record["id_conversation"] != id_conversation:
                id_conversation = record["id_conversation"]
                if messages:
                    conversations.append({"messages" : messages})
                    messages = []

            role = record["role"]
            if role in ["assistant", "user", "system", "directives"]:
                messages.append({"role": role, "content": record["content"]})
            else:
                raise ValueError(f"Unknown role: {role}")

        if messages:
            conversations.append({"messages" : messages})

        return conversations

    def from_wide_dataset_to_dat(self, wide_dataset: Dataset, dat_filename: str) -> None:
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
            for record in self._iter_wide_records(wide_dataset):
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

    def _iter_wide_records(self, wide_dataset: Dataset) -> Iterator[WideConversationRecord]:
        """
        Returns an iterator over a Hugging Face Dataset with each record typed as a
        WideConversationRecord.

        This helper function enables static type checking and clean field access like
        `record["id_conversation"]`, `record["role"]`, etc., which are expected fields in
        wide-format conversation datasets.

        Args:
            wide_dataset (Dataset):
                A Hugging Face Dataset where each row contains fields
                `id_conversation`, `id_message`, `role`, and `content`.

        Returns:
            Iterator[WideConversationRecord]:
                An iterator over the dataset where each item is typed as a ConversationRecord.
        """
        return cast(Iterator[WideConversationRecord], iter(wide_dataset))
