from abc import ABC, abstractmethod
import os
from typing import Iterator, TypedDict, cast
import huggingface_hub as hub
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
    DatasetDict
)

JsonConversation = list[dict[str, list[dict[str, str]]]]
"""
List of conversations in JSON format.

Each conversation is a dictionary with:
    messages (list[dict[str, str]]): A list of message dictionaries.

Each message dictionary contains:
    role (str): The role of the speaker ("user", "assistant", "system", or "directives").
    content (str): The text content of the message.

    example: JsonConversation = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Convert 1200 feet into meters."},
                {"role": "assistant", "content": "1200 feet is equal to 365.76 meters."}
            ]
        }
    ]
"""

class StructureMessageRecord(TypedDict):
    """
    A TypedDict representing a structured message record.

    Attributes:
        role (str):
            The role of the message sender (e.g., 'user', 'assistant').

        content (str):
            The textual content of the message.
    """
    role: str
    content: str

class StructuredConversationRecord(TypedDict):
    """
    A full conversation consisting of multiple messages.

    This format is used to represent the structured JSON format
    loaded into `datasets.Dataset` for fine-tuning language models.
    It contrasts with wide-format datasets, where each row is a single message.

    Attributes:
        messages (list[StructureMessageRecord]):
            A list of messages that make up the conversation.
    """
    messages: list[StructureMessageRecord]

class DatasetAdapter(ABC):
    """
    Abstract base class for adapting various DAT formats into Huggingface-compatible datasets.

    Subclasses must implement the format-specific methods mark as abstract.

    - from_dat_to_wide_dataset(dat_filename)
        Parses a DAT file into a wide-format dataset.

    - from_hub_to_dataset_wide_dict(hub_dataset)
        Loads a wide-format dataset from the Huggingface hub.

    - from_wide_dataset_to_json(dataset)
        Converts a wide-format dataset into JSON-style format.

    - from_dat_to_hub(dat_filename, hub_dataset, commit_message, seed=42,
                      split_ratios=(0.7, 0.15, 0.15))
        Uploads a dataset to the Huggingface Hub from a DAT file, partitioned into train/val/test.

    - from_dir_to_hub(dat_dir, hub_dataset, commit_message)
        Uploads multiple split .dat files from a directory to the Hugging Face Hub.

    - from_hub_to_dataset_dict(hub_dataset)
        Builds a Huggingface Dataset from a JSON loader function.

    - from_dat_to_dataset(filename)
        Builds a Dataset from a DAT file using format-specific parsing.

    - from_dataset_to_dat(dataset, dat_filename)
        Converts a structured Hugging Face Dataset to a .dat file using wide-format conversion.

    - from_wide_dataset_to_dat(wide_dataset, dat_filename)
        Writes a wide-format dataset to a DAT file. (abstract method)

    - from_dat_to_wide_dataset(dat_filename)
        Parses a DAT file into a wide-format Huggingface Dataset. (abstract method)

    - from_dataset_to_wide_dataset(dataset)
        Converts a Hugging Face Dataset into a wide-format Dataset suitable for .dat serialization. 
        (abstract method)

    - from_wide_dataset_to_json(wide_dataset)
        Converts a wide-format dataset into JSON-style format. (abstract method)

    - _iter_structured_records(dataset)
        Returns a statically typed iterator over a structured Hugging Face Dataset where each record
        represents a conversation.

    Dataset: A Hugging Face `Dataset` object where each record is a full conversation.
    This format is directly suitable for fine-tuning LLMs.

    Example:
    Dataset({
        features: ['messages'],
        num_rows: 12
    })

    Wide dataset: A Hugging Face `Dataset` object structured in a flat, message-level layout.
    Instead of storing a list of messages per conversation, each row represents a single message,
    with fields such as conversation ID, message ID, role, and content.

    Example:
    Dataset({
        features: ['id_conversation', 'id_message', 'role', 'content'],
        num_rows: 78
    })

    Each adapter subclass defines its own wide format.

    .dat: A plain-text editable format specific to each adapter. Useful for manual inspection or 
    editing.

    hub: A Hugging Face Hub dataset, storing the wide-format version.

    json: A list of 'messages' dictionaries. This is the internal representation used to build the 
    `Dataset`.


    Conversion Matrix (source -> target):

                  | dataset          | wide_dataset | .dat      | hub        | json
    --------------|------------------|--------------|-----------|------------|----------
    dataset       | —                | direct       | indirect  | —          | —
    wide_dataset  | —                | —            | direct    | —          | direct
    .dat          | indirect         | direct       | —         | indirect   | —
    hub           | indirect(dict)   | direct(dict) | —         | —          | —
    json          | —                | —            | —         | —          | —

    Notes:
    direct     = The method performs the conversion directly.
    indirect   = The method calls other methods to perform the conversion.
    (dict)     = the Dataset that is returned is a DictDataset
    Not all directions are supported (e.g., JSON -> others is not implemented).
    """

    def from_dat_to_hub(
        self,
        dat_filename: str,
        hub_dataset: str,
        commit_message: str,
        seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> None:
        """
        Uploads a dataset to the Huggingface Hub from a DAT file, partitioned into train/val/test.

        Args:
            dat_filename (str):
                Path to the DAT file.

            hub_dataset (str):
                Name of the Hugging Face dataset repository.

            commit_message (str):
                Required commit message for upload.

            seed (int):
                Seed for deterministic shuffling.

            split_ratios (tuple):
                Tuple of (train, validation, test) ratios. Must sum to 1.0.

        Raises:
            ValueError: If `commit_message` is missing or the split ratios are invalid.
        """
        if not commit_message:
            raise ValueError("Commit message is required")

        # Split the dataset using the helper
        splits = self.from_dat_to_dataset_dict(
            dat_filename=dat_filename,
            seed=seed,
            split_ratios=split_ratios,
        )

        # convert back to wide format
        splits = {
            split_name: self.from_dataset_to_wide_dataset(split_dataset)
            for split_name, split_dataset in cast(dict[str, Dataset], splits).items()
        }

        # Push each split to hub
        DatasetDict(splits).push_to_hub(  # type: ignore[reportUnknownMemberType]
            hub_dataset, commit_message=commit_message
        )

    def from_dir_to_hub(
        self,
        dat_dir: str,
        hub_dataset: str,
        commit_message: str,
    ) -> None:
        """
        Uploads multiple split `.dat` files from a directory to the Hugging Face Hub.

        Expects files named `train.dat`, `validation.dat`, and/or `test.dat` — and nothing else.

        Args:
            dat_dir (str):
                Directory containing `.dat` files for each split.

            hub_dataset (str):
                Hugging Face dataset repo ID (e.g., `your-username/your-dataset`).

            commit_message (str):
                Commit message for upload.

        Raises:
            ValueError: If commit_message is missing or invalid files are present.
        """
        if not commit_message:
            raise ValueError("Commit message is required")

        expected_files = {"train.dat", "validation.dat", "test.dat"}
        optional_files = {"README.md", "LICENSE"}
        actual_files = set(os.listdir(dat_dir))

        if not expected_files.issubset(actual_files):
            missing = expected_files - actual_files
            raise ValueError(
                "Directory must contain train.dat, validation.dat, and test.dat. "
                f"Missing required files: {', '.join(sorted(missing))}"
            )

        extras = {
            f for f in actual_files - expected_files - optional_files if not f.startswith('.')
        }
        if extras:
            raise ValueError(
                "Directory contains unsupported files: " + ", ".join(sorted(extras))
            )

        splits = {}
        for split_name in ("train", "validation", "test"):
            dat_path = os.path.join(dat_dir, f"{split_name}.dat")
            ds = self.from_dat_to_wide_dataset(dat_path)
            splits[split_name] = ds

        api = hub.HfApi()
        for extra in ("README.md", "LICENSE"):
            local_path = os.path.join(dat_dir, extra)
            if os.path.exists(local_path):
                try:
                    # Pylance: Type of hf_hub_download() is partially unknown
                    remote_path = hub.hf_hub_download(  # type: ignore[reportUnknownMemberType]
                        hub_dataset, filename=extra, repo_type="dataset"
                    )
                    with open(remote_path, "rb") as f:
                        remote_content = f.read()
                except (hub.errors.RepositoryNotFoundError, hub.errors.EntryNotFoundError):
                    remote_content = None
                with open(local_path, "rb") as f:
                    local_content = f.read()
                if remote_content != local_content:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=extra,
                        repo_id=hub_dataset,
                        repo_type="dataset",
                        commit_message=commit_message,
                    )

        # Pylance: Type of push_to_hub() is partially unknown
        DatasetDict(splits).push_to_hub(  # type: ignore[reportUnknownMemberType]
            hub_dataset, commit_message=commit_message
        )

    def from_dat_to_dataset_dict(
        self,
        dat_filename: str,
        seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    ) -> DatasetDict:
        """
        Return a shuffled `DatasetDict` split into train/validation/test.

        Args:
            dat_filename (str):
                Path to the input `.dat` file.

            seed (int):
                Seed used when shuffling before the split.

            split_ratios (tuple[float, float, float]):
                Tuple of `(train, validation, test)` ratios. Must
                sum to `1.0` and be non-negative.

        Returns:
            datasets.DatasetDict:
                a `DatasetDict` with `"train"`, `"validation"` and `"test"` splits.

        Raises:
            ValueError: If ratios are negative, do not sum to `1.0` or if any
            resulting split is empty.
        """
        if any(r < 0 for r in split_ratios) or not abs(sum(split_ratios) - 1.0) < 1e-6:
            if any(r < 0 for r in split_ratios):
                raise ValueError("Split ratios must be non-negative")
            raise ValueError("Split ratios must sum to 1.0")

        dataset = self.from_dat_to_dataset(dat_filename)
        dataset = dataset.shuffle(seed=seed)

        val_ratio, test_ratio = split_ratios[1], split_ratios[2]
        total = len(dataset)

        if total < 3:
            raise ValueError("Dataset too small to split into train/val/test sets")

        val_size = int(total * val_ratio)
        test_size = int(total * test_ratio)
        train_size = total - val_size - test_size

        if val_size == 0:
            raise ValueError(
                f"Validation dataset is empty (val_ratio={val_ratio}, total={total})"
            )

        if test_size == 0:
            raise ValueError(
                f"Test dataset is empty (test_ratio={test_ratio}, total={total})"
            )

        if train_size == 0:
            raise ValueError(
                f"Train dataset is empty (total={total}, val={val_size}, test={test_size})"
            )

        return DatasetDict(
            {
                # Pylance: Type of select() is partially unknown
                "train": dataset.select(  # type: ignore[reportUnknownMemberType]
                    range(train_size)
                ),
                # Pylance: Type of select() is partially unknown
                "validation": dataset.select(  # type: ignore[reportUnknownMemberType]
                    range(train_size, train_size + val_size)
                ),
                # Pylance: Type of select() is partially unknown
                "test": dataset.select(  # type: ignore[reportUnknownMemberType]
                    range(train_size + val_size, total)
                ),
            }
        )

    def from_hub_to_dataset_dict(
        self,
        hub_dataset: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> DatasetDict:
        """
        Loads a split dataset from the Hugging Face Hub and converts each split
        to JSON-format records using adapter-specific logic.

        Args:
            hub_dataset (str):
                Name of the dataset to load from the Hugging Face Hub.

        Keyword Args:
            revision (str | None):
                Git revision (branch, tag or commit SHA) to download. If `None`,
                the latest commit on the dataset's default branch is used.
            cache_dir (str | None):
                Directory where downloaded files should be stored. If `None`,
                the default Hugging Face cache location is used.

        Returns:
            DatasetDict:
                A dictionary of splits ('train', 'validation', 'test').
        """
        wide_dataset = self.from_hub_to_dataset_wide_dict(
            hub_dataset,
            revision=revision,
            cache_dir=cache_dir
        )

        required_splits = ("train", "validation", "test")

        # Ensure all required splits are present
        for split in required_splits:
            if split not in wide_dataset:
                raise ValueError(f"Missing required split: '{split}'")

        # Sort each required split and convert to JSON
        json_splits = {
            split: self.from_wide_dataset_to_json(
                wide_dataset[split]
            )
            for split in required_splits
        }

        return DatasetDict({
            # Pylance: Type of from_list() is partially unknown
            split: Dataset.from_list(  # type: ignore[reportUnknownMemberType]
                data
            )
            for split, data in json_splits.items()
        })

    def from_dat_to_dataset(self, filename: str) -> Dataset:
        """
        Builds a Dataset from a DAT file using format-specific parsing.

        Args:
            filename (str):
                Path to the input DAT file.

            seed (int | None):
                Seed for deterministic shuffling (optional).

        Returns:
            Dataset:
                The parsed Dataset object.
        """
        # Pylance: Type of from_list() is partially unknown
        return Dataset.from_list(  # type: ignore[reportUnknownMemberType]
            self.from_wide_dataset_to_json(
                self.from_dat_to_wide_dataset(filename)
            )
        )

    def from_dataset_to_dat(self, dataset: Dataset, dat_filename: str) -> None:
        """
        Converts a structured Hugging Face Dataset (list of conversations)
        to a `.dat` file using wide-format conversion.

        Args:
            dataset (Dataset):
                A structured dataset with a 'messages' field per example.

            dat_filename (str):
                Output path for the `.dat` file.
        """
        wide_dataset = self.from_dataset_to_wide_dataset(dataset)
        self.from_wide_dataset_to_dat(wide_dataset=wide_dataset, dat_filename=dat_filename)

    def _iter_structured_records(self, dataset: Dataset) -> Iterator[StructuredConversationRecord]:
        """
        Returns a statically typed iterator over a structured Hugging Face Dataset where
        each record represents a conversation (i.e., a list of messages with roles and content).

        This helper enables field access like `record["messages"][i]["role"]`.

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each record is a conversation with a `messages` field.

        Returns:
            Iterator[StructuredConversationRecord]:
                An iterator over structured conversation examples.
        """
        return cast(Iterator[StructuredConversationRecord], iter(dataset))

    @abstractmethod
    def from_wide_dataset_to_dat(self, wide_dataset: Dataset, dat_filename: str) -> None:
        """
        Writes a wide-format dataset to a DAT file.

        Args:
            wide_dataset (Dataset):
                Wide-format dataset.

            dat_filename (str):
                Output path for the DAT file.
        """

    @abstractmethod
    def from_dat_to_wide_dataset(self, dat_filename: str) -> Dataset:
        """
        Parses a DAT file into a wide-format Huggingface Dataset.

        Args:
            dat_filename (str):
                Path to the input DAT file.

        Returns:
            Dataset:
                A wide-format Dataset.

        Raises:
            SyntaxError: If the file is malformed.
        """

    @abstractmethod
    def from_dataset_to_wide_dataset(self, dataset: Dataset) -> Dataset:
        """
        Converts a Hugging Face Dataset into a wide-format Dataset suitable for `.dat`
        serialization.

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each item is a structured record, i.e.
                a conversation with a list of messages.

        Returns:
            Dataset:
                A wide-format Dataset.
        """

    @abstractmethod
    def from_hub_to_dataset_wide_dict(
        self,
        hub_dataset: str,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> DatasetDict:
        """
        Loads a wide-format dataset from the Hugging Face Hub, organized into train, validation,
        and test splits.

        This method assumes the dataset follows an adapter-specific wide schema and must be
        implemented by subclasses to validate and parse the structure accordingly.

        Args:
            hub_dataset (str):
                The Hugging Face dataset identifier (e.g., "username/dataset").

        Keyword Args:
            revision (str | None):
                Git revision (branch, tag or commit SHA) to download. If `None`,
                the latest commit on the dataset's default branch is used.
            cache_dir (str | None):
                Directory where downloaded files should be stored. If `None`,
                the default Hugging Face cache location is used.

        Returns:
            DatasetDict:
                A dictionary containing train, validation, and test splits, each in wide format.

        Raises:
            ValueError:
                If the dataset is not split or does not match the expected wide-format structure.
        """

    @abstractmethod
    def from_wide_dataset_to_json(self, wide_dataset: Dataset) -> JsonConversation:
        """
        Converts a wide-format dataset into JSON-style format.

        Args:
            wide_dataset (Dataset):
                wide-format dataset.

        Returns:
            JsonConversation:
                A list of dicts with `messages` containing user and assistant messages.
        """
