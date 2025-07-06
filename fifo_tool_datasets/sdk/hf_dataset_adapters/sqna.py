from typing import Dict, Iterator, cast
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
    DatasetDict,
    # Pylance: Type of load_dataset is partially unknown
    load_dataset  # type: ignore[reportUnknownVariableType]
)
from .common import DatasetAdapter, JsonConversation

class SQNAAdapter(DatasetAdapter):
    """
    Adapter for handling Single Question and Answer (SQNA) datasets.

    "Single Question aNd Answer (SQNA)" is a conversation composed of exactly one user input to
    which the model (assistant) must provide exactly one answer.

    This class supports:
        - Reading a SQNA DAT file into a Huggingface `Dataset`
        - Uploading SQNA data to the Huggingface hub
        - Downloading SQNA hub datasets into DAT files
        - Converting between wide-format and JSON-format SQNA data

    Expected DAT file syntax:
        >question line
        <answer line

    Example DAT file:
        >What is 2+2?
        <4

    Corresponding dataset stored on the hub:
        [
            {
                "in": "What is 2+2?",
                "out": "4"
            }
        ]

    Corresponding JSON format:
        [{"messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]}]
    """

    def from_dat_to_wide_dataset(self, dat_filename: str) -> Dataset:
        """
        Parses a SQNA DAT file into a wide-format Huggingface Dataset.

        Args:
            dat_filename (str):
                Path to the input DAT file.

        Returns:
            Dataset:
                A Dataset with two fields: `in` and `out`.

        Raises:
            SyntaxError: If the file is malformed (e.g. unpaired question/answer).
        """
        flat_data: dict[str, list[str]] = {"in": [], "out": []}

        with open(dat_filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line.startswith(">"):
                    flat_data["in"].append(line[1:].strip())
                    if len(flat_data["in"]) - 1 != len(flat_data["out"]):
                        raise SyntaxError("Invalid syntax (missing answer)")
                elif line.startswith("<"):
                    flat_data["out"].append(line[1:].strip())
                    if len(flat_data["in"]) != len(flat_data["out"]):
                        raise SyntaxError("Invalid syntax (missing question)")
                else:
                    raise SyntaxError("Invalid syntax (line does not start with > or <)")

        if len(flat_data["in"]) != len(flat_data["out"]):
            raise SyntaxError("Invalid syntax (missing answer)")

        # Pylance: Type of from_dict() is partially unknown
        return Dataset.from_dict(flat_data) # type: ignore[reportUnknownMemberType]

    def from_dataset_to_wide_dataset(self, dataset: Dataset) -> Dataset:
        """
        Converts a structured SQNA dataset (as 2-message conversations) into a wide-format Dataset
        with `in` and `out` fields.

        Each conversation must contain exactly two messages: a user input followed by an assistant
        output.

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each item contains a list of two messages
                with roles: 'user' and 'assistant'.

        Returns:
            Dataset:
                A wide-format dataset with fields: `in` (user prompt), `out` (assistant reply).

        Raises:
            ValueError:
                If any conversation is not exactly two messages or roles are incorrect.
        """
        flat_data: dict[str, list[str]] = {"in": [], "out": []}

        for i, structured_record in enumerate(self._iter_structured_records(dataset)):
            messages = structured_record.get("messages")

            # messages is never none but done for type checker
            assert messages is not None

            if len(messages) != 2:
                raise ValueError(f"Conversation {i} must have exactly two messages")

            user_msg, assistant_msg = messages

            if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
                raise ValueError(
                    f"Conversation {i} must have roles 'user' followed by 'assistant', "
                    f"got: {user_msg['role']}, {assistant_msg['role']}"
                )

            flat_data["in"].append(user_msg["content"])
            flat_data["out"].append(assistant_msg["content"])

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
        Loads a SQNA-style dataset from the Hugging Face Hub and returns it as a split DatasetDict.

        Each split (`train`, `validation`, `test`) must exist and follow the wide-format schema
        expected by SQNA, containing one row per input/output pair with `in` and `out` fields.

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
                A dictionary containing train, validation, and test splits with SQNA-wide format.

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

        required_columns = {"in", "out"}
        for split in cast(list[str], wide_dataset.keys()):
            columns = set(wide_dataset[split].column_names)
            if not required_columns.issubset(columns):
                raise ValueError(f"Split '{split}' is missing required "
                                 f"columns: {required_columns - columns}")

        return wide_dataset

    def from_wide_dataset_to_json(self, wide_dataset: Dataset) -> JsonConversation:
        """
        Converts a wide-format SQNA dataset into JSON-style format.

        Args:
            wide_dataset (Dataset):
                Dataset with `in` and `out` fields.

        Returns:
            JsonConversation:
                A list of dicts with `messages` containing user and assistant messages.
        """
        return [
            {"messages": [
                {"role": "user", "content": record["in"]},
                {"role": "assistant", "content": record["out"]},
            ]}
            for record in self._iter_wide_records(wide_dataset)
        ]

    def from_wide_dataset_to_dat(self, wide_dataset: Dataset, dat_filename: str) -> None:
        """
        Writes a wide-format SQNA dataset to a DAT file.

        Args:
            wide_dataset (Dataset):
                Dataset with `in` and `out` fields.

            dat_filename (str):
                Output path for the DAT file.
        """
        with open(dat_filename, "w", encoding="utf-8") as f:
            for record in self._iter_wide_records(wide_dataset):
                f.write(f">{record['in']}\n")
                f.write(f"<{record['out']}\n")

    def _iter_wide_records(self, dataset: Dataset) -> Iterator[Dict[str, str]]:
        """
        Returns an iterator over a Hugging Face Dataset with each record typed as a dictionary.

        This helper function casts each item in the dataset to a `Dict[str, str]` to enable
        static type checking and clean field access (`record["in"]`, `record["out"]`), which
        are expected fields in wide-format SQNA datasets.

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each row is expected to contain
                string fields `"in"` and `"out"`.

        Returns:
            Iterator[Dict[str, str]]:
                An iterator over the dataset where each item is typed as a dictionary with string
                keys and values.
        """
        return cast(Iterator[Dict[str, str]], iter(dataset))
