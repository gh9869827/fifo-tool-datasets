from typing import Dict, Iterator, cast
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset,
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

    def _from_dat_to_wide_dataset(self, dat_filename: str) -> Dataset:
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
        messages: dict[str, list[str]] = {"in": [], "out": []}

        with open(dat_filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line.startswith(">"):
                    messages["in"].append(line[1:].strip())
                    if len(messages["in"]) - 1 != len(messages["out"]):
                        raise SyntaxError("Invalid syntax (missing answer)")
                elif line.startswith("<"):
                    messages["out"].append(line[1:].strip())
                    if len(messages["in"]) != len(messages["out"]):
                        raise SyntaxError("Invalid syntax (missing question)")
                else:
                    raise SyntaxError("Invalid syntax (line does not start with > or <)")

        # Pylance: Type of from_dict() is partially unknown
        return Dataset.from_dict(messages) # type: ignore[reportUnknownMemberType]

    def _load_from_hub(self, hub_dataset: str) -> Dataset:
        """
        Loads a SQNA dataset from the Huggingface hub.

        Args:
            hub_dataset (str):
                The Huggingface dataset identifier.

        Returns:
            Dataset:
                A dataset containing `in` and `out` fields.
        """
        # Explicitly loading a split returns a Dataset, not a DatasetDict - safe to cast
        return cast(Dataset, load_dataset(hub_dataset, split="train"))

    def _from_wide_dataset_to_json(self, wide_dataset: Dataset) -> JsonConversation:
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
            for record in self._iter_records(wide_dataset)
        ]

    def to_dat(self, wide_dataset: Dataset, dat_filename: str) -> None:
        """
        Writes a wide-format SQNA dataset to a DAT file.

        Args:
            wide_dataset (Dataset):
                Dataset with `in` and `out` fields.

            dat_filename (str):
                Output path for the DAT file.
        """
        with open(dat_filename, "w", encoding="utf-8") as f:
            for record in self._iter_records(wide_dataset):
                f.write(f">{record['in']}\n")
                f.write(f"<{record['out']}\n")

    def _iter_records(self, dataset: Dataset) -> Iterator[Dict[str, str]]:
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
