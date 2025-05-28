from abc import ABC, abstractmethod
# Pylance: suppress missing type stub warning for datasets
from datasets import (  # type: ignore
    Dataset
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
class DatasetAdapter(ABC):
    """
    Abstract base class for adapting various DAT formats into Huggingface-compatible datasets.
    
    Subclasses must implement the following format-specific methods:
      - _from_dat_to_wide_dataset(self, dat_filename): parses a DAT file into a wide-format dataset.
      - _load_from_hub(self, hub_dataset): loads a wide-format dataset from the Huggingface hub.
      - _from_wide_dataset_to_json(self, dataset): converts a wide-format dataset into JSON-style 
        format.

    This class provides generic methods for:
      - Uploading to the Huggingface hub from a DAT file
      - Converting a hub dataset to JSON
      - Loading and shuffling datasets from hub or DAT files
    """

    def from_dat_to_hub(self, dat_filename: str, hub_dataset: str, commit_message: str) -> None:
        """
        Uploads a dataset to the Huggingface hub from a DAT file.

        Args:
            dat_filename (str):
                Path to the DAT file.

            hub_dataset (str):
                Name of the Huggingface dataset.

            commit_message (str):
                Required commit message.

        Raises:
            ValueError: If commit_message is not provided.
        """
        if not commit_message:
            raise ValueError("Commit message is required")

        dataset = self._from_dat_to_wide_dataset(dat_filename)
        dataset.push_to_hub(hub_dataset, commit_message=commit_message)

    def from_hub_to_json(self, hub_dataset: str) -> JsonConversation:
        """
        Loads a dataset from the Huggingface hub and converts it to a JSON-style list.

        Args:
            hub_dataset (str):
                The Huggingface dataset identifier.

        Returns:
            JsonConversation:
                List of conversation dictionaries.
        """
        wide_dataset = self._load_from_hub(hub_dataset)
        return self._from_wide_dataset_to_json(wide_dataset)

    def from_hub_to_dataset(self, hub_dataset: str, seed: int | None = None) -> Dataset:
        """
        Builds a shuffled Huggingface Dataset from a JSON loader function.

        Args:
            hub_dataset (str):
                Name of the dataset to load from the Huggingface hub.

            seed (int | None):
                Optional seed for reproducible shuffling.

        Returns:
            Dataset:
                A shuffled Dataset containing the JSON-formatted conversations.
        """
        # Pylance: Type of from_list() is partially unknown
        return Dataset.from_list(  # type: ignore[reportUnknownMemberType]
            self.from_hub_to_json(hub_dataset)
        ).shuffle(seed=seed)

    def from_dat_to_dataset(self, filename: str, seed: int | None = None) -> Dataset:
        """
        Builds a shuffled Dataset from a DAT file using format-specific parsing.

        Args:
            filename (str):
                Path to the input DAT file.

            seed (int | None):
                Seed for deterministic shuffling (optional).

        Returns:
            Dataset:
                The parsed and shuffled Dataset object.
        """
        # Pylance: Type of from_list() is partially unknown
        return Dataset.from_list(  # type: ignore[reportUnknownMemberType]
            self._from_wide_dataset_to_json(
                self._from_dat_to_wide_dataset(filename)
            )
        ).shuffle(seed=seed)

    @abstractmethod
    def to_dat(self, wide_dataset: Dataset, dat_filename: str) -> None:
        """
        Writes a wide-format dataset to a DAT file.

        Args:
            wide_dataset (Dataset):
                Wide-format dataset.

            dat_filename (str):
                Output path for the DAT file.
        """

    @abstractmethod
    def _from_dat_to_wide_dataset(self, dat_filename: str) -> Dataset:
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
    def _load_from_hub(self, hub_dataset: str) -> Dataset:
        """
        Loads a wide-format dataset from the Huggingface hub.

        Args:
            hub_dataset (str):
                The Huggingface dataset identifier.

        Returns:
            Dataset:
                A wide-format dataset.
        """

    @abstractmethod
    def _from_wide_dataset_to_json(self, wide_dataset: Dataset) -> JsonConversation:
        """
        Converts a wide-format dataset into JSON-style format.

        Args:
            wide_dataset (Dataset):
                wide-format dataset.

        Returns:
            JsonConversation:
                A list of dicts with `messages` containing user and assistant messages.
        """
