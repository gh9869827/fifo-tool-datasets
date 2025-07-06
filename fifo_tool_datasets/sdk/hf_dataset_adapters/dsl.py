from typing import Dict, Iterator, TextIO, cast
from datasets import (  # type: ignore
    Dataset,
    DatasetDict,
    load_dataset  # type: ignore[reportUnknownVariableType]
)
from .common import DatasetAdapter, JsonConversation

class DSLAdapter(DatasetAdapter):
    """
    Adapter for handling datasets used to fine-tune models on DSL (Domain-Specific Language)
    generation tasks.

    This format assumes:
      - One system prompt per sample
      - One user input per sample
      - One DSL output per sample

    Expected compact `.dat` file format (3 lines per sample - with optional space after marker):
        ---
        $ <system_prompt>
        ><user_input>  # no space required after marker
        < <dsl_output>
        ---

    Multi-line entries are supported and can be freely mixed with single-line entries. To write a
    multi-line value, place the marker on its own line (e.g., just `$`, `>`, or `<`), followed by
    the content block:

        ---
        $
        <system_prompt line 1>
        <system_prompt line 2>
        >
        <user_input line 1>
        <user_input line 2>
        < <dsl_output> # single line input
        ---

    Each block (`$`, `>`, `<`) supports multi-line values using this style. The parser automatically
    detects and parses both formats.

    To avoid repeating the same system prompt across many samples, a `$` section
    may contain only `...`. This placeholder indicates that the system prompt is
    identical to the previous explicit one (on the same line as `$ ...` or on a
    separate line after `$`). At least one explicit system prompt must appear
    before any `...` is used. When writing a dataset back to `.dat`, consecutive
    identical system prompts are automatically replaced with `$ ...`.

    Wide-format dataset fields:
        - system (str): system prompt (can be reused or unique)
        - in (str): user input string
        - out (str): expected DSL output string

    Example `.dat` file:
        ---
        $ You are a precise DSL parser.
        > set alarm tomorrow at 7am
        < SET_ALARM(TOMORROW, 7, 0)
        ---

    Corresponding dataset (wide format):
        [
            {
                "system": "You are a precise DSL parser.",
                "in": "set alarm tomorrow at 7am",
                "out": "SET_ALARM(TOMORROW, 7, 0)"
            }
        ]

    JSON format:
        [
            {
                "messages": [
                    {"role": "system", "content": "You are a precise DSL parser."},
                    {"role": "user", "content": "set alarm tomorrow at 7am"},
                    {"role": "assistant", "content": "SET_ALARM(TOMORROW, 7, 0)"}
                ]
            }
        ]
    """

    def from_dat_to_wide_dataset(self, dat_filename: str) -> Dataset:
        """
        Parses a DSL DAT file into a wide-format Huggingface Dataset.

        Args:
            dat_filename (str):
                Path to the input DAT file.

        Returns:
            Dataset:
                A Dataset with three fields: `system`, `in` and `out`.

        Raises:
            SyntaxError: If the file is malformed (e.g. unpaired question/answer).
        """
        flat_data: dict[str, list[str]] = {"system": [], "in": [], "out": []}

        with open(dat_filename, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]

        if not lines:
            raise SyntaxError("The file is empty.")

        if lines[0] != "---":
            raise SyntaxError("The file must start with '---'.")

        expected_tags = ["$", ">", "<"]
        tag_idx = 0
        current_tag: str | None = None
        content_lines: list[str] = []
        tag_values: list[str | None] = [None, None, None]
        previous_system: str | None = None

        def finalize_tag(tag: str,
                         line_no: int,
                         tag_idx: int,
                         previous_system: str | None
                         ) -> tuple[int, str | None]:
            if not content_lines or all(x == "" for x in content_lines):
                raise SyntaxError(f"Empty tag '{tag}' detected at line {line_no}.")

            value = "\n".join(content_lines)

            if tag == "$":
                if value.strip() == "...":
                    if previous_system is None:
                        raise SyntaxError("System prompt placeholder '...' without "
                                          f"preceding system at line {line_no}."
                        )
                    value = previous_system
                else:
                    previous_system = value

            tag_values[tag_idx] = value
            tag_idx += 1
            content_lines.clear()
            return tag_idx, previous_system

        for line_number, line in enumerate(lines[1:], start=2):
            if tag_idx == 3 or line == "---":
                if current_tag is not None:
                    tag_idx, previous_system = \
                        finalize_tag(current_tag, line_number, tag_idx, previous_system)
                    current_tag = None
                if line != "---":
                    raise SyntaxError(f"Missing '---' block delimiter at line {line_number}.")
                if tag_idx != 3:
                    raise SyntaxError("Each DSL sample must contain $, > and < in order "
                                      f"at line {line_number}.")
                for _idx, _tag in enumerate(["system", "in", "out"]):
                    flat_data[_tag].append(cast(str, tag_values[_idx]))
                tag_values[:] = [None, None, None]
                tag_idx = 0
                continue

            if line.startswith(("$", ">", "<")):
                if current_tag is not None:
                    tag_idx, previous_system = \
                        finalize_tag(current_tag, line_number, tag_idx, previous_system)
                    current_tag = None
                tag_char = line[0]
                if tag_char != expected_tags[tag_idx]:
                    role = ["system", "input", "output"][tag_idx]
                    raise SyntaxError(
                        f"Expected '{expected_tags[tag_idx]}' at start of {role} line in block "
                        f"at line {line_number}."
                    )
                rest = line[1:]
                if rest:
                    if rest.startswith(" "):
                        rest = rest[1:]
                    content_lines = [rest]
                    tag_idx, previous_system = \
                        finalize_tag(tag_char, line_number, tag_idx, previous_system)
                else:
                    current_tag = tag_char
                    content_lines = []
                continue

            if current_tag is None:
                role = ["system", "input", "output"][tag_idx]
                raise SyntaxError(
                    f"Expected '{expected_tags[tag_idx]}' at start of {role} line in block "
                    f"at line {line_number}."
                )

            content_lines.append(line)

        if current_tag is not None or tag_idx != 0:
            raise SyntaxError(f"DSL sample is not closed properly, last line {len(lines)}")

        if not previous_system:
            raise SyntaxError("File must contain at least one explicit"
                              "system prompt before using '...'.")

        # Pylance: Type of from_dict() is partially unknown
        return Dataset.from_dict(flat_data) # type: ignore[reportUnknownMemberType]

    def from_dataset_to_wide_dataset(self, dataset: Dataset) -> Dataset:
        """
        Converts a structured DSL dataset (as 3-message conversations) into a wide-format Dataset
        with `system`, `in` and `out` fields.

        Each conversation must contain exactly three messages: a system prompt, a user input (the
        text to be converted into a DSL expression) and an assistant output (the parsed DSL
        expression).

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each item contains a list of two messages
                with roles: 'user' and 'assistant'.

        Returns:
            Dataset:
                A wide-format dataset with fields: `system`, `in` (user prompt), `out` (assistant
                reply).

        Raises:
            ValueError:
                If any conversation is not exactly three messages or roles are incorrect.
        """
        flat_data: dict[str, list[str]] = {"system": [], "in": [], "out": []}

        for i, structured_record in enumerate(self._iter_structured_records(dataset)):
            messages = structured_record.get("messages")
            assert messages is not None and len(messages) == 3

            roles = [msg["role"] for msg in messages]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"Record {i} must contain roles system, user, assistant in order")

            flat_data["system"].append(messages[0]["content"])
            flat_data["in"].append(messages[1]["content"])
            flat_data["out"].append(messages[2]["content"])

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
        expected by DSL, containing one row per system/input/output pair with `system`, `in` and
        `out` fields.

        Args:
            hub_dataset (str):
                The Hugging Face dataset identifier (e.g., "username/dataset").

        Keyword Args:
            revision (str | None):
                Git revision to download. If ``None``, the latest commit on the
                dataset's default branch is used.
            cache_dir (str | None):
                Location to store downloaded files. Uses the default HF cache if
                omitted.

        Returns:
            DatasetDict:
                A dictionary containing train, validation, and test splits with DSL-wide format.

        Raises:
            ValueError:
                If the dataset is not split, required splits are missing, or expected fields are
                absent.
        """
        wide_dataset = load_dataset(
            hub_dataset,
            revision=revision,
            cache_dir=cache_dir,
        )

        if not isinstance(wide_dataset, DatasetDict):
            raise ValueError("Expected a split DatasetDict, but got a flat Dataset.")

        required_columns = {"system", "in", "out"}
        for split in cast(list[str], wide_dataset.keys()):
            columns = set(wide_dataset[split].column_names)
            if not required_columns.issubset(columns):
                raise ValueError(f"Split '{split}' is missing required "
                                 f"columns: {required_columns - columns}")

        return wide_dataset

    def from_wide_dataset_to_json(self, wide_dataset: Dataset) -> JsonConversation:
        """
        Converts a wide-format DSL dataset into JSON-style format.

        Args:
            wide_dataset (Dataset):
                Dataset with `system`, `in` and `out` fields.

        Returns:
            JsonConversation:
                A list of dicts with `messages` containing system, user and assistant messages.
        """
        return [
            {"messages": [
                {"role": "system", "content": record["system"]},
                {"role": "user", "content": record["in"]},
                {"role": "assistant", "content": record["out"]},
            ]}
            for record in self._iter_wide_records(wide_dataset)
        ]

    def from_wide_dataset_to_dat(self, wide_dataset: Dataset, dat_filename: str) -> None:
        """
        Writes a wide-format DSL dataset to a DAT file.

        Args:
            wide_dataset (Dataset):
                Dataset with `system`, `in` and `out` fields.

            dat_filename (str):
                Output path for the DAT file.

        Consecutive rows with the same system prompt are collapsed using
        `$ ...` to avoid repetition.
        """
        def write_section(fh: TextIO, tag: str, text: str) -> None:
            if "\n" in text:
                fh.write(f"{tag}\n{text}\n")
            else:
                fh.write(f"{tag} {text}\n")

        with open(dat_filename, "w", encoding="utf-8") as f:
            f.write("---\n")
            previous_system: str | None = None
            for record in self._iter_wide_records(wide_dataset):
                system_prompt = record["system"]
                if previous_system is not None and system_prompt == previous_system:
                    write_section(f, "$", "...")
                else:
                    write_section(f, "$", system_prompt)
                    previous_system = system_prompt
                write_section(f, ">", record["in"])
                write_section(f, "<", record["out"])
                f.write("---\n")

    def _iter_wide_records(self, dataset: Dataset) -> Iterator[Dict[str, str]]:
        """
        Returns an iterator over a Hugging Face Dataset with each record typed as a dictionary.

        This helper function casts each item in the dataset to a `Dict[str, str]` to enable
        static type checking and clean field access (`record["system"]`, `record["in"]`, 
        `record["out"]`), which are expected fields in wide-format DSL datasets.

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each row is expected to contain
                string fields `"system"`, `"in"` and `"out"`.

        Returns:
            Iterator[Dict[str, str]]:
                An iterator over the dataset where each item is typed as a dictionary with string
                keys and values.
        """
        return cast(Iterator[Dict[str, str]], iter(dataset))

    def sort_dat_file(self, dat_filename: str) -> None:
        """
        Sort a DSL `.dat` file in place by system prompt, input, and output.

        Args:
            dat_filename (str):
                Path to the `.dat` file to sort. The file is read, parsed into
                a wide-format dataset, sorted lexicographically by `system`,
                `in`, and `out` fields, and written back to the same location.
        """
        dataset = self.from_dat_to_wide_dataset(dat_filename)
        sorted_dataset = dataset.sort(["system", "in", "out"])
        self.from_wide_dataset_to_dat(sorted_dataset, dat_filename)
