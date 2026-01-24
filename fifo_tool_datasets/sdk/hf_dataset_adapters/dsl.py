from typing import Dict, Iterator, TextIO, cast
from datasets import (  # type: ignore
    Dataset,
    DatasetDict,
    load_dataset  # type: ignore[reportUnknownVariableType]
)
from .common import (
    DatasetAdapter,
    JsonConversation,
    StructureMessageRecord,
    StructuredConversationRecord
)

class DSLAdapter(DatasetAdapter):
    """
    Adapter for handling datasets used to fine-tune models on DSL (Domain-Specific Language)
    generation tasks.

    This format assumes:
      - One system prompt per sample
      - One user input per sample
      - One DSL output per sample
      - Optional reasoning (chain-of-thought) per sample

    Expected compact `.dat` file format (3-4 lines per sample - with optional space after marker):
        ---
        $ <system_prompt>
        ><user_input>  # no space required after marker
        ? <reasoning>  # optional
        < <dsl_output>
        ---

    Multi-line entries are supported and can be freely mixed with single-line entries. You can write
    a multi-line value in two ways:

    1. Place the marker on its own line (e.g., just `$`, `>`, `?`, or `<`), followed by the content
       block:
        ---
        $
        <system_prompt line 1>
        <system_prompt line 2>
        >
        <user_input line 1>
        <user_input line 2>
        ?
        <reasoning line 1>
        <reasoning line 2>
        < <dsl_output>
        ---

    2. Start content on the same line as the marker, followed by additional lines:
        ---
        $ <system line 1>
        <system line 2>
        > <user line 1>
        <user line 2>
        ? <reasoning line 1>
        <reasoning line 2>
        < <dsl_output>
        ---

    Each block (`$`, `>`, `?`, `<`) supports multi-line values using either style. The parser
    automatically detects and parses both formats. The `?` (reasoning) section is optional and
    appears between the `>` (input) and `<` (output) sections when present.

    To avoid repeating the same system prompt across many samples, a `$` section
    may contain only `...`. This placeholder indicates that the system prompt is
    identical to the previous explicit one (on the same line as `$ ...` or on a
    separate line after `$`). At least one explicit system prompt must appear
    before any `...` is used. When writing a dataset back to `.dat`, consecutive
    identical system prompts are automatically replaced with `$ ...`.

    Wide-format dataset fields:
        - system (str): system prompt (can be reused or unique)
        - in (str): user input string
        - reasoning (str): optional reasoning content (only present if at least one record has reasoning)
        - out (str): expected DSL output string

    Example `.dat` file with reasoning:
        ---
        $ You are a precise DSL parser.
        > today at 5:30PM
        ? base=TODAY
        time.hour=17
        time.minute=30
        < SET_TIME(TODAY, 17, 30)
        ---

    Example `.dat` file without reasoning:
        ---
        $ You are a precise DSL parser.
        > set alarm tomorrow at 7am
        < SET_ALARM(TOMORROW, 7, 0)
        ---

    Corresponding dataset (wide format) with reasoning:
        [
            {
                "system": "You are a precise DSL parser.",
                "in": "today at 5:30PM",
                "reasoning": "base=TODAY\ntime.hour=17\ntime.minute=30",
                "out": "SET_TIME(TODAY, 17, 30)"
            }
        ]

    Corresponding dataset (wide format) without reasoning:
        [
            {
                "system": "You are a precise DSL parser.",
                "in": "set alarm tomorrow at 7am",
                "out": "SET_ALARM(TOMORROW, 7, 0)"
            }
        ]

    JSON format with reasoning:
        [
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

    JSON format without reasoning:
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
                A Dataset with three or four fields: `system`, `in`, and `out`, plus
                `reasoning` (only present if at least one record has reasoning content).

        Raises:
            SyntaxError: If the file is malformed (e.g. unpaired question/answer).
        """
        flat_data: dict[str, list[str]] = {"system": [], "in": [], "reasoning": [], "out": []}

        with open(dat_filename, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]

        if not lines:
            raise SyntaxError("The file is empty.")

        if lines[0] != "---":
            raise SyntaxError("The file must start with '---'.")

        # Fixed indices: 0=system, 1=input, 2=reasoning, 3=output
        tag_values: list[str | None] = [None, None, None, None]
        previous_system: str | None = None

        # Convert to list for easier iteration with lookahead
        remaining_lines = list(enumerate(lines[1:], start=2))
        current_pos = 0

        def peek_line() -> tuple[int, str] | None:
            """Peek at the next line without consuming it."""
            if current_pos < len(remaining_lines):
                return remaining_lines[current_pos]
            return None

        def consume_line() -> tuple[int, str] | None:
            """Consume and return the next line."""
            nonlocal current_pos
            if current_pos < len(remaining_lines):
                line = remaining_lines[current_pos]
                current_pos += 1
                return line
            return None

        def process_tag(expected_tag: str, mandatory: bool) -> str | None:
            """
            Process a single tag section. Returns the parsed value or None if optional and not
            present. Stops when the next tag is detected (lookahead, does not consume).

            Args:
                expected_tag (str):
                    The tag character to expect ('$', '>', '?', '<')
                mandatory (bool):
                    Whether this tag must be present

            Returns:
                str | None:
                    The parsed value for this tag, or None if optional and not present
            """
            nonlocal previous_system

            peeked = peek_line()
            if peeked is None:
                if mandatory:
                    raise SyntaxError(f"Expected '{expected_tag}' but reached end of file.")
                return None

            line_number, line = peeked

            # Check if line starts with a tag
            if not line.startswith(("$", ">", "?", "<")):
                if mandatory:
                    raise SyntaxError(
                        f"Expected '{expected_tag}' at start of line {line_number}."
                    )
                return None

            tag_char = line[0]

            # Check if this is the expected tag
            if tag_char != expected_tag:
                # Currently only the '?' tag is optional
                if not mandatory:
                    # Optional tag not present
                    return None
                # For mandatory tags, this is an error
                raise SyntaxError(
                    f"Expected '{expected_tag}' but got '{tag_char}' at line {line_number}."
                )

            # Consume the tag line
            consume_line()

            # Parse the tag content
            rest = line[1:]
            if rest.startswith(" "):
                rest = rest[1:]

            content_lines = [rest] if rest else []

            # Continue reading lines until we hit the next tag or block delimiter
            while True:
                peeked = peek_line()
                if peeked is None:
                    break
                _, next_line = peeked
                if next_line == "---" or next_line.startswith(("$", ">", "?", "<")):
                    break
                consume_line()
                content_lines.append(next_line)

            # Validate and finalize content
            if not content_lines or all(x == "" for x in content_lines):
                raise SyntaxError(f"Empty tag '{tag_char}' detected at line {line_number}.")

            value = "\n".join(content_lines)

            # Handle system prompt placeholder
            if tag_char == "$":
                if value.strip() == "...":
                    if previous_system is None:
                        raise SyntaxError(
                            f"System prompt placeholder '...' without "
                            f"preceding system at line {line_number}."
                        )
                    value = previous_system
                else:
                    previous_system = value

            return value

        # Main parsing loop: process blocks
        # Note: lines[1:] starts after the first ---, so we process tags directly
        while current_pos < len(remaining_lines):
            # Process one complete block: $ > [?] <
            tag_values[0] = process_tag("$", mandatory=True)
            tag_values[1] = process_tag(">", mandatory=True)
            tag_values[2] = process_tag("?", mandatory=False)
            tag_values[3] = process_tag("<", mandatory=True)

            # Note: The following check is defensive and should never trigger since
            # process_tag raises errors for missing mandatory tags. However, it's kept
            # for additional safety in case of future code changes.
            if tag_values[0] is None or tag_values[1] is None or tag_values[3] is None:
                peeked = peek_line()
                line_num = peeked[0] if peeked else len(lines)
                raise SyntaxError("Each DSL sample must contain $, > and < in order "
                                  f"at line {line_num}.")

            # Store the data (fixed indices: 0=system, 1=input, 2=reasoning, 3=output)
            flat_data["system"].append(tag_values[0])
            flat_data["in"].append(tag_values[1])
            flat_data["reasoning"].append(tag_values[2] if tag_values[2] is not None else "")
            flat_data["out"].append(tag_values[3])

            # Reset for next block
            tag_values[:] = [None, None, None, None]

            # Check for closing ---
            peeked = peek_line()
            if peeked is None:
                raise SyntaxError(f"DSL sample is not closed properly, last line {len(lines)}")

            closing_line_number, closing_line = peeked
            if closing_line != "---":
                raise SyntaxError(
                    f"Expected closing '---' but got '{closing_line}' "
                    f"at line {closing_line_number}."
                )
            # Consume the closing ---
            consume_line()

            # Check if there's more content (another block)
            peeked_next = peek_line()
            if peeked_next is None:
                # End of file
                break

        if not previous_system:
            raise SyntaxError("File must contain at least one explicit "
                              "system prompt before using '...'.")

        # Check if all reasoning values are empty, and if so, drop the column
        if all(r == "" for r in flat_data["reasoning"]):
            del flat_data["reasoning"]

        # Pylance: Type of from_dict() is partially unknown
        return Dataset.from_dict(flat_data) # type: ignore[reportUnknownMemberType]

    def from_dataset_to_wide_dataset(self, dataset: Dataset) -> Dataset:
        """
        Converts a structured DSL dataset (as 3-message conversations) into a wide-format Dataset
        with `system`, `in`, and `out` fields, plus `reasoning` if at least one record has it.

        Each conversation must contain exactly three messages: a system prompt, a user input (the
        text to be converted into a DSL expression) and an assistant output (the parsed DSL
        expression). The reasoning field is extracted from assistant.metadata.reasoning if present.

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each item contains a list of three messages
                with roles: 'system', 'user', and 'assistant'.

        Returns:
            Dataset:
                A wide-format dataset with fields: `system`, `in`, and `out`, plus `reasoning`
                (only present if at least one record has reasoning content).

        Raises:
            ValueError:
                If any conversation is not exactly three messages or roles are incorrect.
        """
        flat_data: dict[str, list[str]] = {"system": [], "in": [], "reasoning": [], "out": []}

        for i, structured_record in enumerate(self._iter_structured_records(dataset)):
            messages = structured_record.get("messages")
            assert messages is not None and len(messages) == 3

            roles = [msg["role"] for msg in messages]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"Record {i} must contain roles system, user, assistant in order")

            flat_data["system"].append(messages[0]["content"])
            flat_data["in"].append(messages[1]["content"])

            # Extract reasoning from assistant message metadata if present.
            # Note: due to Arrow schema unification, `metadata` may be `None` when absent.
            assistant_metadata = messages[2].get("metadata") or {}
            reasoning_content = assistant_metadata.get("reasoning", "")
            flat_data["reasoning"].append(reasoning_content)

            flat_data["out"].append(messages[2]["content"])

        # Check if all reasoning values are empty, and if so, drop the column
        if all(r == "" for r in flat_data["reasoning"]):
            del flat_data["reasoning"]

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
                Git revision to download. If `None`, the latest commit on the
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
            cache_dir=cache_dir
        )

        if not isinstance(wide_dataset, DatasetDict):
            raise ValueError("Expected a split DatasetDict, but got a flat Dataset.")

        required_columns = {"system", "in", "out"}
        for split in cast(list[str], wide_dataset.keys()):
            columns = set(wide_dataset[split].column_names)
            if not required_columns.issubset(columns):
                raise ValueError(f"Split '{split}' is missing required "
                                 f"columns: {required_columns - columns}")

            # Add reasoning column if missing (for backward compatibility with datasets
            # uploaded to the Hub before the reasoning column optimization was implemented)
            # After adding it, check if all values are empty and drop it if so to maintain
            # a consistent compact layout regardless of whether the dataset is old or new
            if "reasoning" not in columns:
                # Create a new column with empty strings
                split_dataset = wide_dataset[split]
                reasoning_values = [""] * len(split_dataset)
                # Pylance: Type of add_column() is partially unknown
                wide_dataset[split] = split_dataset.add_column( # type: ignore[reportUnknownMemberType] # pylint: disable=line-too-long
                    "reasoning", reasoning_values
                )

            # Check if all reasoning values are empty and drop the column to keep layout compact.
            # This optimization applies to both old datasets (after adding the column above) and
            # new datasets that may already have an empty reasoning column.
            split_dataset = wide_dataset[split]
            if "reasoning" in split_dataset.column_names:
                reasoning_values = split_dataset["reasoning"]
                if all(r == "" for r in reasoning_values):
                    # Remove the column using remove_columns
                    wide_dataset[split] = split_dataset.remove_columns(["reasoning"])

        return wide_dataset

    def from_wide_dataset_to_json(self, wide_dataset: Dataset) -> JsonConversation:
        """
        Converts a wide-format DSL dataset into JSON-style format.

        Args:
            wide_dataset (Dataset):
                Dataset with `system`, `in`, and `out` fields. May optionally include
                `reasoning` field if at least one record has reasoning content.

        Returns:
            JsonConversation:
                A list of dicts with `messages` containing system, user and assistant messages.
                Reasoning, if present, is stored in the assistant message's `metadata` field.
        """
        result : list[StructuredConversationRecord] = []
        for record in self._iter_wide_records(wide_dataset):
            # Build assistant message
            assistant_msg : StructureMessageRecord = {
                "role": "assistant",
                "content": record["out"]
            }

            # Add reasoning to metadata if present and non-empty
            reasoning_content = record.get("reasoning", "")
            if reasoning_content:
                assistant_msg["metadata"] = {"reasoning": reasoning_content}

            conversation_record : StructuredConversationRecord = {
                "messages": [
                    {"role": "system", "content": record["system"]},
                    {"role": "user", "content": record["in"]},
                    assistant_msg
                ]
            }

            result.append(conversation_record)

        return result

    def from_wide_dataset_to_dat(self, wide_dataset: Dataset, dat_filename: str) -> None:
        """
        Writes a wide-format DSL dataset to a DAT file.

        Args:
            wide_dataset (Dataset):
                Dataset with `system`, `in`, and `out` fields. May optionally include
                `reasoning` field if at least one record has reasoning content.

            dat_filename (str):
                Output path for the DAT file.

        Consecutive rows with the same system prompt are collapsed using
        `$ ...` to avoid repetition. Reasoning is written as a `?` section
        between input and output if present and non-empty.
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

                # Write reasoning if present and non-empty
                reasoning = record.get("reasoning", "")
                if reasoning:
                    write_section(f, "?", reasoning)

                write_section(f, "<", record["out"])
                f.write("---\n")

    def _iter_wide_records(self, dataset: Dataset) -> Iterator[Dict[str, str]]:
        """
        Returns an iterator over a Hugging Face Dataset with each record typed as a dictionary.

        This helper function casts each item in the dataset to a `Dict[str, str]` to enable
        static type checking and clean field access (`record["system"]`, `record["in"]`,
        `record["out"]`, and optionally `record.get("reasoning", "")`). The `reasoning` field
        may or may not be present in the dataset depending on whether any record contains reasoning.

        Args:
            dataset (Dataset):
                A Hugging Face Dataset where each row is expected to contain
                string fields `"system"`, `"in"`, and `"out"`. May optionally include
                `"reasoning"` field if at least one record has reasoning content.

        Returns:
            Iterator[Dict[str, str]]:
                An iterator over the dataset where each item is typed as a dictionary with string
                keys and values.
        """
        return cast(Iterator[Dict[str, str]], iter(dataset))

    def sort_dat_file(self, dat_filename: str) -> None:
        """
        Sort a DSL `.dat` file in place by system prompt, input, reasoning, and output.

        Args:
            dat_filename (str):
                Path to the `.dat` file to sort. The file is read, parsed into
                a wide-format dataset, sorted lexicographically by `system`,
                `in`, `reasoning`, and `out` fields, and written back to the same location.
        """
        dataset = self.from_dat_to_wide_dataset(dat_filename)
        sort_keys = ["system", "in"]
        if "reasoning" in dataset.column_names:
            sort_keys.append("reasoning")
        sort_keys.append("out")
        sorted_dataset = dataset.sort(sort_keys)
        self.from_wide_dataset_to_dat(sorted_dataset, dat_filename)
