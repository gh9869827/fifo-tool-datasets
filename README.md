[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Test Status](https://github.com/gh9869827/fifo-tool-datasets/actions/workflows/test.yml/badge.svg)

# `fifo-tool-datasets`

`fifo-tool-datasets` provides standardized adapters to convert plain `.dat` files into formats compatible with LLM training — including Hugging Face `datasets.Dataset` and JSON message arrays.

It supports both:

- ✅ **A Python SDK** — for structured loading and conversion  
- ✅ **A CLI** — to upload/download `.dat` files to/from the Hugging Face Hub

`.dat` files are plain-text datasets designed for LLM fine-tuning. They come in three styles:

- 💬 `sqna` (single-turn): prompt-response pairs  
- 🧠 `conversation` (multi-turn): role-tagged chat sessions  
- ⚙️ `dsl` (structured): system → input → DSL output triplets

These files are human-editable, diffable, and ideal for version control — especially during dataset development and iteration.

This tool enables a complete round-trip workflow:

1. Create and edit a `.dat` file locally  
2. Convert and upload it as a training-ready Hugging Face `datasets.Dataset`  
3. Later, download and deserialize it back into `.dat` for further edits

This gives you the best of both worlds:
- ✍️ Easy editing and version control via `.dat`  
- 🚀 Compatibility with HF pipelines using `load_dataset()`

See format examples below in each adapter section.

---

## 📚 Table of Contents

- [📐 Dataset Formats](#-dataset-formats)
- [🔁 Conversion Matrix](#-conversion-matrix)
- [📦 Installation](#-installation)
- [🚀 CLI Usage](#-cli-usage)
  - [🛠️ Command Reference](#%EF%B8%8F-command-reference)
  - [💡 Command examples](#-command-examples)
- [📦 SDK Usage](#-sdk-usage)
- [🔌 Available Adapters](#-available-adapters)
  - [🧠 `ConversationAdapter`](#-conversationadapter)
  - [💬 `SQNAAdapter`](#-sqnaadapter)
  - [⚙️ `DSLAdapter`](#%EF%B8%8F-dsladapter)
- [✅ Validation Rules](#-validation-rules)
- [🧪 Tests](#-tests)
- [✅ License](#-license)
- [📄 Disclaimer](#-disclaimer)

---

## 📐 Dataset Formats

| Format           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `.dat`           | Editable plain-text format with tags (e.g. `>`, `<`, `---`)                |
| `Dataset`        | Hugging Face `datasets.Dataset` object — used for fine-tuning              |
| `wide_dataset`   | Flattened `Dataset` with one row per message — format depends on adapter   |
| `json`           | A list of `messages` dictionaries                                           |
| `hub`            | A `DatasetDict` with `train`, `validation`, and `test` splits              |

All datasets uploaded to the Hub — if not already split — are automatically divided into `train`, `validation`, and `test` partitions using the wide format.

---

## 🔁 Conversion Matrix

| From \ To       | dataset                                               | wide_dataset                                          | dat                                                  | hub                                                              | json                                               |
|-----------------|--------------------------------------------------------|--------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------|
| **dataset**     | —                                                      | <span title="from_dataset_to_wide_dataset (direct)">✅</span> | <span title="from_dataset_to_dat (indirect)">🧩</span>     | —                                                                 | —                                                  |
| **wide_dataset**| —                                                      | —                                                      | <span title="from_wide_dataset_to_dat (direct)">✅</span>   | —                                                                 | <span title="from_wide_dataset_to_json (direct)">✅</span>  |
| **dat**         | <span title="from_dat_to_dataset (indirect)">🧩</span> | <span title="from_dat_to_wide_dataset (direct)">✅</span> | —                                                   | <span title="from_dat_to_hub (indirect)">🧩</span>                 | —                                                  |
| **hub**         | <span title="from_hub_to_dataset_dict (indirect)">🧩📦</span> | <span title="from_hub_to_dataset_wide_dict (direct)">✅📦</span> | —                                                   | —                                                                 | —                                                  |
| **json**        | —                                                      | —                                                      | —                                                   | —                                                                 | —                                                  |

**Legend**:
- ✅ **direct**: single-step conversion. Hover to view the function name.
- 🧩 **indirect**: composed of helper conversions. Hover to view the function name.
- 📦 **returns dict**: result is a `DatasetDict`.

---

## 📦 Installation

Install both the CLI and SDK in one step:

```bash
git clone https://github.com/gh9869827/fifo-tool-datasets.git

cd fifo-tool-datasets

python3 -m pip install -e .
```

This enables the `fifo-tool-datasets` command.

---

## 🚀 CLI Usage

### 🛠️ Command Reference

```bash
fifo-tool-datasets <command> [options]
```

#### `upload`

Upload a local `.dat` file or directory to the Hugging Face Hub.

```bash
fifo-tool-datasets upload <src> <dst> [--adapter <adapter>] --commit-message <msg> [--seed <int>]
```

The source must be an existing local path and the destination must be in `username/repo` format. If `--adapter` is omitted when uploading a directory, it is read from `.hf_meta.json`.

#### `download`

Download a dataset from the Hugging Face Hub.

```bash
fifo-tool-datasets download <src> <dst> [--adapter <adapter>] [-y]
```

The source must be in `username/repo` format. The destination can be a `.dat` file (merged) or a directory (one `.dat` per split). When downloading to a directory and `--adapter` is omitted, the CLI tries to read the adapter from the local `.hf_meta.json` file (if present from a previous download).

#### `push`

```bash
fifo-tool-datasets push [<dir>] --commit-message <msg> [-y]
```

Push the dataset directory to the repo specified in `.hf_meta.json`. Defaults to the current directory.

#### `pull`

```bash
fifo-tool-datasets pull [<dir>] [--adapter <adapter>] [-y]
```

Download the dataset referenced by `.hf_meta.json`. Defaults to the current directory. The adapter is read from the metadata unless overridden.

#### `split`

```bash
fifo-tool-datasets split <src> --adapter <adapter> [--to <dir>] [--split-ratio <train> <val> <test>] [-y]
```

Default split ratio is `[0.7, 0.15, 0.15]` if `--split-ratio` is omitted.

#### `merge`

Recombine split `.dat` files into a single dataset.

```bash
fifo-tool-datasets merge <dir> --adapter <adapter> [--to <file>] [-y]
```

#### `sort`

Sort the samples of a DSL `.dat` file by their full content: system prompt, user input, and assistant response. Sorting is done **in place**, meaning the original file is overwritten with the sorted result.

You can provide either a single file or a directory. If a directory is given, all `.dat` files within it will be sorted in place.

```bash
fifo-tool-datasets sort <path> [--adapter dsl]
```

Currently, only the `dsl` adapter is supported. If the `--adapter` flag is omitted, it defaults to `dsl` automatically.

#### `info`

Show record counts and metadata for a `.dat` file or split directory.

```bash
fifo-tool-datasets info <path>
```

### 🔄 Documentation and Metadata Sync

When using `upload` or `download` with a **directory source or target**, the CLI automatically:

- Upload `README.md` and `LICENSE` files from the source directory if they exist
- Download `README.md` and `LICENSE` files from the Hub if they are present in the remote repository
- Create a `.hf_meta.json` file when downloading, storing the adapter, repo ID, download timestamp, and commit hash
- Use that metadata to verify the remote commit before upload
- Auto-detect the adapter on download if `--adapter` isn't provided
- Block uploads if the remote has changed since download, unless `-y` is passed to override
- Diff local vs. remote documentation files and skip upload if content has not changed

This ensures smooth syncing of documentation while minimizing the risk of overwriting others' changes.

> ⚠️ **Note:** While this provides lightweight safety for collaborative workflows, it does **not offer Git-level guarantees**. For strict versioning, conflict resolution, or rollback, consider using `git clone` and managing pushes manually.

---

### 💡 Command examples

```bash
# Upload
fifo-tool-datasets upload dsl.dat username/my-dataset --adapter dsl --commit-message "init"

# Download (auto-detected adapter)
fifo-tool-datasets download username/my-dataset ./dsl_dir

# Download (explicit adapter override)
fifo-tool-datasets download username/my-dataset ./dsl_dir --adapter dsl

# Push updated data
fifo-tool-datasets push ./dsl_dir --commit-message "update"

# Pull latest version
fifo-tool-datasets pull ./dsl_dir

# Split
fifo-tool-datasets split dsl.dat --adapter dsl --to split_dsl

# Merge
fifo-tool-datasets merge split_dsl --adapter dsl --to full.dsl.dat

# Sort
fifo-tool-datasets sort dsl.dat --adapter dsl

# Info
fifo-tool-datasets info split_dsl
```

---

## 📦 SDK Usage

```python
from fifo_tool_datasets.sdk.hf_dataset_adapters.dsl import DSLAdapter

adapter = DSLAdapter()

# Upload to the Hugging Face Hub
adapter.from_dat_to_hub(
    "dsl.dat",
    "username/my-dataset",
    commit_message="initial upload"
)

# Download from the Hub as a DatasetDict (train/validation/test)
splits = adapter.from_hub_to_dataset_dict("username/my-dataset")

# Access splits for fine-tuning
train_dataset = splits["train"]
test_dataset = splits["test"]

# You can now use train_dataset / test_dataset to fine-tune your LLM
# e.g., with Hugging Face Transformers Trainer, SFTTrainer, etc.

# You can also directly load from a local .dat file
dataset = adapter.from_dat_to_dataset("dsl.dat")

# Convert to structured JSON format
json_records = adapter.from_wide_dataset_to_json(dataset)
```

---

## 🔌 Available Adapters

### 🧠 `ConversationAdapter`

#### `.dat`

```
---
$
You are a helpful assistant.
>
Hi
<
Hello!
---
```

#### Wide Format

```python
[
  {"id_conversation": 0, "id_message": 0, "role": "system", "content": "You are a helpful assistant."},
  {"id_conversation": 0, "id_message": 1, "role": "user",   "content": "Hi"},
  {"id_conversation": 0, "id_message": 2, "role": "assistant", "content": "Hello!"}
]
```

#### JSON Format

```python
[
  {
    "messages": [
      {"role": "system",    "content": "You are a helpful assistant."},
      {"role": "user",      "content": "Hi"},
      {"role": "assistant", "content": "Hello!"}
    ]
  }
]
```

---

### 💬 `SQNAAdapter`

#### `.dat`

```
>What is 2+2?
<4
```

#### Wide Format

```python
[
  {"in": "What is 2+2?", "out": "4"}
]
```

#### JSON Format

```python
[
  {
    "messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "4"}
    ]
  }
]
```

---

### ⚙️ `DSLAdapter`

#### `.dat`

```text
---
$ You are a precise DSL parser.
> today at 5:30PM
< SET_TIME(TODAY, 17, 30)
---
```

Multi-line entries are also supported and can be freely mixed with single-line ones.
A space after the marker on single-line entries is optional:

```text
---
$
multi-line system
prompt
> single-line input
<single-line output
---
```

To reuse the previous system prompt across multiple samples, use `...`:

```text
---
$ first prompt
> q1
< a1
---
$ ...
> q2
< a2
---
$
...
> q3
< a3
---
```

Any `$` block that contains only `...` — either directly after the `$` or on the following line — will inherit the most recent explicitly defined system prompt.

- At least one non-`...` system prompt is required in the file.
- When generating `.dat` files, consecutive identical system prompts are automatically collapsed into `$ ...`.

#### Wide Format

```python
[
  {"system": "You are a precise DSL parser.", "in": "today at 5:30PM", "out": "SET_TIME(TODAY, 17, 30)"}
]
```

#### JSON Format

```python
[
  {
    "messages": [
      {"role": "system", "content": "You are a precise DSL parser."},
      {"role": "user", "content": "today at 5:30PM"},
      {"role": "assistant", "content": "SET_TIME(TODAY, 17, 30)"}
    ]
  }
]
```

---

## ✅ Validation Rules

Each adapter enforces its own parsing rules:

- `ConversationAdapter`: tag order, message required after each tag, conversation structure
- `SQNAAdapter`: strictly `>` then `<`, per pair
- `DSLAdapter`: each block must contain `$`, `>`, `<` in this order. `$ ...` reuses the previous system prompt. Values may span multiple lines; single-line values are written with a space after the tag when generating `.dat` files. When writing `.dat` files, consecutive identical system prompts are replaced by `$ ...` automatically.

---

## 🧪 Tests

```bash
pytest tests/
```

---

## ✅ License

MIT — see [LICENSE](LICENSE)

---

## 📄 Disclaimer

This project is not affiliated with or endorsed by Hugging Face or the Python Software Foundation.  
It builds on their open-source technologies under their respective licenses.
