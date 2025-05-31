[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

# `fifo-tool-datasets`

`fifo-tool-datasets` provides standardized adapters to convert `.dat` files into formats compatible with LLM training — including Hugging Face `datasets.Dataset` and JSON message structures.

It supports both:

- ✅ **A Python SDK** — for structured loading and conversion  
- ✅ **A CLI** — to upload/download `.dat` files to/from the Hugging Face Hub

`.dat` files are plain-text datasets designed for LLM training. They come in three styles:

- 💬 `sqna` (single-turn): prompt-response pairs  
- 🧠 `conversation` (multi-turn): role-tagged chat sessions  
- ⚙️ `dsl` (structured): system → input → DSL output triplets

See format examples below in each adapter section.

---

## 📚 Table of Contents

- [📐 Dataset Formats](#-dataset-formats)
- [🔁 Conversion Matrix](#-conversion-matrix)
- [📦 Installation](#-installation)
- [🚀 CLI Usage](#-cli-usage)
  - [🛠️ Command Reference](#%EF%B8%8F-command-reference)
  - [💡 Command examples](#%EF%B8%8F-command-examples)
- [📦 SDK Usage](#-sdk-usage)
- [🔌 Available Adapters](#-available-adapters)
  - [🧠 `ConversationAdapter`](#-conversationadapter)
  - [💬 `SQNAAdapter`](#-sqnaadapter)
  - [⚙️ `DSLAdapter`](#%EF%B8%8F-dsladapter)
- [✅ Validation Rules](#-validation-rules)
- [🧪 Tests](#-tests)
- [✅ License](#-license)

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

| From \ To       | dataset                                    | wide_dataset                                  | dat                                              | hub                                                     | json                                          |
|-----------------|---------------------------------------------|-----------------------------------------------|--------------------------------------------------|----------------------------------------------------------|-----------------------------------------------|
| **dataset**     | —                                           | ✅ `from_dataset_to_wide_dataset`<br>*(direct)* | ✅ `from_dataset_to_dat`<br>*(indirect)*         | —                                                        | —                                             |
| **wide_dataset**| —                                           | —                                             | ✅ `from_wide_dataset_to_dat`<br>*(direct)*      | —                                                        | ✅ `from_wide_dataset_to_json`<br>*(direct)*  |
| **dat**         | ✅ `from_dat_to_dataset`<br>*(indirect)*    | ✅ `from_dat_to_wide_dataset`<br>*(direct)*    | —                                                | ✅ `from_dat_to_hub`<br>*(indirect)*                     | —                                             |
| **hub**         | ✅ `from_hub_to_dataset_dict`<br>*(indirect, returns dict)* | ✅ `from_hub_to_dataset_wide_dict`<br>*(direct, returns dict)* | —                                                | —                                                        | —                                             |
| **json**        | —                                           | —                                             | —                                                | —                                                        | —                                             |

**Legend**:
- ✅ **direct**: single-step conversion  
- ✅ **indirect**: composed of helper conversions  
- **(returns dict)**: result is a `DatasetDict`  

---

## 📦 Installation

Install both the CLI and SDK in one step:

```bash
python3 -m pip install -e .
```

This enables the `fifo-tool-datasets` command.

---

## 🚀 CLI Usage

### 🛠️ Command Reference

```bash
fifo-tool-datasets <command> [options]
```

#### `copy`

Upload or download datasets between `.dat` files (or directories) and the Hugging Face Hub.

```bash
fifo-tool-datasets copy <src> <dst> --adapter <adapter> [--commit-message <msg>] [--seed <int>]
```

- `.dat` or directory → hub: requires `--commit-message`
- hub → `.dat` or directory: downloads as a file (datasets are merged) or as a directory (each split is preserved)

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

---

### 💡 Command examples

```bash
# Upload
fifo-tool-datasets copy dsl.dat username/my-dataset --adapter dsl --commit-message "init"

# Download
fifo-tool-datasets copy username/my-dataset dsl.dat --adapter dsl

# Split
fifo-tool-datasets split dsl.dat --adapter dsl --to split_dsl

# Merge
fifo-tool-datasets merge split_dsl --adapter dsl --to full.dsl.dat
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

```
---
$You are a precise DSL parser.
>today at 5:30PM
<SET_TIME(TODAY, 17, 30)
---
```

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
- `DSLAdapter`: must follow `---`, `$`, `>`, `<`, `---` per block, all in single lines

---

## 🧪 Tests

```bash
pytest tests/
```

---

## ✅ License

MIT — see [LICENSE](LICENSE)
