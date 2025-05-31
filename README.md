[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)

# `fifo-tool-datasets`

`fifo-tool-datasets` provides standardized adapters to convert `.dat` files into formats compatible with LLM training — including Hugging Face `datasets.Dataset` and JSON message structures.

It supports both:

- ✅ **A Python SDK** — for structured loading and conversion  
- ✅ **A CLI** — to upload/download `.dat` files to/from the Hugging Face Hub

`.dat` files are plain-text datasets designed for LLM training. They come in two styles:

- 💬 `sqna` (single-turn): prompt-response pairs  
- 🧠 `conversation` (multi-turn): role-tagged chat sessions  

See format examples below in each adapter section.

---

## 📚 Table of Contents

- [📐 Dataset Formats](#-dataset-formats)
- [🔁 Conversion Matrix](#-conversion-matrix)
- [📦 Installation](#-installation)
- [🚀 CLI Usage](#-cli-usage)
  - [🛠️ Command Reference](#-command-reference)
  - [💡 Command examples](#️-command-examples)  
- [📦 SDK Usage](#-sdk-usage)
- [🔌 Available Adapters](#-available-adapters)
  - [🧠 `ConversationAdapter`](#-conversationadapter)
  - [💬 `SQNAAdapter`](#-sqnaadapter)
- [✅ Validation Rules](#-validation-rules)
- [🧪 Tests](#-tests)
- [✅ License](#-license)

---

## 📐 Dataset Formats

| Format           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `.dat`           | Editable plain-text format with tags (e.g. `>`, `<`, `---`)                |
| `Dataset`        | refers to the Hugging Face `datasets.Dataset` object — a table of records used for LLM fine tuning |
| `wide_dataset`   | Flattened `Dataset` with one row per message. Format is **adapter-specific** (e.g., fields like `role`, `content`, `in`, `out`, etc.) |
| `json`           | A list of `messages` dictionaries — the internal format used to build the `Dataset` |
| `hub`            | A `DatasetDict` of `Dataset` splits (`train`, `validation`, `test`) in wide format, stored on the Hugging Face Hub |

Datasets uploaded to the Hub are split into `train`, `validation`, and `test` partitions using the wide format.

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
- — not supported

---

## 📦 Installation

Install both the CLI and SDK in one step:

```bash
pip install -e .
```

This enables the `fifo-tool-datasets` command.

---

## 🚀 CLI Usage

### 🛠️ Command Reference

```bash
fifo-tool-datasets <command> [options]
```

#### `copy`

Upload/download between `.dat` files and the Hugging Face Hub.

```bash
fifo-tool-datasets copy <src> <dst> --adapter <adapter> [--commit-message <msg>] [--seed <int>]
```

- `.dat` → hub: requires `--commit-message`  
- hub → `.dat`: downloads to file

#### `split`

Split a `.dat` file into `train/validation/test`.

```bash
fifo-tool-datasets split <src> --adapter <adapter> [--to <dir>] [--split-ratio <train> <validation> <test>] [-y]
```

Default split ratio is `[0.7, 0.15, 0.15]` if `--split-ratio` is omitted.

#### `merge`

Recombine split `.dat` files into a single dataset.

```bash
fifo-tool-datasets merge <dir> --adapter <adapter> [--to <file>] [-y]
```

---

### 💡 Command examples

#### ⬆️ Upload a `.dat` file to the Hugging Face Hub

```bash
fifo-tool-datasets copy my_dataset.dat your-username/my-dataset --adapter sqna --commit-message "initial upload"
```

#### ⬇️ Download from the Hugging Face Hub and regenerate `.dat`

```bash
fifo-tool-datasets copy your-username/my-dataset my_dataset.dat --adapter conversation
```

#### ✂️ Split a `.dat` file into train/val/test

```bash
fifo-tool-datasets split my_dataset.dat --adapter conversation --to my_dataset_split
```

#### 🔁 Merge split files back into one `.dat` file

```bash
fifo-tool-datasets merge my_dataset_split --adapter conversation --to merged.dat
```

---

## 📦 SDK Usage

### 📥 Importing

```python
from fifo_tool_datasets.sdk.hf_dataset_adapters.conversation import ConversationAdapter
from fifo_tool_datasets.sdk.hf_dataset_adapters.sqna import SQNAAdapter
```

### 🧪 Example

```python
adapter = SQNAAdapter()

# .dat to Dataset
dataset = adapter.from_dat_to_dataset("data.dat")

# Upload
adapter.from_dat_to_hub("data.dat", "username/dataset", "upload msg", split_ratios=[0.6, 0.2, 0.2])

# Download and save back
splits = adapter.from_hub_to_dataset_dict("username/dataset")
adapter.from_dataset_to_dat(splits["train"], "out.dat")
```

---

## 🔌 Available Adapters

### 🧠 `ConversationAdapter`

#### 🧪 `.dat` format

```text
---
$
You are a helpful assistant.
>
Hello!
<
Hi there.
---
```

#### 🧪 Wide Format

```python
[
  {"id_conversation": 0, "id_message": 0, "role": "system",    "content": "You are a helpful assistant."},
  {"id_conversation": 0, "id_message": 1, "role": "user",      "content": "Hello!"},
  {"id_conversation": 0, "id_message": 2, "role": "assistant", "content": "Hi there."}
]
```

#### 🧪 JSON Format

```python
[
  {
    "messages": [
      {"role": "system",    "content": "You are a helpful assistant."},
      {"role": "user",      "content": "Hello!"},
      {"role": "assistant", "content": "Hi there."}
    ]
  }
]
```

---

### 💬 `SQNAAdapter`

#### 🧪 `.dat` format

```text
>What is 2+2?
<4
```

#### 🧪 Wide Format

```python
[
  {"in": "What is 2+2?", "out": "4"}
]
```

#### 🧪 JSON Format

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

## ✅ Validation Rules

### `ConversationAdapter`

- No trailing spaces after tags  
- No repeated role tags (e.g. `> >`)  
- All conversations must start with `---`  
- Conversations must be properly closed  
- Each tag must be followed by a message  

### `SQNAAdapter`

- Every question (`>`) must be followed by an answer (`<`)  
- Fails with `SyntaxError` if any question/answer is missing or misaligned  

---

## 🧪 Tests

Test suite includes:

- `.dat` → wide → structured → back round-trip  
- Detection of invalid `.dat` structure  
- Seeded shuffling consistency  
- Merge/split consistency  

```bash
pytest tests/
```

---

## ✅ License

MIT — see [LICENSE](LICENSE) for details.
