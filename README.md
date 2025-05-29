# `fifo-tool-datasets`

This module provides standardized adapters for loading `.dat` files into structured formats suitable for LLM training or evaluation.

All adapters inherit from the shared `DatasetAdapter` interface and implement conversion functions to/from:

- `.dat` → JSON (list of dicts)
- `.dat` → Hugging Face `Dataset`
- JSON ↔ Hugging Face `Dataset`

---

## 🔌 Available Adapters

### `ConversationAdapter`

Parses multi-turn conversations with role-based tagging.

Supports tags:
- `>` → user
- `<` → assistant
- `$` → system
- `!` → directives
- `---` → start of new conversation

### `SQNAAdapter`

Parses single-turn prompt-response pairs with simple line-based syntax.

---

## 🧩 Main Functions

The `DatasetAdapter` base class defines five core methods to convert between `.dat`, Hugging Face Hub, and JSON-style formats:

### `from_dat_to_hub(dat_filename: str, hub_dataset: str, commit_message: str) -> None`

Parses a `.dat` file and uploads the dataset to the Hugging Face Hub.

---

### `from_hub_to_json(hub_dataset: str) -> list[dict]`

Downloads a dataset from the Hugging Face Hub and returns it as a list of JSON-like dictionaries.

---

### `from_hub_to_dataset(hub_dataset: str, seed: int | None = None) -> Dataset`

Fetches a dataset from the Hugging Face Hub and returns a shuffled `datasets.Dataset` instance.

---

### `from_dat_to_dataset(filename: str, seed: int | None = None) -> Dataset`

Loads a `.dat` file and returns a shuffled Hugging Face `Dataset`.

---

### `from_dataset_to_dat(wide_dataset: Dataset, dat_filename: str) -> None`

Serializes a wide-format `Dataset` back to `.dat` format.

---

## 🧪 Format Examples

### `ConversationAdapter` 

#### Input

```
---
$
You are a helpful assistant.
>
Hello!
<
Hi there.
---
```

#### Wide Format Output (`Dataset` rows)

```python
[
  {"id_conversation": 1, "id_message": 1, "role": "system",    "content": "You are a helpful assistant."},
  {"id_conversation": 1, "id_message": 2, "role": "user",      "content": "Hello!"},
  {"id_conversation": 1, "id_message": 3, "role": "assistant", "content": "Hi there."}
]
```

#### JSON Format Output (`from_hub_to_json()`)

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

### `SQNAAdapter` 

#### Input

```
>What sound does a dog make?
<Bark
>What is 12 multiplied by 8?
<96
```

#### Wide Format Output (`Dataset` rows)

```python
[
  {"in": "What sound does a dog make?", "out": "Bark"},
  {"in": "What is 12 multiplied by 8?", "out": "96"}
]
```

#### JSON Format Output (`from_hub_to_json()`)

```python
[
  {
    "messages": [
      {"role": "user", "content": "What sound does a dog make?"},
      {"role": "assistant", "content": "Bark"}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "What is 12 multiplied by 8?"},
      {"role": "assistant", "content": "96"}
    ]
  }
]
```

---

## ✅ Validation Rules

### ConversationAdapter

- Disallows malformed tags (e.g. `> ` with trailing space)
- Disallows two identical tags in a row (`>` `>`, `<` `<`)
- Disallows unclosed conversations
- Requires non-empty content for each tag
- Requires that each conversation starts with `---`

### SQNAAdapter

- Disallows odd number of lines
- Raises `SyntaxError` for missing questions or answers

---

## 🧪 Tests

All adapters are fully covered with unit tests under `tests/`:

- Valid `.dat` fixtures
- Broken edge case `.dat` fixtures
- Assertions on both structure and error handling

---

## ✅ License

MIT — see [LICENSE](LICENSE) for details.
