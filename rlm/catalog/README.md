# catalog

Each `.py` module in this directory that defines a module-level `DESCRIPTOR` is auto-registered as a halo dataset at server startup. The pattern is:

```python
# catalog/my_dataset.py
from pathlib import Path
from dataset import DatasetDescriptor, OutcomeSpec

DESCRIPTOR = DatasetDescriptor(
    id="my_dataset",                      # URL-safe identifier, used in routes
    name="My Dataset (nice display name)",
    source_path=Path("/data/my_traces.jsonl"),
    source_model="claude-opus-4-7",       # model whose runs produced the traces
    description="What's in this dataset, 1-2 sentences for the UI.",

    # Field mappings (dotted paths into each record).
    id_field="query_id",
    query_field="query",
    messages_field="messages",
    final_answer_field="final_answer",

    # Ground truth + outcome (set to None for datasets without these).
    ground_truth_field="expected",
    outcome_field="score",
    outcome=OutcomeSpec(
        kind="score_01",                 # or "binary" or "none"
        higher_is_better=True,
        perfect_threshold=0.999,
        zero_threshold=0.001,
        display_name="score",
    ),

    # Documents with paths (file-search datasets). Set to None if N/A.
    documents_field="documents",
    document_path_field="path",

    # Agent-metadata fields (dotted paths, same defaults as grepfruit).
    usage_field="metadata.usage",
    turns_field="metadata.turns_used",
    tool_calls_total_field="metadata.total_tool_calls",
    tool_errors_field="metadata.tool_errors",

    # Categorical labels the UI and tools expose as filters.
    label_fields=["task_type", "difficulty", "domain"],

    # Questions that populate the preset list for this dataset.
    seed_questions=[
        "...",
    ],
)
```

Once saved:

```bash
uv run halo index --dataset my_dataset    # build the summary index
uv run halo serve                         # UI picks it up at the next boot
```

## Conventions

* **One dataset per file.** File name matches `DESCRIPTOR.id`; underscores, not hyphens.
* **Leave out what doesn't apply.** Every `*_field` can be `None`. Every `label_fields` entry is optional. The indexer, tools, and UI silently omit fields that don't exist on a given dataset.
* **Start from the template.** `_template.py.example` in this directory is a starter; copy it to `<id>.py`, fill in the fields, and the registry picks it up.
* **Modules starting with `_` are skipped** by the registry (reserved for registry internals and the template).
