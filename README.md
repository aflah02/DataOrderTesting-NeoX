# Preserve Data Order Smoke Test

This directory contains a small end-to-end smoke test for
`preserve_data_order`. It creates a three-document mmap dataset and loads it
through the GPT-NeoX dataset builder twice. The first load enables the flag and
must preserve document and sample order across document and epoch boundaries.
The second load omits the flag and must shuffle both mappings and the emitted
samples.

## Run

From the repository root:

```bash
docker run --rm --ipc=host --user 0:0 \
  -e PYTHONPATH=/review \
  -v "$PWD:/review:ro" \
  -v "$PWD/logs/preserve_data_order_smoke:/review/logs/preserve_data_order_smoke:rw" \
  -w /review \
  gpt-neox:te \
  python logs/preserve_data_order_smoke/run_smoke.py
```

A successful run ends with:

```text
PASS: flag preserves order and the default path shuffles it
```

The ordered mappings are expected to be:

```text
doc_idx=[0, 1, 2, 0, 1, 2]
shuffle_idx=[0, 1, 2, 3, 4, 5, 6]
```

The default mappings are expected to differ from these identity mappings.
Running the command again also verifies that the persisted ordered and default
index maps load successfully.

## Files

- `run_smoke.py`: creates the corpus and compares ordered and default loads.
- `dummy_text_document.bin` and `.idx`: generated mmap dataset files.
- `*_ordered_*.npy`: generated ordered document, sample, and shuffle maps.
- Other `*_indexmap_*.npy` files: generated default shuffled maps.

This directory is under `logs/` and is ignored by Git.
