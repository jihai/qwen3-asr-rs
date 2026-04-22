# Security Todo

Status date: 2026-04-21

- [x] P1: Block path traversal in Hub cache/download paths (`SEC-01`, `SEC-02`)
  - Implemented strict `model_id` validation (`cache_leaf_from_model_id`).
  - Implemented safe relative-path join for shard filenames (`join_safe_relative`).
  - Applied validation in both download path (`src/hub.rs`) and local shard load path (`src/inference.rs`).
  - Added regression tests:
    - `reject_model_id_traversal_before_mutating_filesystem` (hub)
    - `test_load_weights_rejects_traversal_in_shard_filename` (inference)
- [ ] P1: Add centralized config/runtime validation to prevent panic DoS (`SEC-03`, `SEC-04`)
- [ ] P2: Remove or harden `unsafe impl Send` invariants (`SEC-05`)
- [ ] P2: Add hub download limits/timeouts (`SEC-06`)
- [ ] P3: Remove debug NaN panic edge (`SEC-07`)
- [ ] P3: Track warning-level dependency hygiene (`SEC-09`)

## Exploit Verification Example (Old vs Fixed)

This reproduces the `model_id` traversal issue (`SEC-01`).

### Vulnerable behavior (old code)

Old code used:

```rust
let model_dir = cache_dir.join(model_id.replace('/', "--"));
```

With `MODEL_ID=..`, that resolves to `cache_dir/..` (parent directory).  
The old cleanup branch could run `remove_dir_all` on that parent path.

Repro flow:

```bash
mkdir -p /tmp/qwen-poc/cache
echo "keep" > /tmp/qwen-poc/sentinel.txt

# On vulnerable revision:
MODEL_ID=.. CACHE_DIR=/tmp/qwen-poc/cache \
  cargo +stable run --example transcribe --features hub -- tests/fixtures/audio/sample1.wav || true

test -f /tmp/qwen-poc/sentinel.txt && echo "sentinel exists" || echo "sentinel deleted (vulnerable)"
```

### Fixed behavior (current code)

The same command now fails fast with `invalid model_id`, and no parent files are touched:

```bash
MODEL_ID=.. CACHE_DIR=/tmp/qwen-poc/cache \
  cargo +stable run --example transcribe --features hub -- tests/fixtures/audio/sample1.wav || true

test -f /tmp/qwen-poc/sentinel.txt && echo "sentinel exists (expected)"
```

You can also run the added regression test directly:

```bash
cargo +stable test --features hub reject_model_id_traversal_before_mutating_filesystem
```
