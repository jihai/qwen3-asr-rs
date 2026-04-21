# Security Review Report: `qwen3-asr-rs`

Date: 2026-04-21  
Reviewer: Codex (manual static review + local tooling pass)

## Scope

Reviewed the full repository contents:

- `Cargo.toml`, `Cargo.lock`, `README.md`, `.gitignore`
- all Rust sources under `src/` and `src/bin/`
- all examples under `examples/`
- all tests under `tests/`

## Executive Summary

The repository has **two high-severity filesystem traversal issues** in model download/loading paths that can lead to arbitrary file overwrite/read and potentially destructive directory deletion when untrusted model identifiers or model index data are used.

It previously had a **confirmed high-severity dependency exposure** (RustSec advisories in `rustls-webpki`) in the `hub` feature path, which is now remediated in lockfile by upgrading to `rustls-webpki 0.103.12`.

Additional medium/low issues are mostly availability and hardening concerns (panic-driven DoS, unbounded network behavior, and an `unsafe impl Send` that requires stronger safety guarantees).

## Methodology

- Full manual code review with line-level tracing of input and filesystem/network boundaries.
- Pattern scans for high-risk primitives (`unsafe`, filesystem/network ops, panic sites, env usage).
- Attempted dependency/tool checks:
  - `cargo audit` unavailable in environment (`no such command: audit`).
  - `cargo check --all-targets` and `cargo tree --locked` failed because the local Cargo binary cannot parse lockfile v4.

## Tooling Update (User-Run, 2026-04-21)

User provided additional command outputs after upgrading toolchain:

- `cargo +stable --version` → `cargo 1.95.0` (lockfile v4 compatible)
- `cargo +stable check --all-targets --features hub` → success
- `cargo +stable check --all-targets --no-default-features --features hub` → success
- `cargo +stable audit` (after installing `cargo-audit`) initially → **3 vulnerabilities found, 2 warnings**
- `cargo +stable update -p rustls-webpki --precise 0.103.12` applied
- `cargo +stable audit` after update → **0 vulnerabilities, 2 warnings**
- `cargo +stable check --all-targets --all-features` still fails on host without CUDA toolkit (`nvcc` missing), which is expected for `cuda` feature builds.
- `cargo +stable tree -i rustls-webpki` without `--features hub` returned no match (expected when optional `hub` feature graph is not selected).

## Findings

### SEC-01 (High): Path traversal via `model_id` enables destructive filesystem access

- **Affected code:** `src/hub.rs:67`, `src/hub.rs:68`, `src/hub.rs:80`, `src/hub.rs:84`
- **Issue:** `model_id` is sanitized only by replacing `/` with `--`:
  - `let sanitized = model_id.replace('/', "--");`
  - This still permits values like `..` (or Windows-style separators like `\` on Windows).
  - `cache_dir.join(&sanitized)` can escape the intended cache root.
  - If `.complete` is missing, `remove_dir_all(&model_dir)` can recursively delete unintended directories.
- **Impact:** Arbitrary directory deletion and writes outside cache root (high-impact local data destruction risk).
- **Exploit example:** `from_pretrained("..", Path::new("models"), ...)` can resolve to parent paths and trigger cleanup logic on unintended directories.
- **Recommendation:**
  - Strictly validate `model_id` against an allowlist pattern.
  - Reject path separators (`/`, `\`), `.`/`..`, and absolute paths.
  - Canonicalize candidate path and enforce `candidate.starts_with(canonical_cache_dir)` before any delete/write.

### SEC-02 (High): Shard filename path traversal from model index allows arbitrary file write/read

- **Affected code:** `src/hub.rs:96`, `src/hub.rs:100`, `src/hub.rs:105`, `src/inference.rs:552`, `src/inference.rs:566`
- **Issue:** shard names are taken from `model.safetensors.index.json` and joined directly:
  - download path: `model_dir.join(shard)`
  - load path: `model_dir.join(&shard)`
  - No validation against `..`, absolute paths, or platform-specific separators.
- **Impact:** Untrusted model metadata can write files outside the model directory and later read arbitrary filesystem paths via shard loading.
- **Recommendation:**
  - Accept only safe relative filenames (prefer `file_name`-only policy).
  - Reject absolute paths and any path containing parent traversal.
  - Canonicalize joined path and enforce it remains under canonical `model_dir`.

### SEC-03 (Medium): Unvalidated model config can trigger panic/DoS via divide-by-zero and invalid indexing

- **Affected code:** `src/encoder.rs:65`, `src/encoder.rs:396`, `src/encoder.rs:428`, `src/encoder.rs:438`, `src/decoder.rs:84`, `src/decoder.rs:237`
- **Issue:** several arithmetic/index operations assume valid model config values:
  - `d_model / num_heads`
  - `n_window_infer / chunk_size` and later division by `chunks_per_window`
  - `nqh / nkvh`
  - `sections.len() - 1` for empty `mrope_section`
- **Impact:** malicious or malformed `config.json` can crash process (panic), causing denial of service.
- **Recommendation:**
  - Add `AsrConfig::validate()` with explicit invariants (non-zero heads, valid window relationships, non-empty rope sections, etc.).
  - Fail with structured errors instead of panicking.

### SEC-04 (Medium): Input validation gaps allow panic on empty audio / invalid streaming chunk size

- **Affected code:** `src/mel.rs:19`, `src/mel.rs:165`, `src/streaming.rs:123`, `src/streaming.rs:303`
- **Issue:**
  - `reflection_pad` computes `n - 1`; with empty input (`n == 0`) this underflows/panics.
  - `chunk_size_sec` is cast to `usize` without validation; zero/NaN/negative can produce invalid chunk sizing behavior.
- **Impact:** DoS risk if upstream service passes untrusted/edge-case audio or unvalidated streaming options.
- **Recommendation:**
  - Reject empty samples early (`Err`).
  - Validate `chunk_size_sec.is_finite() && chunk_size_sec > 0.0`.
  - Add upper bounds for chunk size and explicit errors for invalid option values.

### SEC-05 (Medium): `unsafe impl Send` requires stronger proof or removal

- **Affected code:** `src/inference.rs:90`
- **Issue:** `AsrInferenceInner` is force-marked `Send` via `unsafe impl Send` despite containing backend/device/tensor internals with non-trivial thread-safety assumptions.
- **Impact:** If assumptions are invalid for any backend/runtime evolution, this can lead to undefined behavior.
- **Recommendation:**
  - Prefer removing unsafe marker and redesigning ownership/thread model to satisfy compiler traits naturally.
  - If unavoidable, enforce backend-specific invariants with stronger guards and targeted stress tests.

### SEC-06 (Low): Hub downloads are unbounded and can hang indefinitely

- **Affected code:** `src/hub.rs:11`, `src/hub.rs:27`, `src/hub.rs:37`
- **Issue:**
  - HTTP clients use `timeout(None)` (infinite timeout).
  - Some responses are buffered fully in memory (`hf_get_bytes`), with no max-size controls.
- **Impact:** Availability risk (hang, memory/disk exhaustion) with malicious or unstable remote responses.
- **Recommendation:**
  - Configure connect/read timeouts and retry strategy.
  - Enforce maximum content lengths and streamed size caps.
  - Optionally support expected checksums/signatures for model artifacts.

### SEC-07 (Low): Debug-only panic risk when logits contain NaN

- **Affected code:** `src/inference.rs:322`
- **Issue:** `partial_cmp(...).unwrap()` can panic for NaN logits in debug logging path.
- **Impact:** Unexpected crash when debug logs are enabled and model numerics degrade.
- **Recommendation:** use `f32::total_cmp` or safe fallback ordering.

### SEC-08 (High, Remediated): Known vulnerable TLS validation dependency in `hub` download path

- **Affected component:** `reqwest` dependency used by optional `hub` feature (`src/hub.rs`, `Cargo.toml:33`)
- **Evidence:** `cargo audit` reported:
  - `RUSTSEC-2026-0098` (`rustls-webpki 0.103.9`)
  - `RUSTSEC-2026-0099` (`rustls-webpki 0.103.9`)
  - `RUSTSEC-2026-0049` (`rustls-webpki 0.103.9`)
- **Impact:** certificate validation weaknesses in transitive TLS stack for remote model downloads can reduce trust guarantees of HTTPS endpoint validation.
- **Recommendation:**
  - Update lockfile to a patched `rustls-webpki` (`>=0.103.12`) if semver-compatible.
  - If resolver does not pick patched transitive automatically, pin/override via dependency update strategy and re-run `cargo audit`.
  - Treat this as release-blocking for builds shipping `hub` support.
- **Status (2026-04-21):** remediated in this workspace by lockfile update to `rustls-webpki 0.103.12`; follow-up `cargo audit` reports warnings only.

### SEC-09 (Low): Dependency hygiene warnings (maintenance/soundness)

- **Evidence from `cargo audit`:**
  - `RUSTSEC-2024-0436`: `paste` unmaintained (transitive through `tokenizers` / `candle` stack)
  - `RUSTSEC-2026-0097`: `rand 0.9.2` unsoundness warning (transitive)
- **Impact:** presently warning-level in audit output; still increases long-term supply-chain risk.
- **Recommendation:**
  - Track upstream updates in `candle` / `tokenizers`.
  - Re-audit regularly and document temporary exceptions if releases must proceed before upstream fixes.

## Positive Observations

- No command execution primitives (`Command::new` etc.) in runtime library path.
- Secrets handling is minimal; `HUGGING_FACE_HUB_TOKEN` is used for auth and not logged directly.
- Error propagation generally uses `Result` instead of pervasive panics in core inference path.

## Prioritized Remediation Plan

1. Fix path traversal in `model_id` and shard path handling (`SEC-01`, `SEC-02`).
2. Add centralized config validation and input guards (`SEC-03`, `SEC-04`).
3. Replace/justify `unsafe impl Send` with stronger guarantees (`SEC-05`).
4. Add network/resource safety controls for hub downloads (`SEC-06`).
5. Remove debug panic edge case (`SEC-07`).
6. Track warning-level supply-chain issues (`SEC-09`).
