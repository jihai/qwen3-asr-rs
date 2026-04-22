use std::path::{Component, Path, PathBuf};

/// Convert a Hugging Face model id into a safe cache directory leaf.
///
/// Example: `Qwen/Qwen3-ASR-0.6B` -> `Qwen--Qwen3-ASR-0.6B`
#[cfg(feature = "hub")]
pub(crate) fn cache_leaf_from_model_id(model_id: &str) -> anyhow::Result<String> {
    if model_id.is_empty() {
        anyhow::bail!("model_id cannot be empty");
    }
    if model_id.contains('\\') {
        anyhow::bail!("model_id must not contain backslashes");
    }

    let mut parts = Vec::new();
    for part in model_id.split('/') {
        if part.is_empty() {
            anyhow::bail!("model_id contains an empty path segment");
        }
        if part == "." || part == ".." {
            anyhow::bail!("model_id contains a traversal segment");
        }
        if !part
            .bytes()
            .all(|b| matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'-' | b'.'))
        {
            anyhow::bail!("model_id contains unsupported characters");
        }
        parts.push(part);
    }

    Ok(parts.join("--"))
}

/// Safely joins a user-controlled relative path under `base`.
///
/// Rejects absolute paths, `.`/`..` segments, and platform-prefix components.
pub(crate) fn join_safe_relative(
    base: &Path,
    rel: &str,
    field_name: &str,
) -> anyhow::Result<PathBuf> {
    if rel.is_empty() {
        anyhow::bail!("{field_name} cannot be empty");
    }
    if rel.contains('\\') {
        anyhow::bail!("{field_name} must not contain backslashes");
    }

    let mut clean = PathBuf::new();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(seg) => clean.push(seg),
            Component::CurDir => anyhow::bail!("{field_name} must not contain '.' segments"),
            Component::ParentDir => anyhow::bail!("{field_name} must not contain '..' segments"),
            Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!("{field_name} must be a relative path")
            }
        }
    }

    if clean.as_os_str().is_empty() {
        anyhow::bail!("{field_name} cannot be empty");
    }

    Ok(base.join(clean))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "hub")]
    #[test]
    fn cache_leaf_from_model_id_happy_path() {
        let got = cache_leaf_from_model_id("Qwen/Qwen3-ASR-0.6B").unwrap();
        assert_eq!(got, "Qwen--Qwen3-ASR-0.6B");
    }

    #[cfg(feature = "hub")]
    #[test]
    fn cache_leaf_from_model_id_rejects_traversal() {
        assert!(cache_leaf_from_model_id("..").is_err());
        assert!(cache_leaf_from_model_id("org/../model").is_err());
    }

    #[cfg(feature = "hub")]
    #[test]
    fn cache_leaf_from_model_id_rejects_empty_segments() {
        assert!(cache_leaf_from_model_id("/model").is_err());
        assert!(cache_leaf_from_model_id("org//model").is_err());
    }

    #[test]
    fn join_safe_relative_happy_path() {
        let base = Path::new("/tmp/models");
        let got = join_safe_relative(base, "model-00001-of-00002.safetensors", "shard").unwrap();
        assert_eq!(
            got,
            PathBuf::from("/tmp/models/model-00001-of-00002.safetensors")
        );
    }

    #[test]
    fn join_safe_relative_rejects_escape_attempts() {
        let base = Path::new("/tmp/models");
        assert!(join_safe_relative(base, "../outside", "shard").is_err());
        assert!(join_safe_relative(base, "/abs/path", "shard").is_err());
        assert!(join_safe_relative(base, "./local", "shard").is_err());
    }
}
