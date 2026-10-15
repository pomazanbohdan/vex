use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

const MAGIC: &[u8; 4] = b"VEXR";
pub(super) const CURRENT: u16 = 1;
const FILE_NAME: &str = "index.parse-revision";

fn marker_bytes() -> [u8; 6] {
    let version = CURRENT.to_le_bytes();
    [
        MAGIC[0], MAGIC[1], MAGIC[2], MAGIC[3], version[0], version[1],
    ]
}

/// The marker versions parser carry semantics independently from cache payloads.
pub(super) fn is_current(index_dir: &Path) -> bool {
    fs::read(index_dir.join(FILE_NAME)).is_ok_and(|bytes| bytes == marker_bytes())
}

/// Durably make the index stale before the first required output mutation.
pub(super) fn invalidate(index_dir: &Path) -> Result<()> {
    let path = index_dir.join(FILE_NAME);
    let marker = match OpenOptions::new().write(true).truncate(true).open(&path) {
        Ok(file) => file,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err).with_context(|| format!("invalidate {}", path.display())),
    };
    marker
        .sync_all()
        .with_context(|| format!("sync invalidated marker {}", path.display()))
}

/// Publish the current parser carry revision after all outputs succeed.
pub(super) fn save(index_dir: &Path) -> Result<()> {
    fs::create_dir_all(index_dir)
        .with_context(|| format!("create index directory {}", index_dir.display()))?;
    let path = index_dir.join(FILE_NAME);
    let tmp = index_dir.join("index.parse-revision.tmp");
    let mut file = File::create(&tmp)
        .with_context(|| format!("create revision marker tmp {}", tmp.display()))?;
    file.write_all(&marker_bytes())
        .with_context(|| format!("write revision marker tmp {}", tmp.display()))?;
    file.sync_all()
        .with_context(|| format!("sync revision marker tmp {}", tmp.display()))?;
    // The same-directory rename atomically replaces the invalidated marker.
    fs::rename(&tmp, &path).with_context(|| format!("publish revision marker {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn marker_is_exact_and_invalidated_durably() {
        let tmp = TempDir::new().unwrap();
        save(tmp.path()).unwrap();
        assert_eq!(
            fs::read(tmp.path().join(FILE_NAME)).unwrap(),
            b"VEXR\x01\x00"
        );
        assert!(is_current(tmp.path()));
        invalidate(tmp.path()).unwrap();
        assert!(!is_current(tmp.path()));
        save(tmp.path()).unwrap();
        assert!(is_current(tmp.path()));
    }

    #[test]
    fn failed_marker_rename_keeps_stale_marker_and_temp_for_retry() {
        let tmp = TempDir::new().unwrap();
        save(tmp.path()).unwrap();
        invalidate(tmp.path()).unwrap();

        let marker = tmp.path().join(FILE_NAME);
        fs::remove_file(&marker).unwrap();
        fs::create_dir(&marker).unwrap();
        let tmp_marker = tmp.path().join("index.parse-revision.tmp");

        assert!(save(tmp.path()).is_err());
        assert!(!is_current(tmp.path()));
        assert!(tmp_marker.is_file());

        fs::remove_dir(&marker).unwrap();
        save(tmp.path()).unwrap();
        assert!(is_current(tmp.path()));
        assert!(!tmp_marker.exists());
    }
}
