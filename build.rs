use std::path::{Path, PathBuf};
use std::process::Command;

fn git_path(manifest_dir: &Path, option: &str) -> Option<PathBuf> {
    let output = Command::new("git")
        .current_dir(manifest_dir)
        .args(["rev-parse", option])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let path = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    let path = if path.is_absolute() {
        path
    } else {
        manifest_dir.join(path)
    };
    Some(path.canonicalize().unwrap_or(path))
}

fn rerun_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

fn watch_git_metadata(manifest_dir: &Path, git_entry: &Path) {
    // A linked worktree stores `.git` as a pointer file; watching it also
    // catches a checkout whose gitdir pointer is replaced.
    if git_entry.is_file() {
        rerun_if_exists(git_entry);
    }

    let git_dir = git_path(manifest_dir, "--git-dir")
        .or_else(|| git_entry.is_dir().then(|| git_entry.to_path_buf()));
    let Some(git_dir) = git_dir else {
        return;
    };

    // HEAD is worktree-local (including detached HEAD), while branch and tag
    // refs and packed-refs live in the common Git directory.
    rerun_if_exists(&git_dir.join("HEAD"));
    let common_dir = git_path(manifest_dir, "--git-common-dir").unwrap_or(git_dir);
    for refs_dir in ["refs/heads", "refs/tags"] {
        rerun_if_exists(&common_dir.join(refs_dir));
    }
    rerun_if_exists(&common_dir.join("packed-refs"));
}

fn main() {
    // Once any rerun paths are declared, Cargo no longer assumes that every
    // package file is an input. Keep the build script itself explicit.
    println!("cargo:rerun-if-changed=build.rs");

    // Set VEX_VERSION from git tag at compile time — but only when building
    // from a vex checkout. A crates.io build (`cargo install vex-search`)
    // unpacks into ~/.cargo/registry/src/… with no `.git`; running `git
    // describe` there would walk up into whatever enclosing repo exists and
    // stamp its tag/hash as vex's version.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let git_entry = manifest_dir.join(".git");
    let in_checkout = git_entry.exists();
    if in_checkout {
        watch_git_metadata(&manifest_dir, &git_entry);
    }

    let version = in_checkout
        .then(|| {
            Command::new("git")
                .current_dir(&manifest_dir)
                .args(["describe", "--tags", "--always"])
                .output()
                .ok()
        })
        .flatten()
        .and_then(|output| {
            let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if value.is_empty() {
                None
            } else {
                Some(value)
            }
        })
        .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());

    println!("cargo:rustc-env=VEX_VERSION={version}");
}
