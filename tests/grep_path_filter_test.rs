use std::fs;
use std::path::Path;

use tempfile::TempDir;
use vex::grep;

fn project() -> TempDir {
    let tmp = TempDir::new().unwrap();
    for name in ["src/api/routes.py", "src/apix/routes.py", "other/routes.py"] {
        let path = tmp.path().join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "needle\n").unwrap();
    }
    tmp
}

#[test]
fn directory_separator_filters_find_only_the_selected_directory() {
    let tmp = project();
    let mut filters = vec!["src/api/", "src/api/routes.py"];
    if cfg!(windows) {
        filters.extend([r"src\api\", r"src\api/routes.py", r"src/api\routes.py"]);
    }
    for filter in filters {
        let hits = grep::search(tmp.path(), "needle", Some(filter), 100, &[], false).unwrap();
        assert_eq!(hits.len(), 1, "filter {filter:?}: {hits:?}");
        assert_eq!(
            hits[0].path,
            Path::new("src")
                .join("api")
                .join("routes.py")
                .to_str()
                .unwrap()
        );
        assert_eq!(hits[0].line, 1);
        assert_eq!(hits[0].text, "needle");
    }
}

#[test]
fn path_filter_remains_a_case_sensitive_substring() {
    let tmp = project();
    let hits = grep::search(tmp.path(), "needle", Some("api"), 100, &[], false).unwrap();
    assert_eq!(hits.len(), 2);
    for filter in ["API", "src/api/*", "missing/"] {
        assert!(
            grep::search(tmp.path(), "needle", Some(filter), 100, &[], false)
                .unwrap()
                .is_empty(),
            "filter {filter:?} must not change substring semantics"
        );
    }
}

#[cfg(unix)]
#[test]
fn unix_backslashes_in_filenames_remain_literal() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join(r"literal\name.py"), "needle\n").unwrap();
    let hits = grep::search(tmp.path(), "needle", Some(r"literal\"), 100, &[], false).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].path, r"literal\name.py");
    assert!(
        grep::search(tmp.path(), "needle", Some("literal/"), 100, &[], false)
            .unwrap()
            .is_empty()
    );
}
