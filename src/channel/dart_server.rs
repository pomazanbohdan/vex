use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

mod protocol_json;
use protocol_json::parse_message;

pub(super) fn normal_path(path: &Path) -> Result<PathBuf> {
    let path = path.canonicalize()?;
    #[cfg(windows)]
    {
        let text = path.to_string_lossy();
        if let Some(tail) = text.strip_prefix(r"\\?\UNC\") {
            return Ok(PathBuf::from(format!(r"\\{tail}")));
        }
        if let Some(tail) = text.strip_prefix(r"\\?\") {
            return Ok(PathBuf::from(tail));
        }
    }
    Ok(path)
}

fn executable(path: PathBuf) -> Result<PathBuf> {
    let path = if path.is_dir() {
        path.join("bin")
            .join(if cfg!(windows) { "dart.exe" } else { "dart" })
    } else {
        path
    };
    let path = normal_path(&path)
        .with_context(|| format!("Dart executable {} not found", path.display()))?;
    anyhow::ensure!(path.is_file(), "Dart executable is not a file");
    Ok(path)
}

pub(super) fn discover_sdk(root: &Path) -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("VEX_DART_SDK") {
        return executable(PathBuf::from(explicit));
    }
    for ancestor in root.ancestors() {
        let fvm = ancestor.join(".fvm/flutter_sdk/bin/cache/dart-sdk");
        if fvm.is_dir() {
            return executable(fvm);
        }
        let config = ancestor.join(".fvmrc");
        if config.is_file() {
            let value: Value = serde_json::from_slice(&std::fs::read(&config)?)?;
            if let (Some(version), Some(home)) = (
                value["flutter"].as_str(),
                std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")),
            ) {
                anyhow::ensure!(
                    !version.contains(['/', '\\']) && version != "..",
                    "invalid FVM version"
                );
                let sdk = PathBuf::from(home)
                    .join("fvm/versions")
                    .join(version)
                    .join("bin/cache/dart-sdk");
                if sdk.is_dir() {
                    return executable(sdk);
                }
            }
        }
    }
    if let Some(path) = std::env::var_os("PATH") {
        for directory in std::env::split_paths(&path) {
            let dart = directory.join(if cfg!(windows) { "dart.exe" } else { "dart" });
            if dart.is_file() {
                return executable(dart);
            }
        }
    }
    bail!("no Dart SDK found; set VEX_DART_SDK to an existing SDK directory or dart executable")
}

#[derive(serde::Serialize)]
pub(super) struct Runtime {
    pub(super) mode: &'static str,
    pub(super) executable: PathBuf,
    pub(super) snapshot: PathBuf,
}

struct RequestWrite {
    bytes: Vec<u8>,
    acknowledged: mpsc::Sender<std::io::Result<()>>,
}

fn request_writer(mut input: ChildStdin) -> mpsc::Sender<RequestWrite> {
    let (send, requests) = mpsc::channel::<RequestWrite>();
    std::thread::spawn(move || {
        for request in requests {
            let result = input.write_all(&request.bytes).and_then(|_| input.flush());
            let failed = result.is_err();
            if request.acknowledged.send(result).is_err() || failed {
                break;
            }
        }
    });
    send
}

fn message_reader(output: impl BufRead + Send + 'static) -> Receiver<Result<Value, String>> {
    let (send, messages) = mpsc::sync_channel(64);
    std::thread::spawn(move || {
        for line in output.lines() {
            let value = line
                .map_err(|e| e.to_string())
                .and_then(|line| parse_message(&line).map_err(|e| e.to_string()));
            let value = match value {
                Ok(None) => continue,
                Ok(Some(value)) => Ok(value),
                Err(error) => Err(error),
            };
            let failed = value.is_err();
            if send.send(value).is_err() || failed {
                break;
            }
        }
    });
    messages
}

pub(super) struct Server {
    child: Child,
    writes: Option<mpsc::Sender<RequestWrite>>,
    messages: Receiver<Result<Value, String>>,
    next_id: u64,
    deadline: Instant,
    outlines: HashMap<PathBuf, Value>,
    searches: HashMap<String, Vec<Value>>,
    complete: std::collections::HashSet<String>,
    failed: bool,
    pub(super) version: String,
    pub(super) runtime: Runtime,
}

impl Server {
    fn select_runtime(dart: &Path) -> Result<Runtime> {
        let dart = executable(dart.to_owned())?;
        let bin = dart.parent().context("Dart executable parent")?;
        let aot = bin.join(if cfg!(windows) {
            "dartaotruntime.exe"
        } else {
            "dartaotruntime"
        });
        let aot_snapshot = bin.join("snapshots/analysis_server_aot.dart.snapshot");
        if aot.is_file() && aot_snapshot.is_file() {
            return Ok(Runtime {
                mode: "aot",
                executable: normal_path(&aot)?,
                snapshot: normal_path(&aot_snapshot)?,
            });
        }
        let snapshot = bin.join("snapshots/analysis_server.dart.snapshot");
        anyhow::ensure!(
            snapshot.is_file(),
            "SDK has no Analysis Server snapshot: {}",
            snapshot.display()
        );
        Ok(Runtime {
            mode: "jit",
            executable: dart,
            snapshot: normal_path(&snapshot)?,
        })
    }

    pub(super) fn start(dart: &Path, cache: &Path) -> Result<Self> {
        let timeout_ms = std::env::var("VEX_DART_TIMEOUT_MS")
            .ok()
            .map(|s| s.parse::<u64>())
            .transpose()
            .context("invalid VEX_DART_TIMEOUT_MS")?
            .unwrap_or(180_000)
            .clamp(1, 600_000);
        let runtime = Self::select_runtime(dart)?;
        let mut command = Command::new(&runtime.executable);
        command
            .arg(&runtime.snapshot)
            .arg("--protocol=analyzer")
            .arg("--client-id=vex")
            .arg(format!("--cache={}", cache.display()))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            command.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = command.spawn().context("start Dart Analysis Server")?;
        let input = child.stdin.take().context("Analysis Server stdin")?;
        let output = child.stdout.take().context("Analysis Server stdout")?;
        let messages = message_reader(BufReader::new(output));
        let mut server = Self {
            child,
            writes: Some(request_writer(input)),
            messages,
            next_id: 0,
            deadline: Instant::now() + Duration::from_millis(timeout_ms),
            outlines: HashMap::new(),
            searches: HashMap::new(),
            complete: Default::default(),
            failed: false,
            version: String::new(),
            runtime,
        };
        tracing::debug!(mode = server.runtime.mode, executable = %server.runtime.executable.display(), snapshot = %server.runtime.snapshot.display(), "Dart Analysis Server runtime");
        let version = server.request("server.getVersion", json!({}))?;
        server.version = version["version"]
            .as_str()
            .context("Analysis Server version missing")?
            .to_owned();
        Ok(server)
    }

    fn receive(&mut self) -> Result<Value> {
        let result = self.receive_message();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn receive_message(&mut self) -> Result<Value> {
        self.check_deadline()?;
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .context("Dart Analysis Server timeout")?;
        let value = self
            .messages
            .recv_timeout(remaining)
            .context("Dart Analysis Server timeout or disconnected")?
            .map_err(anyhow::Error::msg)?;
        let params = &value["params"];
        match value["event"].as_str() {
            Some("server.error") => bail!("Dart Analysis Server error: {params}"),
            Some("analysis.outline") => {
                let path = normal_path(Path::new(
                    params["file"].as_str().context("outline file missing")?,
                ))?;
                self.outlines.insert(path, params["outline"].clone());
            }
            Some("search.results") => {
                let id = params["id"]
                    .as_str()
                    .context("search id missing")?
                    .to_owned();
                let results = params["results"]
                    .as_array()
                    .context("search results missing")?;
                let is_last = params["isLast"]
                    .as_bool()
                    .context("search isLast missing or not boolean")?;
                self.searches
                    .entry(id.clone())
                    .or_default()
                    .extend(results.iter().cloned());
                if is_last {
                    self.complete.insert(id);
                }
            }
            _ => {}
        }
        Ok(value)
    }

    pub(super) fn check_deadline(&self) -> Result<()> {
        anyhow::ensure!(!self.failed, "Dart Analysis Server session failed");
        anyhow::ensure!(
            Instant::now() < self.deadline,
            "Dart Analysis Server timeout"
        );
        Ok(())
    }

    fn send_request(&mut self, method: &str, params: Value) -> Result<String> {
        self.check_deadline()?;
        self.next_id += 1;
        let id = self.next_id.to_string();
        let mut bytes = serde_json::to_vec(&json!({"id": id, "method": method, "params": params}))?;
        bytes.push(b'\n');
        let (acknowledged, written) = mpsc::channel();
        self.writes
            .as_ref()
            .context("Dart Analysis Server writer closed")?
            .send(RequestWrite {
                bytes,
                acknowledged,
            })
            .context("Dart Analysis Server writer disconnected")?;
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .context("Dart Analysis Server timeout writing request")?;
        written
            .recv_timeout(remaining)
            .context("Dart Analysis Server timeout writing request or disconnected")?
            .context("write Dart Analysis Server request")?;
        self.check_deadline()?;
        Ok(id)
    }

    pub(super) fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let result = self.request_result(method, params);
        if result.is_err() {
            self.failed = true;
            self.searches.clear();
            self.complete.clear();
        }
        result
    }

    fn request_result(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.send_request(method, params)?;
        loop {
            let value = self.receive()?;
            if value["id"] == id {
                if let Some(error) = value.get("error") {
                    bail!("Analysis Server {method}: {error}");
                }
                // Void protocol methods acknowledge with only {id}; typed consumers validate fields.
                return Ok(value.get("result").cloned().unwrap_or_else(|| json!({})));
            }
        }
    }

    pub(super) fn analyze(
        &mut self,
        root: &Path,
        files: &[PathBuf],
        excluded: &[PathBuf],
    ) -> Result<()> {
        self.request(
            "server.setSubscriptions",
            json!({"subscriptions": ["STATUS"]}),
        )?;
        self.request(
            "analysis.setAnalysisRoots",
            json!({"included": [root], "excluded": excluded}),
        )?;
        self.request("analysis.setPriorityFiles", json!({"files": files}))?;
        self.request(
            "analysis.setSubscriptions",
            json!({"subscriptions": {"OUTLINE": files}}),
        )?;
        while files.iter().any(|p| !self.outlines.contains_key(p)) {
            self.receive()?;
        }
        Ok(())
    }

    pub(super) fn outline(&self, file: &Path) -> Result<&Value> {
        self.outlines.get(file).context("Dart outline missing")
    }

    pub(super) fn references_many(
        &mut self,
        declarations: &[(PathBuf, u64)],
    ) -> Result<Vec<(Value, Vec<Value>)>> {
        self.check_deadline()?;
        let result = self.collect_references(declarations);
        self.searches.clear();
        self.complete.clear();
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    fn collect_references(
        &mut self,
        declarations: &[(PathBuf, u64)],
    ) -> Result<Vec<(Value, Vec<Value>)>> {
        let mut results = Vec::with_capacity(declarations.len());
        for batch in declarations.chunks(16) {
            let mut pending = HashMap::new();
            let mut resolved: Vec<Option<(Value, Option<String>)>> = vec![None; batch.len()];
            for (index, (file, offset)) in batch.iter().enumerate() {
                let id = self.send_request(
                    "search.findElementReferences",
                    json!({"file": file, "offset": offset, "includePotential": false}),
                )?;
                pending.insert(id, index);
            }
            while !pending.is_empty()
                || resolved
                    .iter()
                    .flatten()
                    .any(|(_, id)| id.as_ref().is_some_and(|id| !self.complete.contains(id)))
            {
                let value = self.receive()?;
                if let Some(index) = value["id"].as_str().and_then(|id| pending.remove(id)) {
                    if let Some(error) = value.get("error") {
                        bail!("Analysis Server search.findElementReferences: {error}");
                    }
                    let result = &value["result"];
                    let element = result
                        .get("element")
                        .filter(|element| !element.is_null())
                        .cloned()
                        .context("Dart declaration has no resolved element")?;
                    resolved[index] = Some((element, result["id"].as_str().map(str::to_owned)));
                }
            }
            for result in resolved {
                let (element, id) = result.context("Dart reference result missing")?;
                let references = if let Some(id) = id {
                    self.complete.remove(&id);
                    self.searches.remove(&id).unwrap_or_default()
                } else {
                    Vec::new()
                };
                results.push((element, references));
            }
        }
        self.check_deadline()?;
        Ok(results)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        drop(self.writes.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_filters_noise_preserves_payload_and_stops_after_invalid_json() {
        let messages = message_reader(std::io::Cursor::new(concat!(
            "{\"event\":\"analysis.errors\",\"params\":{\"errors\":[]}}\n",
            "{\"result\":{\"version\":\"3.13.5\"},\"id\":\"1\"}\n",
            "{\"event\":\"ignored\",\"params\":[1,]}\n",
            "{\"id\":\"must-not-arrive\"}\n",
        )));
        assert_eq!(
            messages
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap(),
            json!({"id":"1","result":{"version":"3.13.5"}})
        );
        assert!(messages
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .is_err());
        assert!(matches!(
            messages.recv_timeout(Duration::from_secs(2)),
            Err(mpsc::RecvTimeoutError::Disconnected)
        ));
    }

    #[test]
    fn full_reader_queue_applies_backpressure_and_receiver_drop_releases_reader() {
        struct CountedReader {
            inner: std::io::Cursor<Vec<u8>>,
            consumed: std::sync::Arc<std::sync::atomic::AtomicUsize>,
            closed: mpsc::Sender<()>,
        }
        impl std::io::Read for CountedReader {
            fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
                self.inner.read(bytes)
            }
        }
        impl BufRead for CountedReader {
            fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
                self.inner.fill_buf()
            }
            fn consume(&mut self, amount: usize) {
                self.inner.consume(amount);
                self.consumed
                    .fetch_add(amount, std::sync::atomic::Ordering::SeqCst);
            }
        }
        impl Drop for CountedReader {
            fn drop(&mut self) {
                let _ = self.closed.send(());
            }
        }
        let consumed = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (closed_send, closed) = mpsc::channel();
        let line = "{\"id\":\"1\"}\n";
        let messages = message_reader(CountedReader {
            inner: std::io::Cursor::new(line.repeat(100).into_bytes()),
            consumed: consumed.clone(),
            closed: closed_send,
        });
        let deadline = Instant::now() + Duration::from_secs(2);
        while consumed.load(std::sync::atomic::Ordering::SeqCst) < 65 * line.len() {
            assert!(Instant::now() < deadline);
            std::thread::yield_now();
        }
        assert_eq!(
            consumed.load(std::sync::atomic::Ordering::SeqCst),
            65 * line.len()
        );
        assert!(closed.try_recv().is_err());
        drop(messages);
        closed
            .recv_timeout(Duration::from_secs(2))
            .expect("reader must exit when a full queue loses its receiver");
    }

    fn server(values: Vec<Value>) -> (Server, mpsc::Sender<Result<Value, String>>) {
        let mut command = if cfg!(windows) {
            Command::new(
                PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join("System32/more.com"),
            )
        } else {
            Command::new("cat")
        };
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let (send, messages) = mpsc::channel();
        for value in values {
            send.send(Ok(value)).unwrap();
        }
        (
            Server {
                child,
                writes: Some(request_writer(input)),
                messages,
                next_id: 0,
                deadline: Instant::now() + Duration::from_millis(100),
                outlines: HashMap::new(),
                searches: HashMap::new(),
                complete: Default::default(),
                failed: false,
                version: String::new(),
                runtime: Runtime {
                    mode: "jit",
                    executable: PathBuf::new(),
                    snapshot: PathBuf::new(),
                },
            },
            send,
        )
    }

    #[test]
    fn check_deadline_rejects_expired_query_after_receiving_results() {
        let (mut server, _send) = server(vec![]);
        server.deadline = Instant::now() - Duration::from_millis(1);
        assert!(server.check_deadline().is_err());
    }

    #[test]
    #[ignore = "subprocess peer invoked only by the blocked-stdin test"]
    fn refuses_stdin_peer() {
        if std::env::var_os("VEX_TEST_REFUSE_STDIN").is_some() {
            println!("peer ready");
            std::io::stdout().flush().unwrap();
            std::thread::sleep(Duration::from_millis(800));
        }
    }

    #[test]
    fn blocked_stdin_write_obeys_deadline_and_drop_kills_peer() {
        let (mut server, _send) = server(vec![]);
        server.child.kill().unwrap();
        server.child.wait().unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "channel::dart_server::tests::refuses_stdin_peer",
                "--nocapture",
            ])
            .env("VEX_TEST_REFUSE_STDIN", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        server.writes = Some(request_writer(child.stdin.take().unwrap()));
        let output = child.stdout.take().unwrap();
        server.child = child;
        let (ready_send, ready) = mpsc::channel();
        let (closed_send, closed) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                if line.unwrap().contains("peer ready") {
                    ready_send.send(()).unwrap();
                }
            }
            let _ = closed_send.send(());
        });
        ready.recv_timeout(Duration::from_secs(5)).unwrap();
        server.deadline = Instant::now() + Duration::from_millis(30);
        let started = Instant::now();
        let declarations: Vec<_> = (0..16)
            .map(|offset| (PathBuf::from("x".repeat(512 * 1024)), offset))
            .collect();
        let error = server.references_many(&declarations).unwrap_err();
        assert!(
            started.elapsed() < Duration::from_millis(300),
            "blocked request exceeded deadline: {:?}; {error:#}",
            started.elapsed()
        );
        assert!(error.to_string().contains("timeout"), "{error:#}");
        assert!(
            server.child.try_wait().unwrap().is_none(),
            "peer must still be alive before cleanup"
        );
        drop(server);
        closed
            .recv_timeout(Duration::from_millis(300))
            .expect("Drop must kill and reap the peer, closing stdout");
        assert!(started.elapsed() < Duration::from_millis(600));
    }

    #[test]
    fn runtime_prefers_complete_aot_pair_and_falls_back_for_partial_pair() {
        for (runtime_exists, snapshot_exists, mode) in [
            (true, true, "aot"),
            (true, false, "jit"),
            (false, true, "jit"),
            (false, false, "jit"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let bin = dir.path().join("bin");
            std::fs::create_dir_all(bin.join("snapshots")).unwrap();
            let dart = bin.join(if cfg!(windows) { "dart.exe" } else { "dart" });
            let aot = bin.join(if cfg!(windows) {
                "dartaotruntime.exe"
            } else {
                "dartaotruntime"
            });
            std::fs::write(&dart, "").unwrap();
            std::fs::write(bin.join("snapshots/analysis_server.dart.snapshot"), "").unwrap();
            if runtime_exists {
                std::fs::write(&aot, "").unwrap();
            }
            if snapshot_exists {
                std::fs::write(bin.join("snapshots/analysis_server_aot.dart.snapshot"), "")
                    .unwrap();
            }
            let selected = Server::select_runtime(&dart).unwrap();
            assert_eq!(selected.mode, mode);
            assert_eq!(
                selected.executable,
                normal_path(if mode == "aot" { &aot } else { &dart }).unwrap()
            );
            assert_eq!(
                selected.snapshot.file_name().unwrap(),
                if mode == "aot" {
                    "analysis_server_aot.dart.snapshot"
                } else {
                    "analysis_server.dart.snapshot"
                }
            );
        }
    }

    #[test]
    fn runtime_rejects_missing_selected_dart_even_with_aot_pair() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("snapshots")).unwrap();
        std::fs::write(
            dir.path().join(if cfg!(windows) {
                "dartaotruntime.exe"
            } else {
                "dartaotruntime"
            }),
            "",
        )
        .unwrap();
        std::fs::write(
            dir.path()
                .join("snapshots/analysis_server_aot.dart.snapshot"),
            "",
        )
        .unwrap();
        assert!(Server::select_runtime(&dir.path().join("missing-dart")).is_err());
    }

    #[test]
    fn runtime_rejects_missing_jit_snapshot_for_incomplete_aot_pair() {
        let dir = tempfile::tempdir().unwrap();
        let dart = dir
            .path()
            .join(if cfg!(windows) { "dart.exe" } else { "dart" });
        std::fs::write(&dart, "").unwrap();
        assert!(Server::select_runtime(&dart).is_err());
    }

    #[test]
    fn analyze_returns_when_required_outline_arrives_during_global_analysis() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("main.dart");
        std::fs::write(&file, "void main() {}\n").unwrap();
        let file = normal_path(&file).unwrap();
        let (mut server, _send) = server(vec![
            json!({"id": "1"}),
            json!({"event": "server.status", "params": {"analysis": {"isAnalyzing": true}}}),
            json!({"id": "2"}),
            json!({"id": "3"}),
            json!({"event": "analysis.outline", "params": {"file": file, "outline": {"element": {"name": "main"}}}}),
            json!({"id": "4"}),
        ]);
        server
            .analyze(dir.path(), std::slice::from_ref(&file), &[])
            .unwrap();
        assert_eq!(server.outline(&file).unwrap()["element"]["name"], "main");
    }

    #[test]
    fn references_many_preserves_order_with_early_and_interleaved_streams() {
        let (mut server, _send) = server(vec![
            json!({"event": "search.results", "params": {"id": "b", "results": [{"offset": 22}], "isLast": false}}),
            json!({"id": "2", "result": {"element": {"name": "second"}, "id": "b"}}),
            json!({"event": "search.results", "params": {"id": "a", "results": [{"offset": 11}], "isLast": true}}),
            json!({"id": "1", "result": {"element": {"name": "first"}, "id": "a"}}),
            json!({"event": "search.results", "params": {"id": "b", "results": [{"offset": 23}], "isLast": true}}),
        ]);
        let result = server
            .references_many(&[
                (PathBuf::from("one.dart"), 1),
                (PathBuf::from("two.dart"), 2),
            ])
            .unwrap();
        assert_eq!(
            result,
            vec![
                (json!({"name": "first"}), vec![json!({"offset": 11})]),
                (
                    json!({"name": "second"}),
                    vec![json!({"offset": 22}), json!({"offset": 23})]
                )
            ]
        );
    }

    #[test]
    fn references_many_fails_on_error_from_another_pending_request() {
        let (mut server, _send) = server(vec![
            json!({"event": "search.results", "params": {"id": "partial", "results": [{"offset": 11}], "isLast": true}}),
            json!({"id": "2", "error": {"code": "INVALID_PARAMETER", "message": "bad offset"}}),
        ]);
        let error = server
            .references_many(&[
                (PathBuf::from("one.dart"), 1),
                (PathBuf::from("two.dart"), 2),
            ])
            .unwrap_err();
        assert!(error.to_string().contains("INVALID_PARAMETER"), "{error:#}");
        assert!(server.searches.is_empty());
        assert!(server.complete.is_empty());
        assert!(server.failed);
        assert!(server
            .references_many(&[])
            .unwrap_err()
            .to_string()
            .contains("session failed"));
    }

    #[test]
    fn references_many_accepts_no_stream_and_empty_completed_stream() {
        let (mut server, _send) = server(vec![
            json!({"id": "1", "result": {"element": {"name": "first"}}}),
            json!({"id": "2", "result": {"element": {"name": "second"}, "id": "empty"}}),
            json!({"event": "search.results", "params": {"id": "empty", "results": [], "isLast": true}}),
        ]);
        let result = server
            .references_many(&[
                (PathBuf::from("one.dart"), 1),
                (PathBuf::from("two.dart"), 2),
            ])
            .unwrap();
        assert_eq!(
            result,
            vec![
                (json!({"name": "first"}), vec![]),
                (json!({"name": "second"}), vec![])
            ]
        );
        assert!(server.references_many(&[]).unwrap().is_empty());
    }

    #[test]
    fn successive_batches_keep_session_identity_deadline_and_clear_stream_state() {
        let (mut server, _send) = server(vec![
            json!({"id": "1", "result": {"element": {"name": "first"}, "id": "s1"}}),
            json!({"event":"search.results","params":{"id":"s1","results":[{"offset":11}],"isLast":true}}),
            json!({"id": "2", "result": {"element": {"name": "second"}, "id": "s2"}}),
            json!({"event":"search.results","params":{"id":"s2","results":[{"offset":22}],"isLast":true}}),
        ]);
        let process = server.child.id();
        let deadline = server.deadline;
        for (offset, name) in [(11, "first"), (22, "second")] {
            let results = server
                .references_many(&[(PathBuf::from("main.dart"), offset)])
                .unwrap();
            assert_eq!(
                results,
                [(json!({"name":name}), vec![json!({"offset":offset})])]
            );
            assert_eq!(server.child.id(), process);
            assert_eq!(server.deadline, deadline);
            assert!(server.searches.is_empty());
            assert!(server.complete.is_empty());
            assert!(!server.failed);
        }
        assert_eq!(server.next_id, 2);
    }

    #[test]
    fn malformed_transport_and_server_error_poison_session() {
        for message in [
            Err("invalid JSON".to_owned()),
            Ok(json!({"event":"server.error","params":{"message":"failed"}})),
        ] {
            let (mut server, send) = server(vec![]);
            send.send(message).unwrap();
            assert!(server.request("server.getVersion", json!({})).is_err());
            assert!(server.failed);
            assert!(server
                .request("server.getVersion", json!({}))
                .unwrap_err()
                .to_string()
                .contains("session failed"));
        }
    }

    #[test]
    fn malformed_search_completion_fails_immediately_without_partial_state() {
        for invalid in [Value::Null, json!("true"), json!(1), json!([]), json!({})] {
            let mut params = json!({"id":"s1","results":[{"offset":11}]});
            if !invalid.is_null() {
                params["isLast"] = invalid;
            }
            let (mut server, _send) =
                server(vec![json!({"event":"search.results","params":params})]);
            let error = server.receive().unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("search isLast missing or not boolean"),
                "{error:#}"
            );
            assert!(server.failed);
            assert!(server.searches.is_empty());
            assert!(server.complete.is_empty());
        }
    }

    #[test]
    fn references_many_requires_is_last_for_every_stream() {
        let (mut server, _send) = server(vec![
            json!({"id": "1", "result": {"element": {"name": "first"}, "id": "pending"}}),
            json!({"event": "search.results", "params": {"id": "pending", "results": [{"offset": 11}], "isLast": false}}),
        ]);
        let error = server
            .references_many(&[(PathBuf::from("one.dart"), 1)])
            .unwrap_err();
        assert!(error.to_string().contains("timeout"), "{error:#}");
    }

    #[test]
    fn configured_sdk_resolves_batch_with_unicode_and_named_constructor() {
        if std::env::var_os("VEX_DART_SDK").is_none() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let source = "class Service { Service(); Service.named(); void run() {} }\nvoid target() {}\nvoid caller() { final emoji = '🦊'; target(); Service.named().run(); }\n";
        let file = dir.path().join("main.dart");
        std::fs::write(&file, source).unwrap();
        let file = normal_path(&file).unwrap();
        let dart = discover_sdk(dir.path()).unwrap();
        let mut server = Server::start(&dart, &dir.path().join("cache")).unwrap();
        let metadata = serde_json::to_value(&server.runtime).unwrap();
        assert!(metadata["mode"] == "aot" || metadata["mode"] == "jit");
        assert!(Path::new(metadata["executable"].as_str().unwrap()).is_file());
        assert!(Path::new(metadata["snapshot"].as_str().unwrap()).is_file());
        server
            .analyze(dir.path(), std::slice::from_ref(&file), &[])
            .unwrap();
        let declarations: Vec<_> = ["target() {}", "run() {}", "named();"]
            .iter()
            .map(|name| {
                let offset = source[..source.find(name).unwrap()].encode_utf16().count() as u64;
                (file.clone(), offset)
            })
            .collect();
        let results = server.references_many(&declarations).unwrap();
        assert_eq!(results.len(), 3);
        for (_, references) in results {
            assert_eq!(references.len(), 1);
            assert_eq!(references[0]["location"]["startLine"], 3);
            assert_eq!(
                normal_path(Path::new(
                    references[0]["location"]["file"].as_str().unwrap()
                ))
                .unwrap(),
                file
            );
        }
        let process = server.child.id();
        let deadline = server.deadline;
        let repeated = server.references_many(&declarations).unwrap();
        assert_eq!(repeated.len(), 3);
        assert!(repeated.iter().all(|(_, references)| references.len() == 1));
        assert_eq!(server.child.id(), process);
        assert_eq!(server.deadline, deadline);
        assert!(server.searches.is_empty());
        assert!(server.complete.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn references_many_limits_in_flight_searches_to_sixteen() {
        let (mut server, send) = server(vec![]);
        server.child.kill().unwrap();
        server.child.wait().unwrap();
        let shell = PathBuf::from(std::env::var_os("SystemRoot").unwrap())
            .join("System32/WindowsPowerShell/v1.0/powershell.exe");
        let mut child = Command::new(shell).args(["-NoProfile", "-Command", "while ($null -ne ($line = [Console]::ReadLine())) { [Console]::WriteLine($line); [Console]::Out.Flush() }"])
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
        server.writes = Some(request_writer(child.stdin.take().unwrap()));
        let output = child.stdout.take().unwrap();
        server.child = child;
        server.deadline = Instant::now() + Duration::from_secs(5);
        let (requests_send, requests) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                if requests_send
                    .send(serde_json::from_str::<Value>(&line.unwrap()).unwrap())
                    .is_err()
                {
                    break;
                }
            }
        });
        let peer = std::thread::spawn(move || {
            for id in 1..=16 {
                let request = requests.recv_timeout(Duration::from_secs(4)).unwrap();
                assert_eq!(request["id"], id.to_string());
                assert_eq!(request["method"], "search.findElementReferences");
                assert_eq!(request["params"]["offset"], id);
                assert_eq!(request["params"]["includePotential"], false);
            }
            for id in (1..=16).rev() {
                send.send(Ok(json!({"id": id.to_string(), "result": {"element": {"offset": id}, "id": format!("s{id}")}}))).unwrap();
                if id != 16 {
                    send.send(Ok(json!({"event": "search.results", "params": {"id": format!("s{id}"), "results": [], "isLast": true}}))).unwrap();
                }
            }
            assert!(requests.recv_timeout(Duration::from_millis(30)).is_err());
            send.send(Ok(json!({"event": "search.results", "params": {"id": "s16", "results": [], "isLast": true}}))).unwrap();
            assert_eq!(
                requests.recv_timeout(Duration::from_secs(4)).unwrap()["id"],
                "17"
            );
            send.send(Ok(
                json!({"id": "17", "result": {"element": {"offset": 17}}}),
            ))
            .unwrap();
        });
        let declarations: Vec<_> = (1..=17)
            .map(|offset| (PathBuf::from("one.dart"), offset))
            .collect();
        let result = server.references_many(&declarations).unwrap();
        peer.join().unwrap();
        assert_eq!(result.len(), 17);
        assert_eq!(result[0].0["offset"], 1);
        assert_eq!(result[16].0["offset"], 17);
    }
}
