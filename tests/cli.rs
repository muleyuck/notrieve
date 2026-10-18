use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

mod support;

use support::fake_api::{FakeApi, Reply};
use support::remote_config;

struct Env {
    home: TempDir,
}

impl Env {
    fn new() -> Self {
        Self {
            home: TempDir::new().unwrap(),
        }
    }

    fn command(&self, cwd: &Path, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_notrieve"));
        command
            .args(args)
            .current_dir(cwd)
            .env("HOME", self.home.path())
            .env("XDG_CONFIG_HOME", self.home.path().join("config"))
            .env("XDG_CACHE_HOME", self.home.path().join("cache"))
            .env_remove("NOTRIEVE_LOG");
        command
    }

    fn run(&self, cwd: &Path, args: &[&str]) -> Output {
        self.command(cwd, args).output().unwrap()
    }

    fn dir(&self, name: &str) -> PathBuf {
        let dir = self.home.path().join(name);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn vault(&self, name: &str) -> PathBuf {
        let root = self.dir(name);
        fs::create_dir_all(root.join(".notrieve")).unwrap();
        fs::write(
            root.join(".notrieve/config.toml"),
            "model = \"ruri-v3-30m\"\n",
        )
        .unwrap();
        root
    }
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn vault_list_reports_no_vaults_initially() {
    let env = Env::new();
    let out = env.run(env.home.path(), &["vault", "list"]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(stdout(&out), "no vaults registered\n");
}

#[test]
fn first_added_vault_becomes_the_default() {
    let env = Env::new();
    let notes = env.vault("notes");
    let out = env.run(env.home.path(), &["vault", "add", notes.to_str().unwrap()]);
    assert_eq!(stdout(&out), "registered vault `notes`\n");
    let list = stdout(&env.run(env.home.path(), &["vault", "list"]));
    assert!(
        list.starts_with("notes\t") && list.trim_end().ends_with("\tdefault"),
        "{list}"
    );
}

#[test]
fn later_vault_is_default_only_when_requested() {
    let env = Env::new();
    for (name, extra) in [("a", None), ("b", None), ("c", Some("--default"))] {
        let root = env.vault(name);
        let mut args = vec!["vault", "add", root.to_str().unwrap()];
        args.extend(extra);
        assert_eq!(env.run(env.home.path(), &args).status.code(), Some(0));
    }
    let list = stdout(&env.run(env.home.path(), &["vault", "list"]));
    let defaults: Vec<&str> = list.lines().filter(|l| l.ends_with("\tdefault")).collect();
    assert_eq!(defaults.len(), 1);
    assert!(defaults[0].starts_with("c\t"), "{list}");
}

#[test]
fn adding_a_directory_without_notrieve_fails() {
    let env = Env::new();
    let plain = env.dir("plain");
    let out = env.run(env.home.path(), &["vault", "add", plain.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains(".notrieve"), "{}", stderr(&out));
}

#[test]
fn removing_an_unknown_vault_lists_registered_names() {
    let env = Env::new();
    let notes = env.vault("notes");
    env.run(env.home.path(), &["vault", "add", notes.to_str().unwrap()]);
    let out = env.run(env.home.path(), &["vault", "remove", "diary"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("notes"), "{}", stderr(&out));
}

#[test]
fn index_outside_any_vault_exits_with_code_2() {
    let env = Env::new();
    let out = env.run(&env.dir("elsewhere"), &["index"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("notrieve init"), "{}", stderr(&out));
}

#[test]
fn usage_errors_exit_1_with_clap_message() {
    let env = Env::new();
    for args in [&["--bogus"][..], &["index", "--vault"][..]] {
        let out = env.run(env.home.path(), args);
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        assert!(stderr(&out).contains("error:"), "{}", stderr(&out));
    }
}

#[test]
fn help_and_version_exit_0() {
    let env = Env::new();
    for flag in ["--help", "--version"] {
        let out = env.run(env.home.path(), &[flag]);
        assert_eq!(out.status.code(), Some(0), "{flag}");
        assert!(!stdout(&out).is_empty());
    }
}

#[test]
fn concurrent_vault_adds_from_separate_processes_all_register() {
    let env = Env::new();
    let roots: Vec<_> = (0..6).map(|i| env.vault(&format!("n{i}"))).collect();
    let children: Vec<_> = roots
        .iter()
        .map(|root| {
            Command::new(env!("CARGO_BIN_EXE_notrieve"))
                .args(["vault", "add"])
                .arg(root)
                .env("HOME", env.home.path())
                .env("XDG_CONFIG_HOME", env.home.path().join("config"))
                .env("XDG_CACHE_HOME", env.home.path().join("cache"))
                .spawn()
                .unwrap()
        })
        .collect();
    for child in children {
        assert!(child.wait_with_output().unwrap().status.success());
    }
    let out = env.run(env.home.path(), &["vault", "list"]);
    for i in 0..6 {
        assert!(
            stdout(&out).contains(&format!("n{i}\t")),
            "{}",
            stdout(&out)
        );
    }
}

#[test]
fn index_no_wait_exits_with_code_3_while_another_process_holds_the_lock() {
    let env = Env::new();
    let notes = env.vault("notes");
    let lock = fs::File::create(notes.join(".notrieve/index.db.lock")).unwrap();
    lock.try_lock().unwrap();
    let out = env.run(&notes, &["index", "--no-wait"]);
    assert_eq!(out.status.code(), Some(3));
    assert!(
        stderr(&out).contains("another notrieve process"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn init_refuses_an_existing_notrieve_directory() {
    let env = Env::new();
    let notes = env.vault("notes");
    let out = env.run(&notes, &["init"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("already contains .notrieve"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn init_with_a_taken_name_fails_before_creating_files() {
    let env = Env::new();
    let first = env.vault("first");
    env.run(
        env.home.path(),
        &["vault", "add", first.to_str().unwrap(), "--name", "notes"],
    );
    let second = env.dir("notes");
    let out = env.run(&second, &["init"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(!second.join(".notrieve").exists());
}

#[test]
fn search_outside_any_vault_exits_with_code_2() {
    let env = Env::new();
    let out = env.run(&env.dir("elsewhere"), &["search", "税金"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(stderr(&out).contains("notrieve init"), "{}", stderr(&out));
}

#[test]
fn search_k_outside_1_to_50_is_a_usage_error() {
    let env = Env::new();
    for k in ["0", "51"] {
        let out = env.run(env.home.path(), &["search", "税金", "-k", k]);
        assert_eq!(out.status.code(), Some(1), "{k}");
        assert!(stderr(&out).contains("-k"), "{}", stderr(&out));
    }
}

#[test]
fn blank_search_query_fails_before_the_model_is_downloaded() {
    let env = Env::new();
    let notes = env.vault("notes");
    let out = env.run(&notes, &["search", "  "]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("the query is empty"),
        "{}",
        stderr(&out)
    );
    assert!(!env.home.path().join("cache/notrieve").exists());
}

#[test]
fn mcp_outside_any_vault_answers_tools_with_the_vault_error_and_keeps_stdout_to_json_rpc() {
    let env = Env::new();
    let dir = env.dir("plain");
    let (lines, status) = support::mcp_search_session(env.command(&dir, &["mcp"]), "anything");
    assert!(status.success(), "{status}");
    assert!(lines.iter().all(|l| l["jsonrpc"] == "2.0"), "{lines:?}");
    assert_eq!(lines[0]["id"], 1);
    let result = &lines.last().unwrap()["result"];
    assert_eq!(result["isError"], true, "{result}");
    let message = result["content"][0]["text"].as_str().unwrap();
    assert!(message.contains("no vault found"), "{message}");
    assert!(
        !env.home.path().join("cache/notrieve/models").exists(),
        "no model was fetched"
    );
}

fn remote_vault(env: &Env, name: &str, base: &str) -> PathBuf {
    let root = env.dir(name);
    fs::create_dir_all(root.join(".notrieve")).unwrap();
    fs::write(
        root.join(".notrieve/config.toml"),
        remote_config(base, "", ""),
    )
    .unwrap();
    root
}

#[test]
fn model_pull_for_a_remote_model_downloads_nothing() {
    let env = Env::new();
    let root = remote_vault(&env, "notes", "http://127.0.0.1:9/v1");
    let o = env.run(&root, &["model", "pull"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(
        stdout(&o).contains("model is remote (127.0.0.1); nothing to download"),
        "{}",
        stdout(&o)
    );
    assert!(!env.home.path().join("cache/notrieve").exists());
}

#[test]
fn index_with_a_remote_model_goes_through_the_api() {
    let api = FakeApi::start(Reply::Embed { dim: 8 });
    let env = Env::new();
    let root = remote_vault(&env, "notes", &api.base);
    fs::write(root.join("a.md"), "# A\nalpha\n").unwrap();
    let o = env.run(&root, &["index"]);
    assert!(o.status.success(), "{}", stderr(&o));
    assert!(stdout(&o).starts_with("indexed 1,"), "{}", stdout(&o));
    assert_eq!(api.requests().len(), 1);
}
