//! An OpenAI-compatible `/embeddings` endpoint on 127.0.0.1. Replies come from a queue, then
//! from a default; every request is recorded.
use std::collections::{HashMap, VecDeque};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

#[derive(Clone)]
pub enum Reply {
    /// 200 with one `dim`-dimensional `fake_vector` per input, in input order.
    Embed { dim: usize },
    /// Like `Embed`, with `data` in reverse index order.
    EmbedReversed { dim: usize },
    Status {
        code: u16,
        headers: Vec<(String, String)>,
        body: String,
    },
    /// 200 with this body as it is.
    Body(String),
    /// Declares a body of `declared_length` bytes, sends `body` (shorter) and closes.
    Truncated {
        code: u16,
        declared_length: usize,
        body: String,
    },
    /// The reply, sent after a pause.
    Delayed(Duration, Box<Reply>),
    /// Chosen from the request body.
    By(Arc<dyn Fn(&Value) -> Reply + Send + Sync>),
}

impl Reply {
    pub fn status(code: u16) -> Self {
        Reply::Status {
            code,
            headers: Vec::new(),
            body: format!("{{\"error\":\"status {code}\"}}"),
        }
    }

    pub fn by(choose: impl Fn(&Value) -> Reply + Send + Sync + 'static) -> Self {
        Reply::By(Arc::new(choose))
    }
}

#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    /// Names lowercased.
    pub headers: HashMap<String, String>,
    pub body: Value,
}

impl Recorded {
    pub fn inputs(&self) -> Vec<String> {
        inputs(&self.body)
    }
}

pub fn inputs(body: &Value) -> Vec<String> {
    body["input"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

struct State {
    queue: VecDeque<Reply>,
    default: Reply,
    requests: Vec<Recorded>,
}

pub struct FakeApi {
    /// `http://127.0.0.1:<port>/v1`
    pub base: String,
    state: Arc<Mutex<State>>,
}

impl FakeApi {
    pub fn start(default: Reply) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}/v1", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State {
            queue: VecDeque::new(),
            default,
            requests: Vec::new(),
        }));
        let shared = Arc::clone(&state);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let shared = Arc::clone(&shared);
                // One thread per connection, so a delayed reply does not hold up the next request.
                std::thread::spawn(move || serve(stream, &shared));
            }
        });
        Self { base, state }
    }

    /// Queued replies are used first, in order.
    pub fn push(&self, reply: Reply) {
        self.state.lock().unwrap().queue.push_back(reply);
    }

    pub fn set_default(&self, reply: Reply) {
        self.state.lock().unwrap().default = reply;
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.state.lock().unwrap().requests.clone()
    }

    /// Every input sent so far, in order.
    pub fn inputs(&self) -> Vec<String> {
        self.requests().iter().flat_map(Recorded::inputs).collect()
    }
}

fn serve(mut stream: TcpStream, state: &Mutex<State>) {
    let Some(recorded) = read_request(&mut stream) else {
        return;
    };
    let reply = {
        let mut state = state.lock().unwrap();
        state.requests.push(recorded.clone());
        match state.queue.pop_front() {
            Some(reply) => reply,
            None => state.default.clone(),
        }
    };
    respond(&mut stream, reply, &recorded.body);
}

fn read_request(stream: &mut TcpStream) -> Option<Recorded> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let headers: HashMap<String, String> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    let length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    while buf.len() < head_end + length {
        let n = stream.read(&mut chunk).ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = serde_json::from_slice(&buf[head_end..]).unwrap_or(Value::Null);
    Some(Recorded {
        method,
        path,
        headers,
        body,
    })
}

fn respond(stream: &mut TcpStream, reply: Reply, request: &Value) {
    let (code, headers, body) = match reply {
        Reply::Embed { dim } => (200, Vec::new(), embeddings(request, dim, false)),
        Reply::EmbedReversed { dim } => (200, Vec::new(), embeddings(request, dim, true)),
        Reply::Status {
            code,
            headers,
            body,
        } => (code, headers, body),
        Reply::Body(body) => (200, Vec::new(), body),
        Reply::Truncated {
            code,
            declared_length,
            body,
        } => {
            let head = format!(
                "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {declared_length}\r\nConnection: close\r\n\r\n"
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(body.as_bytes());
            return;
        }
        Reply::Delayed(pause, reply) => {
            std::thread::sleep(pause);
            return respond(stream, *reply, request);
        }
        Reply::By(choose) => return respond(stream, choose(request), request),
    };
    let mut head = format!(
        "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
}

fn embeddings(request: &Value, dim: usize, reversed: bool) -> String {
    let mut data: Vec<Value> = inputs(request)
        .iter()
        .enumerate()
        .map(|(i, input)| json!({"object": "embedding", "index": i, "embedding": fake_vector(input, dim)}))
        .collect();
    if reversed {
        data.reverse();
    }
    json!({"object": "list", "data": data, "model": request["model"]}).to_string()
}

/// The vector the fake API returns for `input`: deterministic and of unit length.
pub fn fake_vector(input: &str, dim: usize) -> Vec<f32> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(input.as_bytes());
    let mut bytes = vec![0u8; dim * 4];
    hasher.finalize_xof().fill(&mut bytes);
    let (chunks, _) = bytes.as_chunks::<4>();
    let mut v: Vec<f32> = chunks
        .iter()
        .map(|b| u32::from_le_bytes(*b) as f32 / u32::MAX as f32 - 0.5)
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}
