pub mod download;
pub mod fake;
pub mod onnx;
pub mod remote;

pub use onnx::OnnxEmbedder;
pub use remote::RemoteEmbedder;

use std::sync::{Mutex, MutexGuard};
use std::time::Instant;

use crate::Result;
use crate::markdown::chunk::TokenCount;

pub trait Embedder: Send + Sync {
    fn model_id(&self) -> &str;
    fn dim(&self) -> usize;
    /// Texts without prefix; the embedder adds the document prefix once.
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    /// Text without prefix; the embedder adds the query prefix once.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    /// Tokens of the document input (prefix and special tokens included), never truncated.
    fn count_tokens(&self, text: &str) -> Result<usize>;
    /// Byte offsets in `text` where its tokens end, ascending and ending at `text.len()`.
    /// The default treats every character as a token.
    fn token_ends(&self, text: &str) -> Result<Vec<usize>> {
        Ok(text
            .char_indices()
            .skip(1)
            .map(|(o, _)| o)
            .chain(std::iter::once(text.len()))
            .collect())
    }

    /// `embed_documents` that gives up at `deadline`. Local models ignore it.
    fn embed_documents_within(
        &self,
        texts: &[String],
        _deadline: Option<Instant>,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_documents(texts)
    }

    /// `embed_query` that gives up at `deadline`. Local models ignore it.
    fn embed_query_within(&self, text: &str, _deadline: Option<Instant>) -> Result<Vec<f32>> {
        self.embed_query(text)
    }

    /// The host note text is sent to, when it leaves this machine.
    fn remote_host(&self) -> Option<&str> {
        None
    }

    /// The model runs behind an embedding API (even one on this machine): `dim()` is 0 and the
    /// dimension comes from the stored vectors, and time limits are deadlines (remote spec §4, §5).
    fn is_remote(&self) -> bool {
        false
    }
}

impl<T: Embedder + ?Sized> TokenCount for T {
    fn document_tokens(&self, input: &str) -> Result<usize> {
        self.count_tokens(input)
    }

    fn token_ends(&self, text: &str) -> Result<Vec<usize>> {
        Embedder::token_ends(self, text)
    }
}

/// Takes a model lock even after a panic poisoned it. Spec §6 requires search to continue after
/// the update thread panics; with the poison kept, every later query on the shared model would
/// fail. notrieve treats the model as reusable after a failed call.
pub fn lock_unpoisoned<T>(lock: &Mutex<T>) -> MutexGuard<'_, T> {
    lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::lock_unpoisoned;
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_lock_poisoned_by_a_panic_is_still_usable() {
        let model = Arc::new(Mutex::new(41));
        let poisoner = Arc::clone(&model);
        let _ = std::thread::spawn(move || {
            let _guard = poisoner.lock().unwrap();
            panic!("inference panicked");
        })
        .join();
        assert!(model.is_poisoned());
        let mut guard = lock_unpoisoned(&model);
        *guard += 1;
        assert_eq!(*guard, 42);
    }
}
