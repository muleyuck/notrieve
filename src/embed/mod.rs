pub mod download;
pub mod fake;
pub mod onnx;

pub use onnx::OnnxEmbedder;

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
}

impl<T: Embedder + ?Sized> TokenCount for T {
    fn document_tokens(&self, input: &str) -> Result<usize> {
        self.count_tokens(input)
    }

    fn token_ends(&self, text: &str) -> Result<Vec<usize>> {
        Embedder::token_ends(self, text)
    }
}
