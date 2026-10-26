//! Deterministic embedder for tests: vectors derived from a hash of the text.
use std::sync::atomic::{AtomicUsize, Ordering};

use super::Embedder;
use crate::{Error, Result};

pub struct FakeEmbedder {
    dim: usize,
    fail_marker: Option<String>,
    embedded: AtomicUsize,
}

impl FakeEmbedder {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            fail_marker: None,
            embedded: AtomicUsize::new(0),
        }
    }

    pub fn failing_on(mut self, marker: &str) -> Self {
        self.fail_marker = Some(marker.to_string());
        self
    }

    pub fn embedded_count(&self) -> usize {
        self.embedded.load(Ordering::SeqCst)
    }

    fn vector(&self, text: &str) -> Vec<f32> {
        fake_vector(text, self.dim)
    }
}

/// A deterministic unit vector of `dim` dimensions derived from `text`. The tests' fake
/// embeddings API returns the same vectors.
pub fn fake_vector(text: &str, dim: usize) -> Vec<f32> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(text.as_bytes());
    let mut bytes = vec![0u8; dim * 4];
    hasher.finalize_xof().fill(&mut bytes);
    let mut v: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b) as f32 / u32::MAX as f32 - 0.5)
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

impl Embedder for FakeEmbedder {
    fn model_id(&self) -> &str {
        "fake"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if let Some(marker) = &self.fail_marker
            && texts.iter().any(|t| t.contains(marker.as_str()))
        {
            return Err(Error::Embedding(format!("fake failure on `{marker}`")));
        }
        self.embedded.fetch_add(texts.len(), Ordering::SeqCst);
        Ok(texts
            .iter()
            .map(|t| self.vector(&format!("document:{t}")))
            .collect())
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.vector(&format!("query:{text}")))
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(text.chars().count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_deterministic_normalised_and_text_dependent() {
        let e = FakeEmbedder::new(16);
        let a = e.embed_documents(&["alpha".into(), "beta".into()]).unwrap();
        let again = e.embed_documents(&["alpha".into()]).unwrap();
        assert_eq!(a[0], again[0]);
        assert_ne!(a[0], a[1]);
        assert_eq!(a[0].len(), 16);
        assert!((a[0].iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(e.embedded_count(), 3);
    }

    #[test]
    fn query_vector_differs_from_document_vector() {
        let e = FakeEmbedder::new(8);
        assert_ne!(
            e.embed_query("alpha").unwrap(),
            e.embed_documents(&["alpha".into()]).unwrap()[0]
        );
    }

    #[test]
    fn marker_makes_the_whole_call_fail() {
        let e = FakeEmbedder::new(8).failing_on("BOOM");
        assert!(matches!(
            e.embed_documents(&["ok".into(), "a BOOM b".into()]),
            Err(Error::Embedding(_))
        ));
        assert_eq!(e.embedded_count(), 0);
    }
}
