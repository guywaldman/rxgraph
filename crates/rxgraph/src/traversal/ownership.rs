//! Storage borrowed by Rust cursors or retained by owned plugin sessions.
use std::{ops::Deref, sync::Arc};

pub(crate) enum Owner<'a, T: ?Sized> {
    Borrowed(&'a T),
    Shared(Arc<T>),
}
impl<T: ?Sized> Deref for Owner<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match self {
            Self::Borrowed(value) => value,
            Self::Shared(value) => value,
        }
    }
}
