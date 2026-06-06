//! Example native [`Kernel`](crate::traversal::Kernel) implementations.
//!
//! These exist to show how to write a native Rust traversal kernel against the
//! public API and how to register one by name via [`inventory`](crate::inventory).
//! They are exported so external Rust consumers can use them directly or as a
//! reference for their own kernels.

pub mod kernels;
