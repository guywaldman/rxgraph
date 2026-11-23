use std::{
    error::Error,
    fmt,
    time::{Duration, Instant},
};

/// A traversal exceeded its configured timeout.
#[derive(Debug)]
pub struct SearchTimeout;

impl fmt::Display for SearchTimeout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("search timed out")
    }
}

impl Error for SearchTimeout {}

/// One monotonic time budget shared by every phase, worker, and pull of a search.
#[derive(Clone, Copy, Debug)]
pub(super) struct Deadline(Option<Instant>);

impl Deadline {
    pub(super) const NONE: Self = Self(None);
    pub(super) const POLL_INTERVAL: usize = 128;

    pub(super) fn start(timeout: Option<Duration>) -> Self {
        // A duration beyond the clock's range cannot expire during this run.
        Self(timeout.and_then(|duration| Instant::now().checked_add(duration)))
    }

    pub(super) fn enabled(self) -> bool {
        self.0.is_some()
    }

    #[inline]
    pub(super) fn check(self) -> anyhow::Result<()> {
        if self.0.is_some_and(|at| Instant::now() >= at) {
            return Err(SearchTimeout.into());
        }
        Ok(())
    }

    #[inline]
    pub(super) fn check_every(self, index: usize) -> anyhow::Result<()> {
        if index % Self::POLL_INTERVAL == 0 {
            self.check()?;
        }
        Ok(())
    }
}
