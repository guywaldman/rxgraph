//! Progress reporting for long-running searches.
//! When enabled, reports live search counters to stderr.
//!
//! When the FD is a terminal (TTY), it has an interactive spinner for a nice UX.
//! Otherwise, it logs.
//!
//! Search has no known total upfront, so no percentage or ETA is shown.

use std::{
    io::{IsTerminal, Write},
    time::{Duration, Instant},
};

use crate::traversal::SearchStats;

/// Spinner refresh interval when attached to a terminal.
const SPINNER_INTERVAL: Duration = Duration::from_millis(100);
/// Coarser interval for plain log lines so non-interactive output stays readable.
const LOG_INTERVAL: Duration = Duration::from_secs(5);

// Keyframes for the spinner. Flash throwback :)
const SPINNER: [char; 8] = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷'];

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Style {
    /// In-place spinner for TTY.
    Spinner,
    /// Newline-terminated log lines for non-TTY.
    Log,
}

/// Progress reporter. The disabled variant is a no-op with no overhead.
pub(crate) enum Progress {
    Disabled,
    Enabled {
        style: Style,
        interval: Duration,
        start: Instant,
        last_tick: Instant,
        frame: usize,
    },
}

impl Progress {
    /// Creates a reporter. Disabled when `enabled` is false. When enabled, the
    /// output *style* depends on whether stderr is a terminal, but progress is
    /// always reported (a non-TTY simply gets plain log lines instead of a
    /// spinner).
    pub(crate) fn new(enabled: bool) -> Self {
        if !enabled {
            return Self::Disabled;
        }
        let (style, interval) = if std::io::stderr().is_terminal() {
            (Style::Spinner, SPINNER_INTERVAL)
        } else {
            (Style::Log, LOG_INTERVAL)
        };
        let now = Instant::now();
        Self::Enabled {
            style,
            interval,
            start: now,
            // Force the first tick to render immediately.
            last_tick: now - interval,
            frame: 0,
        }
    }

    /// Renders a frame if the throttle interval has elapsed. Cheap to call often.
    pub(crate) fn tick(&mut self, stats: &SearchStats) {
        let Self::Enabled {
            style,
            interval,
            start,
            last_tick,
            frame,
        } = self
        else {
            return;
        };
        let now = Instant::now();
        if now.duration_since(*last_tick) < *interval {
            return;
        }
        *last_tick = now;
        let body = render_body(now.duration_since(*start), stats);
        let mut err = std::io::stderr().lock();
        match style {
            // \r overwrites the line in place; no newline so it stays a live bar.
            Style::Spinner => {
                let _ = write!(err, "\r{} {body}", SPINNER[*frame % SPINNER.len()]);
            }
            // Newline-terminated so each update is a discrete log record.
            Style::Log => {
                let _ = writeln!(err, "rxgraph search: {body}");
            }
        }
        let _ = err.flush();
        *frame += 1;
    }

    /// Finalizes output once the search completes.
    pub(crate) fn finish(&self, stats: &SearchStats) {
        let Self::Enabled { style, start, .. } = self else {
            return;
        };
        let body = render_body(start.elapsed(), stats);
        let mut err = std::io::stderr().lock();
        match style {
            // Erase the live bar, then print a final summary line.
            Style::Spinner => {
                let _ = write!(err, "\r\x1b[2K");
                let _ = writeln!(err, "✓ {body}");
            }
            Style::Log => {
                let _ = writeln!(err, "rxgraph search done: {body}");
            }
        }
        let _ = err.flush();
    }
}

/// Formats the counter body shared by both styles. Pure (no I/O) for testability.
fn render_body(elapsed: Duration, stats: &SearchStats) -> String {
    format!(
        "{:.1}s  edges {}  accepted {}  paths {}",
        elapsed.as_secs_f64(),
        stats.evaluated_edges,
        stats.accepted_edges,
        stats.stopped_paths,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_progress_is_noop() {
        let mut p = Progress::new(false);
        p.tick(&SearchStats::default());
        p.finish(&SearchStats::default());
        assert!(matches!(p, Progress::Disabled));
    }

    #[test]
    fn render_body_includes_counters() {
        let stats = SearchStats {
            evaluated_edges: 12,
            accepted_edges: 5,
            stopped_paths: 2,
            ..SearchStats::default()
        };
        let body = render_body(Duration::from_millis(1500), &stats);
        assert!(body.contains("edges 12"));
        assert!(body.contains("accepted 5"));
        assert!(body.contains("paths 2"));
        assert!(body.contains("1.5s"));
    }
}
