//! Escape and Return presses the shortcut listener has seen. Each reader
//! compares the count with the one when it started: Labs › YOLO mode's send
//! (an Escape after the release skips its Return) and Labs › Learn from
//! corrections (a Return may be a chat send). Neither lab knows the other.

use std::sync::atomic::{AtomicU64, Ordering};

pub static ESCAPES: AtomicU64 = AtomicU64::new(0);
pub static RETURNS: AtomicU64 = AtomicU64::new(0);

/// Counts `event` if it is an Escape or Return press.
pub fn note(event: crate::suppression::InputEvent) {
    if event.is_escape_down() {
        ESCAPES.fetch_add(1, Ordering::AcqRel);
    }
    if event.is_return_down() {
        RETURNS.fetch_add(1, Ordering::AcqRel);
    }
}

pub fn returns() -> u64 {
    RETURNS.load(Ordering::Acquire)
}
