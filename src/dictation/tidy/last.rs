//! What Tidy after paste knows about the dictations as they commit: how many
//! have (so a review of an older one closes when a new one lands) and
//! whether the newest was pasted or only copied. It keeps an entry id and a
//! flag, never text; History holds the words.

use std::sync::{Mutex, OnceLock};

use super::session::Delivery;
use crate::history::HistoryKind;

#[derive(Clone, Copy, Debug, Default)]
struct Last {
    /// Dictations committed in this process.
    generation: u64,
    /// The newest one's History entry when it only reached the clipboard.
    copied: Option<u64>,
}

static LAST: Mutex<Last> = Mutex::new(Last {
    generation: 0,
    copied: None,
});

/// Called with the new generation after each commit, from the output
/// thread: the review capsule closes a review of an older dictation.
type Committed = Box<dyn Fn(u64) + Send + Sync>;
static ON_COMMIT: OnceLock<Committed> = OnceLock::new();

/// A dictation's output committed: pasted, copied, or printed, with its
/// History entry when one was recorded.
pub(crate) fn committed(entry: Option<u64>, copied: bool) {
    let generation = {
        let mut last = LAST.lock().unwrap_or_else(|error| error.into_inner());
        last.generation += 1;
        last.copied = entry.filter(|_| copied);
        last.generation
    };
    if let Some(on_commit) = ON_COMMIT.get() {
        on_commit(generation);
    }
}

/// How many dictations have committed in this process.
pub(crate) fn generation() -> u64 {
    LAST.lock()
        .unwrap_or_else(|error| error.into_inner())
        .generation
}

/// How History entry `id` of `kind` reached its app: a printed one never
/// reached a field, and the newest one may have only been copied.
pub(crate) fn delivery(id: u64, kind: HistoryKind) -> Delivery {
    let copied = LAST
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .copied
        == Some(id);
    if copied || kind == HistoryKind::Printed {
        Delivery::Copied
    } else {
        Delivery::Pasted
    }
}

/// Installs the one commit observer (the macOS review capsule).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn on_commit(observer: impl Fn(u64) + Send + Sync + 'static) {
    let _ = ON_COMMIT.set(Box::new(observer));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test owns the process-wide marker, so the steps stay in order.
    #[test]
    fn commits_count_and_remember_a_copy_only_dictation() {
        let before = generation();
        committed(Some(41), true);
        assert_eq!(generation(), before + 1);
        assert_eq!(delivery(41, HistoryKind::Dictation), Delivery::Copied);
        committed(Some(42), false);
        assert_eq!(generation(), before + 2);
        assert_eq!(delivery(41, HistoryKind::Dictation), Delivery::Pasted);
        assert_eq!(delivery(42, HistoryKind::Dictation), Delivery::Pasted);
        assert_eq!(delivery(42, HistoryKind::Printed), Delivery::Copied);
    }
}
