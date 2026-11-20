//! Labs › Tidy after paste editing a watched paste in place (T8.5, D-38):
//! the tidy is Hex's edit, not the person's, so it must never reach History
//! as a correction. While an apply runs, the watch of that dictation's
//! entry holds still ([`Mark::Editing`]); once it applied, the watch starts
//! over with the tidied text as its baseline ([`Mark::Rebase`]), so a
//! correction is measured from what the field held after the tidy; an
//! apply whose edit couldn't be confirmed ends the watch with nothing
//! ([`Mark::Abandon`]), since what the field holds is unknown. An apply
//! that changed nothing (moved, refused) lets the watch go on as it was.
//!
//! The marks live in memory only, hold no more than the tidied paste and
//! the field around it, and are bounded: at most [`MAX_MARKS`], the oldest
//! dropped.

use std::sync::Mutex;

/// What the tidied field holds, for the watch to start over from.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rebase {
    /// The tidied paste, as it now stands in the field.
    pub inserted: String,
    /// Where it starts, in UTF-16 units.
    pub start: usize,
    /// The field without it.
    pub before: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Mark {
    Editing,
    Rebase(Rebase),
    Abandon,
}

/// Marks kept at once: more than the watches that can be active or queued.
pub const MAX_MARKS: usize = 16;

static MARKS: Mutex<Vec<(u64, Mark)>> = Mutex::new(Vec::new());

fn set(entry_id: u64, mark: Option<Mark>) {
    let mut marks = MARKS.lock().unwrap_or_else(|error| error.into_inner());
    marks.retain(|(id, _)| *id != entry_id);
    if let Some(mark) = mark {
        if marks.len() >= MAX_MARKS {
            marks.remove(0);
        }
        marks.push((entry_id, mark));
    }
}

/// What the watch of `entry_id` must do before its next step: `Editing`
/// stays until the apply ends; a rebase or an abandon is handed out once.
pub fn take(entry_id: u64) -> Option<Mark> {
    let mut marks = MARKS.lock().unwrap_or_else(|error| error.into_inner());
    let index = marks.iter().position(|(id, _)| *id == entry_id)?;
    if marks[index].1 == Mark::Editing {
        return Some(Mark::Editing);
    }
    Some(marks.remove(index).1)
}

/// One apply over `entry_id`'s paste. Begun before the first edit is sent;
/// dropped without an outcome (a refusal, a panic), it clears its mark.
pub struct TidyEdit {
    entry_id: u64,
    done: bool,
}

impl TidyEdit {
    pub fn begin(entry_id: u64) -> Self {
        set(entry_id, Some(Mark::Editing));
        Self {
            entry_id,
            done: false,
        }
    }

    /// The edit went in: the watch starts over from `rebase`.
    pub fn applied(mut self, rebase: Rebase) {
        set(self.entry_id, Some(Mark::Rebase(rebase)));
        self.done = true;
    }

    /// An edit was sent but couldn't be confirmed: the watch ends unkept.
    pub fn unverified(mut self) {
        set(self.entry_id, Some(Mark::Abandon));
        self.done = true;
    }
}

impl Drop for TidyEdit {
    fn drop(&mut self) {
        if !self.done {
            set(self.entry_id, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rebase() -> Rebase {
        Rebase {
            inserted: "So send it to Kit".into(),
            start: 0,
            before: String::new(),
        }
    }

    #[test]
    fn editing_holds_until_the_apply_ends_and_a_rebase_is_taken_once() {
        let id = 9_100_001;
        let edit = TidyEdit::begin(id);
        assert_eq!(take(id), Some(Mark::Editing));
        assert_eq!(take(id), Some(Mark::Editing));
        edit.applied(rebase());
        assert_eq!(take(id), Some(Mark::Rebase(rebase())));
        assert_eq!(take(id), None);
    }

    #[test]
    fn an_apply_that_changed_nothing_leaves_no_mark() {
        let id = 9_100_002;
        drop(TidyEdit::begin(id));
        assert_eq!(take(id), None);
        TidyEdit::begin(id).unverified();
        assert_eq!(take(id), Some(Mark::Abandon));
        assert_eq!(take(id), None);
    }

    #[test]
    fn marks_are_bounded() {
        let base = 9_200_000;
        for id in base..base + MAX_MARKS as u64 + 4 {
            TidyEdit::begin(id).applied(rebase());
        }
        assert!(MARKS.lock().unwrap().len() <= MAX_MARKS);
        assert_eq!(take(base), None, "the oldest was dropped");
    }
}
