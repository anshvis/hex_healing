//! Labs › Tidy after paste's apply step (T8.4): the accepted edits go into
//! the field Hex pasted the dictation into, as one undoable edit of exactly
//! the pasted range, and nothing else in the field changes.
//!
//! The pure part works over the field's UTF-16 units: [`locate`] finds the
//! pasted text by the anchor's hash and length (at the expected range, and
//! only when it occurs exactly once), [`plan`] builds the tidied replacement
//! and the whole field it should leave, and [`check`] compares a re-read
//! with that. [`apply`] runs the steps against a [`Target`]: the secure-field
//! check first, the frontmost app, `AXSelectedText` when the field takes it,
//! else Hex's paste over the selected range, then the re-read. On macOS the
//! target is the anchored Accessibility element ([`apply_to_last`]).
//!
//! Nothing here keeps or logs the text: the anchor has a SHA-256 and a
//! length, and the log names the outcome, the method, and counts.

use std::ops::Range;

use sha2::{Digest, Sha256};

use super::Suggestion;

/// The longest field Tidy reads whole, in UTF-16 units. A larger one is
/// declined rather than searched through a window, where the pasted text
/// could not be proven unique.
pub(crate) const MAX_FIELD_UNITS: usize = 32 * 1024;

/// What the anchor knows of the text a paste inserted: its SHA-256, a
/// rolling hash to find candidates quickly, and its length in UTF-16 units.
/// Never the text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Inserted {
    sha256: [u8; 32],
    rolling: u64,
    utf16_len: usize,
}

impl Inserted {
    pub(crate) fn of(text: &str) -> Self {
        let units: Vec<u16> = text.encode_utf16().collect();
        Self {
            sha256: sha256(text),
            rolling: rolling(&units),
            utf16_len: units.len(),
        }
    }

    #[cfg(test)]
    pub(crate) fn utf16_len(&self) -> usize {
        self.utf16_len
    }

    /// Whether `units` are exactly the inserted text.
    fn matches(&self, units: &[u16]) -> bool {
        units.len() == self.utf16_len
            && String::from_utf16(units).is_ok_and(|text| sha256(&text) == self.sha256)
    }
}

fn sha256(text: &str) -> [u8; 32] {
    let digest = Sha256::digest(text.as_bytes());
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// A polynomial hash over UTF-16 units, rolled one unit at a time by
/// [`locate`].
const BASE: u64 = 0x0000_0100_0000_01b3;

fn rolling(units: &[u16]) -> u64 {
    units.iter().fold(0_u64, |hash, unit| {
        hash.wrapping_mul(BASE).wrapping_add(u64::from(*unit))
    })
}

/// Where the pasted text is in the field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Located {
    At(Range<usize>),
    /// Not there: edited, deleted, or never landed.
    Missing,
    /// There more than once: Hex can't tell which copy it pasted.
    Duplicated,
}

/// Finds the inserted text in `field` (UTF-16 units). The expected range is
/// tried first; either way the text must occur exactly once (overlapping
/// occurrences count), because without the text itself Hex can't tell its
/// copy from one the person typed (D-75).
pub(crate) fn locate(field: &[u16], inserted: &Inserted, expected_start: Option<usize>) -> Located {
    let length = inserted.utf16_len;
    if length == 0 || length > field.len() {
        return Located::Missing;
    }
    let at = |start: usize| inserted.matches(&field[start..start + length]);
    let expected = expected_start.filter(|start| start + length <= field.len() && at(*start));
    let mut found = expected;
    // B^(length - 1), to roll the oldest unit out of the window.
    let top = (1..length).fold(1_u64, |power, _| power.wrapping_mul(BASE));
    let mut hash = rolling(&field[..length]);
    for start in 0..=field.len() - length {
        if start > 0 {
            hash = hash
                .wrapping_sub(u64::from(field[start - 1]).wrapping_mul(top))
                .wrapping_mul(BASE)
                .wrapping_add(u64::from(field[start + length - 1]));
        }
        if hash != inserted.rolling || Some(start) == expected || !at(start) {
            continue;
        }
        if found.is_some() {
            return Located::Duplicated;
        }
        found = Some(start);
    }
    found.map_or(Located::Missing, |start| Located::At(start..start + length))
}

/// What applying the edits needs: the pasted range in the field, what
/// replaces it, and the whole field afterwards.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Plan {
    pub(crate) range: Range<usize>,
    pub(crate) replacement: String,
    pub(crate) after: Vec<u16>,
}

/// Why no plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoPlan {
    /// The pasted text isn't in the field exactly once.
    Moved,
    /// The pasted text isn't the dictation's (with the paste's own leading
    /// space and first-letter case), so its edits can't be placed.
    Differs,
}

/// The plan for applying `edits` (over `text`, the dictation's final text)
/// to the pasted range of `field`.
pub(crate) fn plan(
    field: &[u16],
    inserted: &Inserted,
    expected_start: Option<usize>,
    text: &str,
    edits: &[Suggestion],
) -> Result<Plan, NoPlan> {
    let Located::At(range) = locate(field, inserted, expected_start) else {
        return Err(NoPlan::Moved);
    };
    // `locate` matched a SHA-256 over valid UTF-16, so this decodes.
    let pasted = String::from_utf16(&field[range.clone()]).map_err(|_| NoPlan::Moved)?;
    let shift = dictation_offset(&pasted, text).ok_or(NoPlan::Differs)?;
    let scalars = text.chars().count();
    if edits.iter().any(|edit| edit.range.scalar.end > scalars) {
        return Err(NoPlan::Differs);
    }
    let shifted: Vec<Suggestion> = edits
        .iter()
        .map(|edit| {
            let mut edit = edit.clone();
            edit.range.scalar = edit.range.scalar.start + shift..edit.range.scalar.end + shift;
            edit
        })
        .collect();
    let replacement = super::apply(&pasted, &shifted);
    let mut after = Vec::with_capacity(field.len());
    after.extend_from_slice(&field[..range.start]);
    after.extend(replacement.encode_utf16());
    after.extend_from_slice(&field[range.end..]);
    Ok(Plan {
        range,
        replacement,
        after,
    })
}

/// Where the dictation's `text` starts in what the paste inserted, in
/// scalars: the paste may add leading whitespace (a continuation's join,
/// Smart spacing) and change the first letter's case (a continuation after
/// a sentence, or into one). Anything else means the edits can't be placed.
fn dictation_offset(pasted: &str, text: &str) -> Option<usize> {
    let pasted: Vec<char> = pasted.chars().collect();
    let text: Vec<char> = text.chars().collect();
    let shift = pasted.len().checked_sub(text.len())?;
    if !pasted[..shift]
        .iter()
        .all(|character| character.is_whitespace())
    {
        return None;
    }
    let mut recased = 0;
    for (pasted, text) in pasted[shift..].iter().zip(&text) {
        if pasted == text {
            continue;
        }
        if !pasted.to_lowercase().eq(text.to_lowercase()) {
            return None;
        }
        recased += 1;
        if recased > 1 {
            return None;
        }
    }
    Some(shift)
}

/// What a re-read of the field shows after an edit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Check {
    /// Exactly the plan: the range holds the replacement and every unit
    /// outside it is as before.
    Applied,
    /// Nothing changed (yet).
    Unchanged,
    /// Something else.
    Other,
}

pub(crate) fn check(before: &[u16], plan: &Plan, now: &[u16]) -> Check {
    if now == plan.after.as_slice() {
        Check::Applied
    } else if now == before {
        Check::Unchanged
    } else {
        Check::Other
    }
}

/// How an applied edit went in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Method {
    /// `AXSelectedTextRange`, then `AXSelectedText`: one undo step.
    SelectedText,
    /// The range selected, then Hex's paste over it (the field ignored or
    /// refused `AXSelectedText`): also one undo step.
    Paste,
}

/// What applying did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Outcome {
    Applied(Method),
    /// The pasted text changed (or is there twice) since the paste; the
    /// field is untouched.
    Moved,
    Unsupported(Why),
}

/// Why Tidy couldn't edit the field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Why {
    /// No anchor: neither lab kept one, or the next paste replaced it.
    NoAnchor,
    /// The anchor is over 10 minutes old.
    Expired,
    /// The anchor belongs to another dictation than the review's.
    OtherDictation,
    /// A secure (password) field: never read or edited.
    SecureField,
    /// The app Hex pasted into isn't frontmost.
    OtherApp,
    /// The paste fallback needs the field focused, and it isn't.
    OtherField,
    /// The field can't be read (gone, no string value, timed out).
    Unreadable,
    /// The field holds more than [`MAX_FIELD_UNITS`].
    TooLarge,
    /// The pasted text isn't the dictation's, so its edits can't be placed.
    Differs,
    /// The field takes neither `AXSelectedText` nor a selected range to
    /// paste over.
    NotEditable,
    /// An edit was sent but the re-read doesn't show exactly the plan.
    Unverified,
    /// Labs › Tidy after paste was turned off before the edit.
    Disabled,
    /// No Accessibility on this OS.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    NotOnThisOs,
}

impl Outcome {
    /// The outcome's name for the log.
    pub(crate) fn kind(self) -> &'static str {
        match self {
            Self::Applied(_) => "applied",
            Self::Moved => "moved",
            Self::Unsupported(_) => "unsupported",
        }
    }
}

/// The field an apply edits, so the steps can run against a fake.
pub(crate) trait Target {
    /// Whether it is a secure (password) field. Asked before anything else.
    fn is_secure(&mut self) -> Result<bool, Why>;
    /// Whether the app Hex pasted into is frontmost, waiting briefly for it
    /// after the capsule hands the keys back.
    fn owner_in_front(&mut self) -> bool;
    /// The whole value, in UTF-16 units.
    fn read(&mut self) -> Result<Vec<u16>, Why>;
    /// Whether both `AXSelectedTextRange` and `AXSelectedText` are settable.
    fn can_set_text(&mut self) -> bool;
    fn select(&mut self, range: Range<usize>) -> bool;
    fn set_selected_text(&mut self, text: &str) -> bool;
    /// Whether the field has the keyboard focus, as a paste needs.
    fn is_focused(&mut self) -> bool;
    /// Whether the app Hex pasted into is frontmost now (no waiting).
    fn is_frontmost(&mut self) -> bool;
    /// Starts sending an edit: takes the delivery lock, held until
    /// [`Self::end_edit`], so no dictation paste goes in meanwhile, and
    /// refuses when a dictation paste started since the anchor was taken
    /// ([`Why::OtherDictation`]) or the lab is off ([`Why::Disabled`]).
    fn begin_edit(&mut self) -> Result<(), Why>;
    /// Releases the delivery lock.
    fn end_edit(&mut self);
    /// Pastes over the selection through Hex's paste path.
    fn paste(&mut self, text: &str) -> bool;
    /// Waits between re-reads, for the app to apply an edit.
    fn wait(&mut self);
}

/// How many times an edit is re-read before it counts as unchanged.
const READ_BACKS: usize = 10;

/// Applies `edits` (over `text`, the dictation's final text) to the pasted
/// text `inserted` names in `target`'s field: the secure check first, then
/// the frontmost app, the bounded read and [`plan`]; `AXSelectedText` when
/// the field takes it, else (or when it ignored it) Hex's paste over the
/// selected range; then the re-read. Nothing is sent to a field the plan
/// can't be made for.
pub(crate) fn apply(
    target: &mut impl Target,
    inserted: &Inserted,
    expected_start: Option<usize>,
    text: &str,
    edits: &[Suggestion],
) -> Outcome {
    match target.is_secure() {
        Ok(false) => {}
        Ok(true) => return Outcome::Unsupported(Why::SecureField),
        Err(why) => return Outcome::Unsupported(why),
    }
    if !target.owner_in_front() {
        return Outcome::Unsupported(Why::OtherApp);
    }
    let before = match target.read() {
        Ok(units) => units,
        Err(why) => return Outcome::Unsupported(why),
    };
    let plan = match plan(&before, inserted, expected_start, text, edits) {
        Ok(plan) => plan,
        Err(NoPlan::Moved) => return Outcome::Moved,
        Err(NoPlan::Differs) => return Outcome::Unsupported(Why::Differs),
    };
    // The delivery lock is held from the first edit through the read-backs
    // and any fallback, so no dictation paste lands on a range Tidy has
    // selected or is still checking (D-79).
    if let Err(why) = target.begin_edit() {
        return Outcome::Unsupported(why);
    }
    let outcome = edit(
        target,
        &before,
        &plan,
        inserted,
        expected_start,
        text,
        edits,
    );
    target.end_edit();
    outcome
}

/// The edit itself, under the delivery lock: `AXSelectedText` when the
/// field takes it, else (or when it ignored it) the paste fallback.
fn edit(
    target: &mut impl Target,
    before: &[u16],
    plan: &Plan,
    inserted: &Inserted,
    expected_start: Option<usize>,
    text: &str,
    edits: &[Suggestion],
) -> Outcome {
    if target.can_set_text() {
        let selected = target.select(plan.range.clone());
        if selected && target.set_selected_text(&plan.replacement) {
            match read_back(target, before, plan) {
                Check::Applied => return Outcome::Applied(Method::SelectedText),
                Check::Other => return Outcome::Unsupported(Why::Unverified),
                // Accepted but ignored (some web views): paste instead.
                Check::Unchanged => {}
            }
        }
    }
    match paste_fallback(target, before, plan, inserted, expected_start, text, edits) {
        Fallback::Pasted(before, plan) => match read_back(target, &before, &plan) {
            Check::Applied => Outcome::Applied(Method::Paste),
            Check::Unchanged | Check::Other => Outcome::Unsupported(Why::Unverified),
        },
        Fallback::LandedLate => Outcome::Applied(Method::SelectedText),
        Fallback::Refused(outcome, range) => {
            // Nothing went in, but the pasted range may still be selected:
            // collapse it to its end, so the next dictation's paste appends
            // rather than replacing it.
            if let Some(range) = range {
                target.select(range.end..range.end);
            }
            outcome
        }
    }
}

/// What the paste fallback did.
enum Fallback {
    /// Pasted, against this read of the field and this plan.
    Pasted(Vec<u16>, Plan),
    /// The ignored `AXSelectedText` landed late after all.
    LandedLate,
    /// Nothing was pasted, for this outcome; the range Tidy may have left
    /// selected, if it is still where the field says.
    Refused(Outcome, Option<Range<usize>>),
}

/// The paste fallback: the field is read again, since the person may have
/// typed during the read-backs, and the range is placed from that read;
/// then, right before the paste (which goes to whatever has the keys), the
/// app must still be frontmost with the anchored field focused.
fn paste_fallback(
    target: &mut impl Target,
    before: &[u16],
    plan: &Plan,
    inserted: &Inserted,
    expected_start: Option<usize>,
    text: &str,
    edits: &[Suggestion],
) -> Fallback {
    let now = match target.read() {
        Ok(now) => now,
        Err(why) => return Fallback::Refused(Outcome::Unsupported(why), None),
    };
    if now == plan.after {
        return Fallback::LandedLate;
    }
    let plan = if now == before {
        plan.clone()
    } else {
        match self::plan(&now, inserted, expected_start, text, edits) {
            Ok(plan) => plan,
            Err(NoPlan::Moved) => return Fallback::Refused(Outcome::Moved, None),
            Err(NoPlan::Differs) => {
                return Fallback::Refused(Outcome::Unsupported(Why::Differs), None);
            }
        }
    };
    let refused = |why| Fallback::Refused(Outcome::Unsupported(why), Some(plan.range.clone()));
    if !target.select(plan.range.clone()) {
        return refused(Why::NotEditable);
    }
    // Checked last, after the read and the selection: the keys go wherever
    // the focus is when the paste is sent.
    if !target.is_frontmost() {
        return refused(Why::OtherApp);
    }
    if !target.is_focused() {
        return refused(Why::OtherField);
    }
    if !target.paste(&plan.replacement) {
        return refused(Why::NotEditable);
    }
    Fallback::Pasted(now, plan)
}

/// Re-reads until the field shows the plan or something else, or the
/// re-reads run out with it unchanged.
fn read_back(target: &mut impl Target, before: &[u16], plan: &Plan) -> Check {
    for attempt in 1..=READ_BACKS {
        let Ok(now) = target.read() else {
            return Check::Other;
        };
        match check(before, plan, &now) {
            Check::Unchanged if attempt < READ_BACKS => target.wait(),
            decided => return decided,
        }
    }
    Check::Unchanged
}

/// Applies `session`'s accepted edits to the field its dictation was
/// pasted into, through the anchor the paste left for that dictation (the
/// same History entry and commit). An anchor the apply didn't use up goes
/// back for another try; one that applied, or whose edit couldn't be
/// confirmed, is spent.
#[cfg(target_os = "macos")]
pub(crate) fn apply_to_last(session: &super::session::TidySession) -> Outcome {
    let edits = session.accepted_suggestions();
    if !crate::labs::tidy_after_paste() {
        let outcome = Outcome::Unsupported(Why::Disabled);
        log(outcome, edits.len());
        return outcome;
    }
    // Noted before the take: a paste starting after this is caught.
    let pastes = crate::dictation::paste_anchor::pastes();
    let outcome = match crate::dictation::paste_anchor::take_for(
        session.dictation_id,
        session.generation,
        std::time::Instant::now(),
    ) {
        Err(why) => Outcome::Unsupported(why),
        Ok(anchor) => {
            let outcome = apply(
                &mut ax::AxTarget::new(&anchor, pastes),
                &anchor.inserted,
                anchor.expected_start,
                &session.text,
                &edits,
            );
            if !matches!(
                outcome,
                Outcome::Applied(_) | Outcome::Unsupported(Why::Unverified)
            ) {
                crate::dictation::paste_anchor::restore(anchor, pastes);
            }
            outcome
        }
    };
    log(outcome, edits.len());
    outcome
}

/// Linux and Windows have no tidy review (D-70) and no Accessibility edit.
#[cfg(not(target_os = "macos"))]
pub(crate) fn apply_to_last(session: &super::session::TidySession) -> Outcome {
    let outcome = Outcome::Unsupported(Why::NotOnThisOs);
    log(outcome, session.accepted_count());
    outcome
}

/// The outcome, its method or reason, and the edit count; never text.
fn log(outcome: Outcome, edits: usize) {
    let method = match outcome {
        Outcome::Applied(Method::SelectedText) => "ax-selected-text",
        Outcome::Applied(Method::Paste) => "paste-fallback",
        _ => "none",
    };
    let why = match outcome {
        Outcome::Unsupported(why) => format!("{why:?}"),
        _ => String::new(),
    };
    tracing::info!(outcome = outcome.kind(), method, why, edits, "tidy apply");
}

/// The anchored Accessibility element as an apply's [`Target`].
#[cfg(target_os = "macos")]
mod ax {
    use std::ops::Range;
    use std::time::{Duration, Instant};

    use std::sync::MutexGuard;

    use super::{MAX_FIELD_UNITS, Target, Why};
    use crate::accessibility::{ReadError, edit};
    use crate::dictation::paste_anchor::{self, PasteAnchor};

    /// How long the app Hex pasted into has to be frontmost again after the
    /// capsule hands the keys back, and how often that is asked.
    const FRONT_WAIT: Duration = Duration::from_millis(600);
    const FRONT_POLL: Duration = Duration::from_millis(20);
    /// Between re-reads of an edit.
    const READ_BACK_PAUSE: Duration = Duration::from_millis(50);

    pub(super) struct AxTarget<'a> {
        anchor: &'a PasteAnchor,
        /// Dictation pastes started before the anchor was taken.
        pastes: u64,
        delivery: Option<MutexGuard<'static, ()>>,
    }

    impl<'a> AxTarget<'a> {
        pub(super) fn new(anchor: &'a PasteAnchor, pastes: u64) -> Self {
            Self {
                anchor,
                pastes,
                delivery: None,
            }
        }
    }

    fn why(error: ReadError) -> Why {
        match error {
            ReadError::Secure => Why::SecureField,
            ReadError::TooLarge => Why::TooLarge,
            ReadError::ElementGone | ReadError::Timeout | ReadError::Failed => Why::Unreadable,
        }
    }

    impl Target for AxTarget<'_> {
        fn is_secure(&mut self) -> Result<bool, Why> {
            edit::is_secure(&self.anchor.element).map_err(why)
        }

        fn owner_in_front(&mut self) -> bool {
            let deadline = Instant::now() + FRONT_WAIT;
            loop {
                if crate::dictation::context::frontmost_pid() == Some(self.anchor.pid) {
                    return true;
                }
                if Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(FRONT_POLL);
            }
        }

        fn read(&mut self) -> Result<Vec<u16>, Why> {
            edit::read_units(&self.anchor.element, MAX_FIELD_UNITS).map_err(why)
        }

        fn can_set_text(&mut self) -> bool {
            edit::is_settable(&self.anchor.element, c"AXSelectedTextRange")
                && edit::is_settable(&self.anchor.element, c"AXSelectedText")
        }

        fn select(&mut self, range: Range<usize>) -> bool {
            edit::select(&self.anchor.element, range)
        }

        fn set_selected_text(&mut self, text: &str) -> bool {
            edit::set_selected_text(&self.anchor.element, text)
        }

        fn is_focused(&mut self) -> bool {
            edit::is_focused(self.anchor.pid, &self.anchor.element)
        }

        fn is_frontmost(&mut self) -> bool {
            crate::dictation::context::frontmost_pid() == Some(self.anchor.pid)
        }

        fn begin_edit(&mut self) -> Result<(), Why> {
            let delivery = paste_anchor::deliver();
            if paste_anchor::pastes() != self.pastes {
                return Err(Why::OtherDictation);
            }
            if !crate::labs::tidy_after_paste() {
                return Err(Why::Disabled);
            }
            self.delivery = Some(delivery);
            Ok(())
        }

        fn end_edit(&mut self) {
            self.delivery = None;
        }

        fn paste(&mut self, text: &str) -> bool {
            crate::paste::replace_selection(text)
                .inspect_err(|error| tracing::warn!(%error, "tidy could not paste its edit"))
                .is_ok()
        }

        fn wait(&mut self) {
            std::thread::sleep(READ_BACK_PAUSE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Kind, Offsets, Source, suggest_with};
    use super::*;

    fn units(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    fn rules(text: &str) -> Vec<Suggestion> {
        suggest_with(text, text, "en", &Default::default(), |_| None)
    }

    fn replace(text: &str, find: &str, with: &str) -> Suggestion {
        let byte = text.find(find).unwrap();
        let scalar = text[..byte].chars().count();
        let utf16 = text[..byte].encode_utf16().count();
        Suggestion {
            range: Offsets {
                utf16: utf16..utf16 + find.encode_utf16().count(),
                scalar: scalar..scalar + find.chars().count(),
            },
            kind: Kind::Misheard,
            replacement: with.into(),
            source: Source::Model,
            confidence: 0.5,
        }
    }

    const DICTATION: &str = "um so I think we should, uh, send it Tuesday, scratch that, Wednesday";

    #[test]
    fn locates_by_range_then_unique_search() {
        let inserted = Inserted::of(" hello there");
        let field = units("Dear Kit, hello there and bye");
        let start = "Dear Kit,".encode_utf16().count();
        let range = start..start + inserted.utf16_len();
        assert_eq!(
            locate(&field, &inserted, Some(start)),
            Located::At(range.clone())
        );
        // A stale expected range (text typed before the paste moved it),
        // or none at all: the unique search finds it.
        assert_eq!(
            locate(&field, &inserted, Some(0)),
            Located::At(range.clone())
        );
        assert_eq!(locate(&field, &inserted, None), Located::At(range));
        // An expected range past the field's end is just a miss.
        assert_eq!(
            locate(&field, &inserted, Some(10_000)),
            Located::At(start..start + 12)
        );
        assert_eq!(
            locate(&units("nothing here"), &inserted, Some(0)),
            Located::Missing
        );
        assert_eq!(locate(&[], &inserted, None), Located::Missing);
    }

    #[test]
    fn duplicate_text_is_moved() {
        let inserted = Inserted::of("send it");
        // Typed a second copy after the paste: even at the expected range,
        // Hex can't tell its own copy.
        let field = units("send it, send it");
        assert_eq!(locate(&field, &inserted, Some(0)), Located::Duplicated);
        // Overlapping copies count too.
        let overlapping = Inserted::of("aa");
        assert_eq!(
            locate(&units("aaa"), &overlapping, Some(0)),
            Located::Duplicated
        );
        let edits = rules("send it");
        assert_eq!(
            plan(&field, &inserted, Some(0), "send it", &edits),
            Err(NoPlan::Moved)
        );
    }

    #[test]
    fn outside_range_byte_identical_after_apply() {
        let before = "Notes:\n";
        let after = "\nThe end.";
        let field = units(&format!("{before}{DICTATION}{after}"));
        let start = before.encode_utf16().count();
        let inserted = Inserted::of(DICTATION);
        let plan = plan(&field, &inserted, Some(start), DICTATION, &rules(DICTATION)).unwrap();
        assert_eq!(plan.replacement, "So I think we should send it Wednesday");
        assert_eq!(plan.range, start..start + inserted.utf16_len());
        let end = plan.range.start + plan.replacement.encode_utf16().count();
        assert_eq!(plan.after[..start], field[..start]);
        assert_eq!(plan.after[end..], field[plan.range.end..]);
        assert_eq!(
            String::from_utf16(&plan.after).unwrap(),
            format!("{before}So I think we should send it Wednesday{after}")
        );
        // The re-read decides: exactly the plan, untouched, or anything
        // else (a unit outside the range changed).
        assert_eq!(check(&field, &plan, &plan.after), Check::Applied);
        assert_eq!(check(&field, &plan, &field), Check::Unchanged);
        let mut outside = plan.after.clone();
        outside[0] = u16::from(b'n');
        assert_eq!(check(&field, &plan, &outside), Check::Other);
        let mut inside = plan.after.clone();
        inside[start] = u16::from(b's');
        assert_eq!(check(&field, &plan, &inside), Check::Other);
    }

    #[test]
    fn typed_text_before_and_after_the_paste_still_applies() {
        let inserted = Inserted::of(" um we ship it");
        // Pasted at 5 into "Hello"; then "Oh. " typed before and "!" after.
        let field = units("Oh. Hello um we ship it!");
        let plan = plan(
            &field,
            &inserted,
            Some(5),
            "um we ship it",
            &rules("um we ship it"),
        )
        .unwrap();
        assert_eq!(
            String::from_utf16(&plan.after).unwrap(),
            "Oh. Hello We ship it!"
        );
    }

    #[test]
    fn an_edit_inside_the_pasted_range_is_moved() {
        let inserted = Inserted::of("um we ship it");
        let field = units("um we shipped it");
        assert_eq!(locate(&field, &inserted, Some(0)), Located::Missing);
        assert_eq!(
            plan(
                &field,
                &inserted,
                Some(0),
                "um we ship it",
                &rules("um we ship it")
            ),
            Err(NoPlan::Moved)
        );
    }

    /// The paste's own changes to the dictation (a leading space, the
    /// first letter's case after a continuation) still place the edits;
    /// anything else doesn't.
    #[test]
    fn the_pastes_leading_space_and_initial_case_are_allowed() {
        let text = "um so we ship it";
        let inserted = Inserted::of(" Um so we ship it");
        let field = units("Done. Um so we ship it");
        let plan = plan(&field, &inserted, Some(5), text, &rules(text)).unwrap();
        assert_eq!(
            String::from_utf16(&plan.after).unwrap(),
            "Done. So we ship it"
        );
        let other = Inserted::of("a different text");
        assert_eq!(
            plan_for(&units("x a different text"), &other, text),
            Err(NoPlan::Differs)
        );
    }

    fn plan_for(field: &[u16], inserted: &Inserted, text: &str) -> Result<Plan, NoPlan> {
        plan(field, inserted, None, text, &rules(text))
    }

    /// Ranges count UTF-16 units: an emoji is two, an accented letter one.
    #[test]
    fn multibyte_and_emoji_ranges() {
        let text = "café 🎉 the the plan, Kip";
        let inserted = Inserted::of(text);
        assert_eq!(inserted.utf16_len(), text.encode_utf16().count());
        let field = units(&format!("🙂🙂 {text} ✅"));
        let start = "🙂🙂 ".encode_utf16().count();
        assert_eq!(start, 5);
        let mut edits = rules(text);
        edits.push(replace(text, "Kip", "Kit"));
        edits.sort_by_key(|edit| edit.range.scalar.start);
        let plan = plan(&field, &inserted, Some(start), text, &edits).unwrap();
        assert_eq!(plan.range, start..start + text.encode_utf16().count());
        assert_eq!(
            String::from_utf16(&plan.after).unwrap(),
            "🙂🙂 café 🎉 the plan, Kit ✅"
        );
        // A range that would split a surrogate pair matches nothing.
        let half = Inserted::of("🎉");
        assert_eq!(locate(&field[..start - 1], &half, None), Located::Missing);
    }

    /// A field in memory: settable or not, ignoring `AXSelectedText` or
    /// not, focused or not, and secure or not.
    struct Fake {
        value: Vec<u16>,
        selection: Range<usize>,
        secure: bool,
        in_front: bool,
        settable: bool,
        ignores_set: bool,
        focused: bool,
        /// `AXSelectedText` lands, but the app changes more than the range.
        mangles: bool,
        /// The paste goes nowhere.
        paste_lost: bool,
        /// Another app comes to the front during the read-backs.
        leaves_front: bool,
        /// Typed into the field (at, text) after the last read-back, just
        /// before the fallback's read.
        typed_meanwhile: Option<(usize, &'static str)>,
        reads: usize,
        /// What `begin_edit` refuses with.
        refuse_edit: Option<Why>,
        editing: bool,
        pasted: usize,
        set: usize,
    }

    impl Fake {
        fn new(value: &str) -> Self {
            Self {
                value: units(value),
                selection: 0..0,
                secure: false,
                in_front: true,
                settable: true,
                ignores_set: false,
                focused: true,
                mangles: false,
                paste_lost: false,
                leaves_front: false,
                typed_meanwhile: None,
                reads: 0,
                refuse_edit: None,
                editing: false,
                pasted: 0,
                set: 0,
            }
        }

        fn replace_selection(&mut self, text: &str) {
            let range = self.selection.clone();
            self.value.splice(range, text.encode_utf16());
        }

        fn text(&self) -> String {
            String::from_utf16(&self.value).unwrap()
        }
    }

    impl Target for Fake {
        fn is_secure(&mut self) -> Result<bool, Why> {
            Ok(self.secure)
        }
        fn owner_in_front(&mut self) -> bool {
            assert!(!self.secure, "asked about a secure field");
            self.in_front
        }
        fn read(&mut self) -> Result<Vec<u16>, Why> {
            assert!(!self.secure, "read a secure field");
            if self.set + self.pasted > 0 {
                assert!(self.editing, "read back outside the delivery lock");
            }
            // The first read, then every read-back of the ignored edit.
            if self.reads == 1 + READ_BACKS
                && let Some((at, typed)) = self.typed_meanwhile.take()
            {
                self.value.splice(at..at, typed.encode_utf16());
            }
            self.reads += 1;
            Ok(self.value.clone())
        }
        fn can_set_text(&mut self) -> bool {
            self.settable
        }
        fn select(&mut self, range: Range<usize>) -> bool {
            self.selection = range;
            true
        }
        fn set_selected_text(&mut self, text: &str) -> bool {
            assert!(self.editing, "edited outside the delivery lock");
            self.set += 1;
            if self.mangles {
                self.replace_selection(&format!("{text}!"));
            } else if !self.ignores_set {
                self.replace_selection(text);
            }
            true
        }
        fn is_focused(&mut self) -> bool {
            self.focused
        }
        fn paste(&mut self, text: &str) -> bool {
            assert!(self.editing, "pasted outside the delivery lock");
            self.pasted += 1;
            if !self.paste_lost {
                self.replace_selection(text);
            }
            true
        }
        fn is_frontmost(&mut self) -> bool {
            !self.leaves_front
        }
        fn begin_edit(&mut self) -> Result<(), Why> {
            if let Some(why) = self.refuse_edit {
                return Err(why);
            }
            self.editing = true;
            Ok(())
        }
        fn end_edit(&mut self) {
            self.editing = false;
        }
        fn wait(&mut self) {}
    }

    fn run(fake: &mut Fake, inserted: &str, expected: Option<usize>) -> Outcome {
        apply(
            fake,
            &Inserted::of(inserted),
            expected,
            inserted.trim_start(),
            &rules(inserted.trim_start()),
        )
    }

    #[test]
    fn a_settable_field_takes_one_selected_text_edit() {
        let mut fake = Fake::new("Hi. um we ship it");
        assert_eq!(
            run(&mut fake, " um we ship it", Some(3)),
            Outcome::Applied(Method::SelectedText)
        );
        assert_eq!(fake.text(), "Hi. We ship it");
        assert_eq!((fake.set, fake.pasted), (1, 0));
    }

    #[test]
    fn a_field_that_refuses_or_ignores_selected_text_gets_a_paste() {
        let mut refuses = Fake::new("um we ship it");
        refuses.settable = false;
        assert_eq!(
            run(&mut refuses, "um we ship it", Some(0)),
            Outcome::Applied(Method::Paste)
        );
        assert_eq!(refuses.text(), "We ship it");
        assert_eq!((refuses.set, refuses.pasted), (0, 1));
        let mut ignores = Fake::new("um we ship it");
        ignores.ignores_set = true;
        assert_eq!(
            run(&mut ignores, "um we ship it", Some(0)),
            Outcome::Applied(Method::Paste)
        );
        assert_eq!(ignores.text(), "We ship it");
        assert_eq!((ignores.set, ignores.pasted), (1, 1));
        // The paste needs the field focused; without it nothing changes.
        let mut unfocused = Fake::new("um we ship it");
        unfocused.settable = false;
        unfocused.focused = false;
        assert_eq!(
            run(&mut unfocused, "um we ship it", Some(0)),
            Outcome::Unsupported(Why::OtherField)
        );
        assert_eq!(unfocused.text(), "um we ship it");
    }

    /// The role check comes first: a secure field is never read, and
    /// nothing is sent to it.
    #[test]
    fn a_secure_field_is_never_read_or_edited() {
        let mut fake = Fake::new("um we ship it");
        fake.secure = true;
        assert_eq!(
            run(&mut fake, "um we ship it", Some(0)),
            Outcome::Unsupported(Why::SecureField)
        );
        assert_eq!((fake.set, fake.pasted), (0, 0));
    }

    #[test]
    fn another_app_in_front_or_moved_text_changes_nothing() {
        let mut behind = Fake::new("um we ship it");
        behind.in_front = false;
        assert_eq!(
            run(&mut behind, "um we ship it", Some(0)),
            Outcome::Unsupported(Why::OtherApp)
        );
        let mut edited = Fake::new("um we shipped it");
        assert_eq!(run(&mut edited, "um we ship it", Some(0)), Outcome::Moved);
        assert_eq!((edited.set, edited.pasted), (0, 0));
        assert_eq!(edited.text(), "um we shipped it");
    }

    /// An edit the re-read can't confirm is reported, never retried or
    /// pasted over a second time: the field may already hold it, and the
    /// capsule says to check the text and that ⌘Z undoes it.
    #[test]
    fn an_edit_the_reread_cannot_confirm_is_unverified() {
        let mut mangled = Fake::new("um we ship it");
        mangled.mangles = true;
        assert_eq!(
            run(&mut mangled, "um we ship it", Some(0)),
            Outcome::Unsupported(Why::Unverified)
        );
        assert_eq!((mangled.set, mangled.pasted), (1, 0));
        let mut lost = Fake::new("um we ship it");
        lost.settable = false;
        lost.paste_lost = true;
        assert_eq!(
            run(&mut lost, "um we ship it", Some(0)),
            Outcome::Unsupported(Why::Unverified)
        );
        assert_eq!((lost.set, lost.pasted), (0, 1));
    }

    /// A newer dictation's paste, or the lab turning off, since the anchor
    /// was taken: nothing is sent.
    #[test]
    fn a_newer_paste_or_the_lab_off_sends_nothing() {
        for why in [Why::OtherDictation, Why::Disabled] {
            for settable in [true, false] {
                let mut fake = Fake::new("um we ship it");
                fake.settable = settable;
                fake.refuse_edit = Some(why);
                assert_eq!(
                    run(&mut fake, "um we ship it", Some(0)),
                    Outcome::Unsupported(why)
                );
                assert_eq!((fake.set, fake.pasted), (0, 0));
                assert_eq!(fake.text(), "um we ship it");
            }
        }
    }

    /// The fallback goes only to the anchored app while it is in front:
    /// another app coming forward during the read-backs gets nothing.
    #[test]
    fn the_fallback_rechecks_the_front_app() {
        let mut fake = Fake::new("um we ship it");
        fake.ignores_set = true;
        fake.leaves_front = true;
        assert_eq!(
            run(&mut fake, "um we ship it", Some(0)),
            Outcome::Unsupported(Why::OtherApp)
        );
        assert_eq!((fake.set, fake.pasted), (1, 0));
        // The pasted range isn't left selected for the next paste.
        assert_eq!(fake.selection, 13..13);
        assert!(!fake.editing, "the delivery lock is released");
    }

    /// Typing after the read-backs: the fallback places the range from a
    /// fresh read, so text typed before the paste survives, and an edit
    /// inside it is Moved with nothing pasted.
    #[test]
    fn the_fallback_rereads_the_field_before_pasting() {
        let mut before = Fake::new("um we ship it");
        before.ignores_set = true;
        before.typed_meanwhile = Some((0, "Note: "));
        assert_eq!(
            run(&mut before, "um we ship it", Some(0)),
            Outcome::Applied(Method::Paste)
        );
        assert_eq!(before.text(), "Note: We ship it");
        let mut inside = Fake::new("um we ship it");
        inside.ignores_set = true;
        inside.typed_meanwhile = Some((6, "really "));
        assert_eq!(run(&mut inside, "um we ship it", Some(0)), Outcome::Moved);
        assert_eq!(inside.pasted, 0);
        assert_eq!(inside.text(), "um we really ship it");
    }
}
