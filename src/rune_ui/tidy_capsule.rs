//! Labs › Tidy after paste's review capsule (T8.3): the last dictation's
//! text with each suggested edit struck through or replaced, a switch per
//! edit, Accept all, Cancel, and the one action that applies them. Return
//! applies and Escape cancels.
//!
//! On macOS it is a Rune overlay that may take keyboard focus
//! (`KeyPolicy::OnDemand`): it asks for the keys only while a review is
//! open, and as a non-activating panel it never brings Hex forward, so
//! hiding it hands the keys back to the app that had them, where T8.4
//! edits the pasted text. One review at a time: a new review replaces the
//! open one, and a new dictation closes it ([`super::super::dictation::tidy::last`]).
//!
//! The same view draws in the window for the `tidy-capsule` scene, through a
//! lens onto the window's state ([`panel`]), so the goldens and the tests see
//! exactly what the overlay shows.

use std::sync::OnceLock;
use std::time::Duration;

use rune::{Cx, El, KeyEvent, OverlayHandle, Span, Tx, col, row, text};
use rune_kit::{Button, ButtonVariant, ControlSize, Switch};

use super::controls::{DESCRIPTION, Paint, PushSize};
use super::lists::{Tone, tone_tag};
use crate::desktop::i18n::{tr, tr_named};
use crate::dictation::tidy::Kind;
use crate::dictation::tidy::review::{self, Applied};
use crate::dictation::tidy::session::{Delivery, Segment, TidySession};

/// The panel's width; its height follows its rows.
pub(super) const PANEL_WIDTH: f32 = 520.0;
/// The overlay window: the panel with room for eight suggestions below the
/// text. Only the panel takes the pointer.
const WINDOW: (f32, f32) = (PANEL_WIDTH + 2.0 * INSET, 560.0);
/// Between the window's edges and the panel.
const INSET: f32 = 8.0;
/// Below the menu bar, where the review reads as a sheet over the app.
const TOP_MARGIN: f32 = 96.0;
/// How long a notice stays before the capsule closes itself.
const NOTICE_SECONDS: f64 = 1.5;
const ROW_HEIGHT: f32 = 32.0;

/// What the capsule says when it isn't reviewing, before it closes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Notice {
    NothingToTidy,
    NoDictation,
    /// A pasted dictation's edits went to the clipboard (T8.4 will apply
    /// them in place).
    CopiedToPaste,
    /// A copy-only dictation's tidied text is on the clipboard.
    Copied,
    CopyFailed,
}

impl Notice {
    fn text(self) -> &'static str {
        match self {
            Self::NothingToTidy => tr("Nothing to tidy in the last dictation"),
            Self::NoDictation => tr("No dictation to tidy yet"),
            Self::CopiedToPaste => {
                tr("Copied the tidied text. Paste it over the dictation to use it.")
            }
            Self::Copied => tr("Copied the tidied text"),
            Self::CopyFailed => tr("Couldn't copy the tidied text"),
        }
    }
}

/// The capsule's state.
#[derive(Default)]
pub(crate) struct TidyCapsule {
    pub(crate) session: Option<TidySession>,
    pub(crate) notice: Option<Notice>,
    /// When the notice closes the capsule, on the overlay's clock.
    closing_at: Option<f64>,
    /// The scene's copy: Return and Copy change nothing outside it.
    preview: bool,
    overlay: Option<OverlayHandle<TidyCapsule>>,
}

impl TidyCapsule {
    /// The `tidy-capsule` scene: AC-15's dictation under review.
    pub(crate) fn preview() -> Self {
        Self {
            session: Some(crate::dictation::tidy::session::preview()),
            preview: true,
            ..Self::default()
        }
    }

    /// Shows `session`, in place of any review open; with no suggestions it
    /// says so and closes itself.
    pub(crate) fn show(&mut self, session: TidySession) {
        self.closing_at = None;
        if session.is_empty() {
            self.session = None;
            self.notice = Some(Notice::NothingToTidy);
        } else {
            self.session = Some(session);
            self.notice = None;
        }
    }

    fn say(&mut self, notice: Notice) {
        self.session = None;
        self.notice = Some(notice);
        self.closing_at = None;
    }

    pub(crate) fn toggle(&mut self, index: usize) {
        if let Some(session) = &mut self.session {
            session.toggle(index);
        }
    }

    pub(crate) fn accept_all(&mut self) {
        if let Some(session) = &mut self.session {
            session.accept_all();
        }
    }

    /// Return: applies the accepted edits through [`review::apply_accepted`]
    /// (T8.4's seam) or, for a copy-only dictation, copies the tidied text.
    pub(crate) fn apply(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        let pasted = session.delivery == Delivery::Pasted;
        let applied = match (self.preview, pasted) {
            (true, _) if session.accepted_count() == 0 => Applied::Nothing,
            (true, _) => Applied::Copied,
            (false, true) => review::apply_accepted(&session),
            (false, false) => review::copy_tidied(&session),
        };
        match applied {
            Applied::Copied if pasted => self.say(Notice::CopiedToPaste),
            Applied::Copied => self.say(Notice::Copied),
            Applied::Nothing => self.close(),
            Applied::Failed(_) => self.say(Notice::CopyFailed),
        }
    }

    /// Escape or Cancel: nothing changes.
    pub(crate) fn cancel(&mut self) {
        if let Some(session) = &self.session {
            tracing::info!(
                suggestions = session.suggestions.len(),
                "tidy review cancelled"
            );
        }
        self.close();
    }

    fn close(&mut self) {
        self.session = None;
        self.notice = None;
        self.closing_at = None;
        if let Some(overlay) = &self.overlay {
            overlay.hide();
        }
    }

    /// A dictation committed: a review of an older one closes.
    pub(crate) fn dictation_committed(&mut self, generation: u64) {
        if self
            .session
            .as_ref()
            .is_some_and(|session| session.is_stale(generation))
        {
            tracing::info!("tidy review closed: a newer dictation committed");
            self.close();
        }
    }

    /// Counts a notice down from when it first shows, then closes.
    fn tick(&mut self, now: f64, tx: &mut Tx) -> bool {
        if self.notice.is_none() {
            return false;
        }
        let closing_at = *self.closing_at.get_or_insert(now + NOTICE_SECONDS);
        if now >= closing_at {
            self.close();
            return true;
        }
        tx.wake_after(Duration::from_secs_f64(closing_at - now));
        false
    }

    /// Return applies and Escape cancels, while a review is open.
    fn key(&mut self, key: &KeyEvent) -> bool {
        use rune::Key;
        match key.key {
            Key::Enter if self.session.is_some() => self.apply(),
            Key::Escape if self.session.is_some() || self.notice.is_some() => self.cancel(),
            _ => return false,
        }
        true
    }
}

/// A view's way to the capsule in its state: the overlay's own state, or
/// the window's scene copy.
pub(super) type Lens<S> = fn(&mut S) -> Option<&mut TidyCapsule>;

/// The capsule, at its natural height.
pub(super) fn panel<S: 'static>(capsule: &TidyCapsule, p: Paint, at: Lens<S>) -> El {
    let c = &p.k.c;
    let shell = col()
        .key("tidy-capsule")
        .w(PANEL_WIDTH)
        .padding(12.0)
        .gap(12.0)
        .radius(p.k.radius.md + 8.0)
        .fill(c.bg_base)
        .border(p.hair, c.border_base)
        .hit();
    if let Some(notice) = capsule.notice {
        return shell.role("status").label(notice.text()).child(
            row()
                .key("notice")
                .gap(8.0)
                .center_y()
                .child(
                    text(tr("Tidy last dictation"))
                        .key("title")
                        .font(p.k.ty.ui_medium)
                        .color(p.text())
                        .a11y_hidden(),
                )
                .child(
                    text(notice.text())
                        .key("message")
                        .font(DESCRIPTION)
                        .color(p.muted())
                        .wrap()
                        .grow(1.0)
                        .min_w(0.0),
                ),
        );
    }
    let Some(session) = &capsule.session else {
        return row().key("tidy-capsule-empty");
    };
    let count = tr_named(
        "{accepted} of {total} edits accepted",
        &[
            ("accepted", &session.accepted_count().to_string()),
            ("total", &session.suggestions.len().to_string()),
        ],
    );
    shell
        .role("dialog")
        .label(tr("Tidy last dictation"))
        .child(
            row()
                .key("header")
                .gap(8.0)
                .center_y()
                .child(
                    text(tr("Tidy last dictation"))
                        .key("title")
                        .role("heading")
                        .font(p.k.ty.ui_medium)
                        .color(p.text())
                        .grow(1.0),
                )
                .child(text(count).key("count").font(p.k.ty.small).color(p.faint())),
        )
        .child(preview_text(session, p))
        .child(suggestion_rows(session, p, at))
        .child(actions(session, p, at))
}

/// The text as accepting would leave it, with what goes struck through and
/// what comes in beside it; a skipped edit's words are underlined.
fn preview_text(session: &TidySession, p: Paint) -> El {
    let c = &p.k.c;
    let mut shown = String::new();
    let mut spans = Vec::new();
    let mut strikes = Vec::new();
    for segment in session.segments() {
        match segment {
            Segment::Plain(words) => shown.push_str(&words),
            Segment::Edit {
                before,
                after,
                accepted: true,
                ..
            } => {
                let start = shown.len();
                shown.push_str(&before);
                spans.push(Span::new(start, shown.len()).color(c.text_faint));
                strikes.push((start, shown.len(), Some(c.text_faint)));
                if !after.is_empty() {
                    shown.push(' ');
                    let start = shown.len();
                    shown.push_str(&after);
                    spans.push(Span::new(start, shown.len()).color(c.text_accent));
                }
            }
            Segment::Edit { before, .. } => {
                let start = shown.len();
                shown.push_str(&before);
                spans.push(Span::new(start, shown.len()).underline(c.border_strong));
            }
        }
    }
    col()
        .key("text")
        .padding(10.0)
        .radius(p.k.radius.md)
        .fill(c.layer_01)
        .child(
            text(shown)
                .key("preview")
                .font(p.k.ty.ui)
                .color(p.text())
                .wrap()
                .spans(spans)
                .strikes(strikes),
        )
}

fn suggestion_rows<S: 'static>(session: &TidySession, p: Paint, at: Lens<S>) -> El {
    let rows = session
        .suggestions
        .iter()
        .enumerate()
        .map(|(index, suggestion)| {
            let before = segment_words(session, index);
            let what = if suggestion.replacement.is_empty() {
                tr_named("Remove “{words}”", &[("words", &before)])
            } else {
                tr_named(
                    "Replace “{words}” with “{replacement}”",
                    &[("words", &before), ("replacement", &suggestion.replacement)],
                )
            };
            let switch = Switch::new(session.accepted[index])
                .control_size(ControlSize::Small)
                .label(&what)
                .hide_label()
                .on_toggle(move |state: &mut S| {
                    if let Some(capsule) = at(state) {
                        capsule.toggle(index);
                    }
                })
                .build(p.k)
                .key("switch");
            row()
                .key(format!("suggestion-{index}"))
                .h(ROW_HEIGHT)
                .gap(10.0)
                .center_y()
                .child(switch)
                .child(
                    text(what)
                        .key("what")
                        .font(DESCRIPTION)
                        .color(p.text())
                        .truncate()
                        .grow(1.0)
                        .min_w(0.0),
                )
                .child(tone_tag(
                    p,
                    "kind",
                    kind_label(suggestion.kind),
                    Tone::Neutral,
                ))
        });
    col().key("suggestions").children(rows)
}

/// The words suggestion `index` replaces, trimmed of the commas and spaces
/// around them, for its row.
fn segment_words(session: &TidySession, index: usize) -> String {
    let range = &session.suggestions[index].range.scalar;
    let words: String = session
        .text
        .chars()
        .skip(range.start)
        .take(range.end - range.start)
        .collect();
    words
        .trim_matches(|character: char| character.is_whitespace() || character == ',')
        .to_owned()
}

fn kind_label(kind: Kind) -> &'static str {
    match kind {
        Kind::Filler => tr("Filler"),
        Kind::Repeat => tr("Repeat"),
        Kind::SelfCorrection => tr("Self-correction"),
        Kind::Misheard => tr("Misheard"),
    }
}

/// Cancel and Accept all, then the one accented action: Apply for a pasted
/// dictation, Copy tidied text for one that only reached the clipboard.
fn actions<S: 'static>(session: &TidySession, p: Paint, at: Lens<S>) -> El {
    let secondary = |key: &str, label: &str, press: fn(&mut TidyCapsule)| {
        Button::new(label)
            .variant(ButtonVariant::Neutral)
            .on_press(move |state: &mut S| {
                if let Some(capsule) = at(state) {
                    press(capsule);
                }
            })
            .build_push(p.k)
            .key(key)
    };
    let apply_label = match session.delivery {
        Delivery::Pasted => tr("Apply"),
        Delivery::Copied => tr("Copy tidied text"),
    };
    let apply = Button::new(apply_label)
        .variant(ButtonVariant::Contrast)
        .on_press(move |state: &mut S| {
            if let Some(capsule) = at(state) {
                capsule.apply();
            }
        })
        .build_push(p.k)
        .key("tidy-apply")
        .default_button();
    row()
        .key("actions")
        .gap(8.0)
        .center_y()
        .child(row().key("spacer").grow(1.0))
        .child(secondary("tidy-cancel", tr("Cancel"), TidyCapsule::cancel))
        .child(secondary(
            "tidy-accept-all",
            tr("Accept all"),
            TidyCapsule::accept_all,
        ))
        .child(apply)
}

/// The `tidy-capsule` scene: the capsule on the window's backdrop, where
/// the overlay would sit over an app.
pub(super) fn scene(capsule: &TidyCapsule, p: Paint) -> El {
    col()
        .key("tidy-scene")
        .w_full()
        .h_full()
        .padding(24.0)
        .items_center()
        .fill(p.k.c.bg_deep)
        .hover_motion_within(rune::Anim::INSTANT, rune::Anim::INSTANT)
        .child(panel(capsule, p, |ui: &mut super::Ui| {
            ui.tidy_preview.as_mut()
        }))
}

// ─── The macOS overlay ───────────────────────────────────────────────────────

static OVERLAY: OnceLock<OverlayHandle<TidyCapsule>> = OnceLock::new();

fn view(capsule: &TidyCapsule, cx: &mut Cx<TidyCapsule>) -> El {
    let p = Paint::new(cx.system_dark(), cx.env().scale);
    col()
        .key("tidy-window")
        .w_full()
        .h_full()
        .padding(INSET)
        .items_center()
        .hover_motion_within(rune::Anim::INSTANT, rune::Anim::INSTANT)
        .child(panel(capsule, p, |capsule: &mut TidyCapsule| Some(capsule)))
}

fn on_key(capsule: &mut TidyCapsule, key: &KeyEvent, _: &mut Tx) -> bool {
    capsule.key(key)
}

fn poll(capsule: &mut TidyCapsule, now: f64, tx: &mut Tx) -> bool {
    capsule.tick(now, tx)
}

/// Adds the capsule's overlay, hidden, and closes an open review when a
/// dictation commits.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn install<S: 'static>(app: &mut rune::App<S>) {
    use rune::{
        Align, DisplayTarget, KeyPolicy, OverlayInput, OverlayOptions, OverlayPlacement,
        OverlaySpec, ScreenEdge,
    };
    let spec = OverlaySpec::new("Hex tidy review", WINDOW, view)
        .options(OverlayOptions {
            placement: Some(OverlayPlacement::Edge {
                edge: ScreenEdge::Top,
                margin: TOP_MARGIN,
                align: Align::Center,
                inset: 0.0,
            }),
            display: DisplayTarget::UnderPointer,
            exclude_from_capture: true,
            input: OverlayInput::Regions,
            keys: KeyPolicy::OnDemand,
            ..Default::default()
        })
        .release_gpu_when_hidden()
        .on_key(on_key)
        .poll(poll);
    let overlay = app.overlay(spec, TidyCapsule::default());
    let handle = overlay.clone();
    overlay.update(move |capsule, _| capsule.overlay = Some(handle));
    let committed = overlay.clone();
    crate::dictation::tidy::last::on_commit(move |generation| {
        committed.update(move |capsule, _| capsule.dictation_committed(generation));
    });
    let _ = OVERLAY.set(overlay);
}

/// Opens the review of `session`, in place of any open, and takes the keys
/// while it shows. Safe from any thread.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn open(session: TidySession) {
    let Some(overlay) = OVERLAY.get().cloned() else {
        return;
    };
    let handle = overlay.clone();
    overlay.update(move |capsule, tx| {
        capsule.show(session);
        handle.show();
        if capsule.session.is_some() {
            handle.request_keys();
        }
        tx.animated();
    });
}

/// Says there is no dictation to tidy, and closes after a moment.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn nothing_to_tidy() {
    let Some(overlay) = OVERLAY.get().cloned() else {
        return;
    };
    let handle = overlay.clone();
    overlay.update(move |capsule, tx| {
        capsule.say(Notice::NoDictation);
        handle.show();
        tx.animated();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(key: rune::Key) -> KeyEvent {
        KeyEvent {
            key,
            mods: rune::Mods::default(),
        }
    }

    #[test]
    fn toggle_and_accept_all() {
        let mut capsule = TidyCapsule::preview();
        let accepted = |capsule: &TidyCapsule| capsule.session.as_ref().unwrap().accepted.clone();
        assert_eq!(accepted(&capsule), [true, true, false, true]);
        capsule.toggle(0);
        capsule.toggle(2);
        assert_eq!(accepted(&capsule), [false, true, true, true]);
        capsule.accept_all();
        assert_eq!(accepted(&capsule), [true; 4]);
        // Cancel changes nothing and closes.
        capsule.cancel();
        assert!(capsule.session.is_none() && capsule.notice.is_none());
    }

    #[test]
    fn return_applies_and_says_so_then_closes_after_a_moment() {
        let mut capsule = TidyCapsule::preview();
        assert!(capsule.key(&press(rune::Key::Enter)));
        assert!(capsule.session.is_none());
        assert_eq!(capsule.notice, Some(Notice::CopiedToPaste));
        let mut tx = Tx::default();
        assert!(!capsule.tick(10.0, &mut tx));
        assert!(!capsule.tick(11.0, &mut tx));
        assert!(capsule.tick(11.5, &mut tx));
        assert_eq!(capsule.notice, None);
    }

    #[test]
    fn escape_cancels_and_other_keys_pass() {
        let mut capsule = TidyCapsule::preview();
        assert!(!capsule.key(&press(rune::Key::Text("a".into()))));
        assert!(capsule.key(&press(rune::Key::Escape)));
        assert!(capsule.session.is_none());
        assert!(!capsule.key(&press(rune::Key::Enter)));
    }

    /// No suggestions: the capsule says so and closes itself after 1.5 s.
    #[test]
    fn no_suggestions_closes_after_a_moment() {
        let mut capsule = TidyCapsule::default();
        let mut empty = crate::dictation::tidy::session::preview();
        empty.suggestions.clear();
        empty.accepted.clear();
        capsule.show(empty);
        assert!(capsule.session.is_none());
        assert_eq!(capsule.notice, Some(Notice::NothingToTidy));
        let mut tx = Tx::default();
        assert!(!capsule.tick(0.0, &mut tx));
        assert!(!capsule.tick(1.4, &mut tx));
        assert!(capsule.tick(1.5, &mut tx));
        assert!(capsule.notice.is_none());
    }

    /// A copy-only dictation offers Copy tidied text, and Return copies.
    #[test]
    fn copy_only_offers_a_copy() {
        let mut capsule = TidyCapsule::preview();
        capsule.session.as_mut().unwrap().delivery = Delivery::Copied;
        capsule.apply();
        assert_eq!(capsule.notice, Some(Notice::Copied));
    }

    /// The next dictation closes an open review of the last one.
    #[test]
    fn session_replaced_by_new_dictation() {
        let mut capsule = TidyCapsule::preview();
        capsule.dictation_committed(0);
        assert!(capsule.session.is_some(), "the same generation stays open");
        capsule.dictation_committed(1);
        assert!(capsule.session.is_none());
    }

    /// Accepting nothing and pressing Return closes with nothing changed.
    #[test]
    fn return_with_nothing_accepted_just_closes() {
        let mut capsule = TidyCapsule::preview();
        let session = capsule.session.as_mut().unwrap();
        session
            .accepted
            .iter_mut()
            .for_each(|accepted| *accepted = false);
        capsule.apply();
        assert!(capsule.session.is_none() && capsule.notice.is_none());
    }
}
