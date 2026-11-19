//! Labs › Tidy after paste's review capsule (T8.3): the last dictation's
//! text with each suggested edit struck through or replaced, a switch per
//! edit, Accept all, Cancel, and the one action that applies them. Return
//! applies and Escape cancels.
//!
//! On macOS it is a Rune overlay that may take keyboard focus
//! (`KeyPolicy::OnDemand`): it asks for the keys only while a review is
//! open, and as a non-activating panel it never brings Hex forward, so
//! hiding it hands the keys back to the app that had them, where T8.4
//! edits the pasted text: Return hides the panel and applies on a worker
//! thread, then says it's done, or reopens the review offering Copy tidied
//! text when the text moved or the field can't be edited. One review at a
//! time: a new review replaces the
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
use crate::dictation::tidy::apply::{Method, Why};
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
    /// The pasted text in its field now reads the tidied text.
    Applied,
    /// The tidied text is on the clipboard.
    Copied,
    CopyFailed,
}

impl Notice {
    fn text(self) -> &'static str {
        match self {
            Self::NothingToTidy => tr("Nothing to tidy in the last dictation"),
            Self::NoDictation => tr("No dictation to tidy yet"),
            Self::Applied => tr("Tidied the dictation"),
            Self::Copied => tr("Copied the tidied text"),
            Self::CopyFailed => tr("Couldn't copy the tidied text"),
        }
    }
}

/// Why a review of a pasted dictation came back offering Copy tidied text
/// instead of editing its field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Offer {
    Moved,
    Unsupported(Why),
}

impl Offer {
    fn text(self) -> &'static str {
        match self {
            Self::Moved => tr("The text changed since it was pasted"),
            Self::Unsupported(why) => match why {
                Why::NoAnchor
                | Why::Expired
                | Why::OtherDictation
                | Why::Disabled
                | Why::NotOnThisOs => tr("Hex no longer knows where this dictation was pasted"),
                Why::OtherApp | Why::OtherField => tr("The app it was pasted into isn't in front"),
                Why::SecureField
                | Why::Unreadable
                | Why::TooLarge
                | Why::Differs
                | Why::NotEditable => tr("Hex can't edit the text in this field"),
                Why::Unverified => {
                    tr("Hex couldn't confirm the edit. Check the text; ⌘Z undoes it.")
                }
            },
        }
    }
}

/// Which copy of the capsule a state is.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum Role {
    /// The `tidy-capsule` scene and the tests: Return and Copy change
    /// nothing outside it, and its notices stay in it.
    #[default]
    Preview,
    /// The overlay that takes the keys while a review is open. A review
    /// that ends hides it at once and hands any notice to [`Role::Notice`].
    Review,
    /// The click-through overlay that never takes the keys, for the notices
    /// a review leaves and the ones no review precedes.
    Notice,
}

/// What a review must still match to show or apply: the dictation count
/// and the lab's live gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Live {
    pub(crate) generation: u64,
    pub(crate) lab_on: bool,
}

impl Live {
    /// The real gate: [`crate::dictation::tidy::last`] and the lab.
    fn now() -> Self {
        Self {
            generation: crate::dictation::tidy::last::generation(),
            lab_on: crate::labs::tidy_after_paste(),
        }
    }

    /// The scene's: always current, the lab always on.
    fn preview() -> Self {
        Self {
            generation: 0,
            lab_on: true,
        }
    }

    /// Why `session` may no longer show or apply, if it may not.
    fn refuses(self, session: &TidySession) -> Option<&'static str> {
        if !self.lab_on {
            Some("the lab is off")
        } else if session.is_stale(self.generation) {
            Some("a newer dictation committed")
        } else {
            None
        }
    }
}

/// How often an open review rereads the lab and the dictation count, so
/// turning the lab off closes it.
const RECHECK_SECONDS: f64 = 0.25;

/// The capsule's state.
pub(crate) struct TidyCapsule {
    pub(crate) session: Option<TidySession>,
    pub(crate) notice: Option<Notice>,
    /// Why an open review offers Copy tidied text after Return couldn't
    /// edit the field.
    pub(crate) offer: Option<Offer>,
    /// When the notice closes the capsule, on the overlay's clock.
    closing_at: Option<f64>,
    role: Role,
    /// The live gate a review is checked against before it shows, while it
    /// is open, and before Return applies it.
    live: fn() -> Live,
    /// Applies a pasted dictation's review off the main loop and hands the
    /// result back to [`TidyCapsule::applied`]; the overlay's spawns a
    /// thread, the scene's and the tests' do nothing.
    run_apply: fn(TidySession, u64),
    /// Counts reviews shown and applies started: an apply's result counts
    /// only while it is still the latest, so a review opened meanwhile is
    /// never closed or replaced by an older one's result.
    request: u64,
    overlay: Option<OverlayHandle<TidyCapsule>>,
}

impl Default for TidyCapsule {
    fn default() -> Self {
        Self {
            session: None,
            notice: None,
            offer: None,
            closing_at: None,
            role: Role::Preview,
            live: Live::preview,
            run_apply: |_, _| {},
            request: 0,
            overlay: None,
        }
    }
}

impl TidyCapsule {
    /// The `tidy-capsule` scene: AC-15's dictation under review.
    pub(crate) fn preview() -> Self {
        Self {
            session: Some(crate::dictation::tidy::session::preview()),
            ..Self::default()
        }
    }

    fn overlay_state(role: Role) -> Self {
        Self {
            role,
            live: Live::now,
            run_apply: spawn_apply,
            ..Self::default()
        }
    }

    /// Shows `session`, in place of any review open; with no suggestions it
    /// says so (and closes itself). Refuses, showing nothing, a review whose
    /// dictation is no longer the last or whose lab turned off while it was
    /// computed; returns whether the review is now open.
    pub(crate) fn show(&mut self, session: TidySession) -> bool {
        if let Some(why) = (self.live)().refuses(&session) {
            tracing::info!(why, "tidy review not shown");
            return false;
        }
        self.closing_at = None;
        self.offer = None;
        self.request += 1;
        if session.is_empty() {
            self.say(Notice::NothingToTidy);
        } else {
            self.session = Some(session);
            self.notice = None;
        }
        true
    }

    /// Ends any review and says `notice`. The review overlay hides at once,
    /// giving the keys back, and the notice overlay shows it.
    fn say(&mut self, notice: Notice) {
        self.session = None;
        self.offer = None;
        self.closing_at = None;
        if self.role == Role::Review {
            self.close();
            notify(notice);
        } else {
            self.notice = Some(notice);
        }
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

    /// Return: for a pasted dictation, hides the panel (handing the keys
    /// back to the field) and applies the accepted edits there on a worker
    /// ([`review::apply_accepted`], T8.4), whose result comes back to
    /// [`Self::applied`]; for a copy-only one, or after an offer, copies the
    /// tidied text. A review whose dictation is no longer the last, or whose
    /// lab turned off, closes with nothing applied.
    pub(crate) fn apply(&mut self) {
        let Some(session) = self.session.take() else {
            return;
        };
        if let Some(why) = (self.live)().refuses(&session) {
            tracing::info!(why, "tidy review closed without applying");
            self.close();
            return;
        }
        let pasted = session.delivery == Delivery::Pasted;
        let applied = match (self.role == Role::Preview, pasted) {
            (true, _) if session.accepted_count() == 0 => Applied::Nothing,
            (true, true) => Applied::Edited(Method::SelectedText),
            (true, false) => Applied::Copied,
            (false, true) if session.accepted_count() > 0 => {
                self.close();
                self.request += 1;
                (self.run_apply)(session, self.request);
                return;
            }
            (false, true) => review::apply_accepted(&session),
            (false, false) => review::copy_tidied(&session),
        };
        self.applied(session, applied);
    }

    /// An apply worker's result for `request`: ignored once a newer review
    /// opened or another apply started, except that an edit that went in
    /// still says so, in the notice overlay, without touching the review.
    pub(crate) fn finished(&mut self, request: u64, session: TidySession, applied: Applied) {
        if request != self.request {
            tracing::info!("tidy apply finished after a newer review");
            if matches!(applied, Applied::Edited(_)) && self.role == Role::Review {
                notify(Notice::Applied);
            }
            return;
        }
        self.applied(session, applied);
    }

    /// What Return's apply or copy did: done says so; text that moved or a
    /// field Hex can't edit brings the review back as a copy-only one that
    /// says why, unless its dictation is no longer the last.
    pub(crate) fn applied(&mut self, mut session: TidySession, applied: Applied) {
        let offer = match applied {
            Applied::Edited(_) => return self.say(Notice::Applied),
            Applied::Copied => return self.say(Notice::Copied),
            Applied::Failed(_) => return self.say(Notice::CopyFailed),
            Applied::Nothing => return self.close(),
            Applied::Moved => Offer::Moved,
            Applied::Unsupported(why) => Offer::Unsupported(why),
        };
        if let Some(why) = (self.live)().refuses(&session) {
            tracing::info!(why, "tidy offer not shown");
            return self.close();
        }
        session.delivery = Delivery::Copied;
        self.session = Some(session);
        self.notice = None;
        self.closing_at = None;
        self.offer = Some(offer);
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
        self.offer = None;
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

    /// While a review is open, rereads the live gate every
    /// [`RECHECK_SECONDS`] and closes it once it refuses; a notice counts
    /// down from when it first shows, then closes.
    fn tick(&mut self, now: f64, tx: &mut Tx) -> bool {
        if let Some(session) = &self.session {
            if let Some(why) = (self.live)().refuses(session) {
                tracing::info!(why, "tidy review closed");
                self.close();
                return true;
            }
            tx.wake_after(Duration::from_secs_f64(RECHECK_SECONDS));
            return false;
        }
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
        .children(capsule.offer.map(|offer| offer_line(offer, p)))
        .child(suggestion_rows(session, p, at))
        .child(actions(session, p, at))
}

/// Why Return couldn't edit the field, above the rows, while the review
/// offers Copy tidied text instead.
fn offer_line(offer: Offer, p: Paint) -> El {
    text(offer.text())
        .key("offer")
        .role("status")
        .font(DESCRIPTION)
        .color(p.muted())
        .wrap()
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

// ─── The macOS overlays ──────────────────────────────────────────────────────

/// The review, which takes the keys while a review is open.
static OVERLAY: OnceLock<OverlayHandle<TidyCapsule>> = OnceLock::new();
/// The notices, click-through and never key, so typing or ⌘V right after a
/// review reaches the app in front.
static NOTICE: OnceLock<OverlayHandle<TidyCapsule>> = OnceLock::new();
/// The notice window: one line of the panel.
const NOTICE_WINDOW: (f32, f32) = (PANEL_WIDTH + 2.0 * INSET, 96.0);

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

/// Adds the review and notice overlays, hidden, and closes an open review
/// when a dictation commits.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn install<S: 'static>(app: &mut rune::App<S>) {
    use rune::{
        Align, DisplayTarget, KeyPolicy, OverlayInput, OverlayOptions, OverlayPlacement,
        OverlaySpec, ScreenEdge,
    };
    let options = |input, keys| OverlayOptions {
        placement: Some(OverlayPlacement::Edge {
            edge: ScreenEdge::Top,
            margin: TOP_MARGIN,
            align: Align::Center,
            inset: 0.0,
        }),
        display: DisplayTarget::UnderPointer,
        exclude_from_capture: true,
        input,
        keys,
        ..Default::default()
    };
    let add = |app: &mut rune::App<S>, name, size, input, keys, role| {
        let spec = OverlaySpec::new(name, size, view)
            .options(options(input, keys))
            .release_gpu_when_hidden()
            .on_key(on_key)
            .poll(poll);
        let overlay = app.overlay(spec, TidyCapsule::overlay_state(role));
        let handle = overlay.clone();
        overlay.update(move |capsule, _| capsule.overlay = Some(handle));
        overlay
    };
    let review = add(
        app,
        "Hex tidy review",
        WINDOW,
        OverlayInput::Regions,
        KeyPolicy::OnDemand,
        Role::Review,
    );
    let notice = add(
        app,
        "Hex tidy notice",
        NOTICE_WINDOW,
        OverlayInput::None,
        KeyPolicy::None,
        Role::Notice,
    );
    let committed = review.clone();
    crate::dictation::tidy::last::on_commit(move |generation| {
        committed.update(move |capsule, _| capsule.dictation_committed(generation));
    });
    let _ = OVERLAY.set(review);
    let _ = NOTICE.set(notice);
}

/// Opens the review of `session`, in place of any open, and takes the keys
/// while it shows. Safe from any thread. On the main loop it checks again
/// that the session's dictation is still the last and the lab still on, so a
/// dictation that committed after the worker's check never meets a stale
/// review.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn open(session: TidySession) {
    let Some(overlay) = OVERLAY.get().cloned() else {
        return;
    };
    let handle = overlay.clone();
    overlay.update(move |capsule, tx| {
        if capsule.show(session) && capsule.session.is_some() {
            handle.show();
            handle.request_keys();
            tx.animated();
        }
    });
}

/// Shows `notice` in the click-through notice overlay, which closes itself
/// after a moment. Safe from any thread.
fn notify(notice: Notice) {
    let Some(overlay) = NOTICE.get().cloned() else {
        return;
    };
    let handle = overlay.clone();
    overlay.update(move |capsule, tx| {
        capsule.say(notice);
        handle.show();
        tx.animated();
    });
}

/// Applies `session` on its own thread, then hands the result to the
/// review overlay, which reopens it (taking the keys again) only to offer a
/// copy.
fn spawn_apply(session: TidySession, request: u64) {
    let spawned = std::thread::Builder::new()
        .name("hex-tidy-apply".into())
        .spawn(move || {
            let applied = review::apply_accepted(&session);
            let Some(overlay) = OVERLAY.get().cloned() else {
                return;
            };
            let handle = overlay.clone();
            overlay.update(move |capsule, tx| {
                capsule.finished(request, session, applied);
                if capsule.session.is_some() {
                    handle.show();
                    handle.request_keys();
                }
                tx.animated();
            });
        });
    if let Err(error) = spawned {
        tracing::warn!(%error, "could not start the tidy apply");
        notify(Notice::CopyFailed);
    }
}

/// Says there is no dictation to tidy, and closes after a moment.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn nothing_to_tidy() {
    notify(Notice::NoDictation);
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
        assert_eq!(capsule.notice, Some(Notice::Applied));
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

    fn review(live: fn() -> Live) -> TidyCapsule {
        TidyCapsule {
            role: Role::Review,
            live,
            ..TidyCapsule::default()
        }
    }

    fn current() -> Live {
        Live {
            generation: 0,
            lab_on: true,
        }
    }

    /// A dictation that committed after the review was computed (but
    /// before the main loop showed it) keeps the stale review from opening.
    #[test]
    fn a_stale_review_never_shows() {
        let mut capsule = review(|| Live {
            generation: 1,
            lab_on: true,
        });
        assert!(!capsule.show(crate::dictation::tidy::session::preview()));
        assert!(capsule.session.is_none() && capsule.notice.is_none());
        let mut fresh = review(current);
        assert!(fresh.show(crate::dictation::tidy::session::preview()));
        assert!(fresh.session.is_some());
    }

    /// The lab turned off while suggestions computed: nothing shows.
    #[test]
    fn a_review_computed_before_the_lab_turned_off_never_shows() {
        let mut capsule = review(|| Live {
            generation: 0,
            lab_on: false,
        });
        assert!(!capsule.show(crate::dictation::tidy::session::preview()));
        assert!(capsule.session.is_none());
    }

    /// Return checks again: a newer dictation or the lab going off since
    /// the review opened closes it with nothing copied.
    #[test]
    fn return_refuses_a_stale_review_or_one_whose_lab_turned_off() {
        for live in [
            (|| Live {
                generation: 1,
                lab_on: true,
            }) as fn() -> Live,
            || Live {
                generation: 0,
                lab_on: false,
            },
        ] {
            let mut capsule = TidyCapsule::preview();
            capsule.live = live;
            capsule.apply();
            assert!(capsule.session.is_none());
            assert_eq!(capsule.notice, None, "nothing was copied");
        }
    }

    /// An open review closes within a poll of the lab turning off.
    #[test]
    fn turning_the_lab_off_closes_an_open_review() {
        let mut capsule = review(current);
        assert!(capsule.show(crate::dictation::tidy::session::preview()));
        let mut tx = Tx::default();
        assert!(!capsule.tick(0.0, &mut tx));
        assert!(capsule.session.is_some());
        capsule.live = || Live {
            generation: 0,
            lab_on: false,
        };
        assert!(capsule.tick(0.25, &mut tx));
        assert!(capsule.session.is_none());
    }

    /// When a review ends, its key-taking panel leaves at once: the notice
    /// goes to the click-through notice overlay, never this one.
    #[test]
    fn a_review_that_ends_keeps_no_notice_of_its_own() {
        let mut capsule = review(current);
        assert!(capsule.show(crate::dictation::tidy::session::preview()));
        capsule.apply();
        assert!(capsule.session.is_none() && capsule.notice.is_none());
        let mut empty = crate::dictation::tidy::session::preview();
        empty.suggestions.clear();
        empty.accepted.clear();
        assert!(capsule.show(empty));
        assert!(capsule.session.is_none() && capsule.notice.is_none());
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

    /// Return on a pasted dictation's review hides the panel at once and
    /// hands the session to the worker; the panel holds nothing meanwhile.
    #[test]
    fn return_hands_a_pasted_review_to_the_worker() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static HANDED: AtomicUsize = AtomicUsize::new(0);
        let mut capsule = review(current);
        capsule.run_apply = |session, _| {
            assert_eq!(session.accepted_count(), 3);
            HANDED.fetch_add(1, Ordering::SeqCst);
        };
        assert!(capsule.show(crate::dictation::tidy::session::preview()));
        capsule.apply();
        assert_eq!(HANDED.load(Ordering::SeqCst), 1);
        assert!(capsule.session.is_none() && capsule.notice.is_none());
    }

    /// Moved and Unsupported bring the review back as a copy-only one that
    /// says why; Return then copies. Applied says it's done.
    #[test]
    fn moved_or_unsupported_offers_copy_tidied_text() {
        let session = crate::dictation::tidy::session::preview;
        for (applied, offer) in [
            (Applied::Moved, Offer::Moved),
            (
                Applied::Unsupported(Why::NotEditable),
                Offer::Unsupported(Why::NotEditable),
            ),
        ] {
            let mut capsule = TidyCapsule::default();
            capsule.applied(session(), applied);
            assert_eq!(capsule.offer, Some(offer));
            let reviewed = capsule.session.as_ref().unwrap();
            assert_eq!(reviewed.delivery, Delivery::Copied);
            assert!(!offer.text().is_empty());
            capsule.apply();
            assert_eq!(capsule.notice, Some(Notice::Copied));
            assert_eq!(capsule.offer, None);
        }
        let mut capsule = TidyCapsule::default();
        capsule.applied(session(), Applied::Edited(Method::Paste));
        assert_eq!(capsule.notice, Some(Notice::Applied));
        assert!(capsule.session.is_none());
        // A newer dictation since: the offer never shows.
        let mut stale = review(|| Live {
            generation: 1,
            lab_on: true,
        });
        stale.applied(session(), Applied::Moved);
        assert!(stale.session.is_none() && stale.offer.is_none());
    }

    /// An apply that finishes after a newer review opened leaves that
    /// review alone; the latest apply's result still counts.
    #[test]
    fn an_older_apply_never_touches_a_newer_review() {
        let session = crate::dictation::tidy::session::preview;
        let mut capsule = TidyCapsule::default();
        assert!(capsule.show(session()));
        capsule.apply();
        let first = capsule.request;
        assert!(capsule.show(session()));
        capsule.toggle(0);
        let open = capsule.session.as_ref().map(TidySession::accepted_count);
        for applied in [Applied::Moved, Applied::Edited(Method::Paste)] {
            capsule.finished(first, session(), applied);
            assert_eq!(
                capsule.session.as_ref().map(TidySession::accepted_count),
                open
            );
            assert!(capsule.notice.is_none() && capsule.offer.is_none());
        }
        let latest = capsule.request;
        capsule.finished(latest, session(), Applied::Moved);
        assert_eq!(capsule.offer, Some(Offer::Moved));
    }
}
