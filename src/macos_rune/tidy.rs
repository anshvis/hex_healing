//! Labs › Tidy after paste's triggers on macOS: the Tidy shortcut (from the
//! listener), the menu bar's Tidy Last Dictation, and `hex dictation tidy`.
//! Each reads the lab where it starts, then prepares the review of the last
//! dictation on its own thread, because T8.1's model pass can take 2 s, and
//! opens the capsule (`rune_ui::tidy_capsule`) on the main loop.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::control::{ControlError, ControlReply};
use crate::dictation::tidy::review::{self, Refusal};
use crate::dictation::tidy::session::TidySession;

/// One review is prepared at a time; a second press while one computes is
/// the same request.
static PREPARING: AtomicBool = AtomicBool::new(false);

/// The Tidy shortcut. Returns at once; the review follows.
pub(crate) fn shortcut() -> Result<(), &'static str> {
    open_in_background();
    Ok(())
}

/// The menu bar's Tidy Last Dictation (shown only with the lab on).
pub(crate) fn menu() {
    open_in_background();
}

/// `hex dictation tidy [--print]`, already off the control thread.
pub(super) fn control(print: bool) -> Result<ControlReply, ControlError> {
    crate::control::tidy::reply(print, prepare(), open)
}

fn open_in_background() {
    if !crate::labs::tidy_after_paste() {
        return;
    }
    if PREPARING.swap(true, Ordering::AcqRel) {
        return;
    }
    let spawned = std::thread::Builder::new()
        .name("hex-tidy".into())
        .spawn(|| {
            let prepared = prepare();
            PREPARING.store(false, Ordering::Release);
            match prepared {
                Ok(session) => open(session),
                Err(refusal) => {
                    tracing::info!(?refusal, "tidy opened nothing");
                    if refusal == Refusal::NoDictation {
                        crate::rune_ui::tidy_capsule::nothing_to_tidy();
                    }
                }
            }
        });
    if let Err(error) = spawned {
        PREPARING.store(false, Ordering::Release);
        tracing::warn!(%error, "could not start the tidy review");
    }
}

/// The review of the last dictation, gated by the lab before anything is
/// read.
fn prepare() -> Result<TidySession, Refusal> {
    let lab_on = crate::labs::tidy_after_paste();
    let settings = if lab_on {
        crate::app_settings::AppSettings::read().unwrap_or_default()
    } else {
        crate::app_settings::AppSettings::default()
    };
    review::prepare(
        lab_on,
        || {
            super::runtime()
                .and_then(|runtime| runtime.history())
                .and_then(|history| history.newest())
        },
        &settings.transcription.language,
        || crate::dictation::tidy::current_habits(&settings.tidy),
        crate::dictation::tidy::suggest,
    )
}

/// Opens `session` unless a newer dictation committed while it computed.
fn open(session: TidySession) {
    if session.is_stale(crate::dictation::tidy::last::generation()) {
        tracing::info!("tidy review skipped: a newer dictation committed");
        return;
    }
    crate::rune_ui::tidy_capsule::open(session);
}
