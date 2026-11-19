//! `hex dictation <start|stop|toggle|cancel|tidy>`: the capture actions the
//! shortcut has, plus Labs › Tidy after paste's review of the last one.

use super::{Output, send};
use crate::control::{CaptureAction, ControlCommand};

/// What `hex dictation` does.
#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum Verb {
    /// Start recording, locked as after a double tap: the shortcut or
    /// `stop` finishes it.
    Start,
    /// Finish the recording and transcribe it.
    Stop,
    /// Start when idle, otherwise stop.
    Toggle,
    /// Discard the recording: nothing is transcribed, pasted, or saved.
    /// Dictations already transcribing finish.
    Cancel,
    /// Review suggested edits to the last dictation (Labs › Tidy after
    /// paste, macOS); with --print, print them as JSON instead.
    Tidy,
}

impl Verb {
    fn capture(self) -> Option<CaptureAction> {
        match self {
            Self::Start => Some(CaptureAction::Start),
            Self::Stop => Some(CaptureAction::Stop),
            Self::Toggle => Some(CaptureAction::Toggle),
            Self::Cancel => Some(CaptureAction::Cancel),
            Self::Tidy => None,
        }
    }
}

/// The request `hex dictation` sends, or the usage error to print.
pub(super) fn command(verb: Verb, print: bool, raw: bool) -> Result<ControlCommand, String> {
    let Some(action) = verb.capture() else {
        if raw {
            return Err(
                "--raw works only with `hex dictation start` and `hex dictation toggle`".into(),
            );
        }
        return Ok(ControlCommand::DictationTidy { print });
    };
    crate::control::check_flags(action, print, raw).map_err(|error| error.message)?;
    Ok(ControlCommand::Dictation { action, print, raw })
}

pub(super) fn run(verb: Verb, print: bool, raw: bool, output: Output) -> u8 {
    match command(verb, print, raw) {
        Ok(command) => send(command, output),
        Err(message) => {
            eprintln!("hex: {message}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tidy_sends_its_own_command_and_print_is_its_json() {
        assert_eq!(
            command(Verb::Tidy, false, false),
            Ok(ControlCommand::DictationTidy { print: false })
        );
        assert_eq!(
            command(Verb::Tidy, true, false),
            Ok(ControlCommand::DictationTidy { print: true })
        );
        assert!(command(Verb::Tidy, false, true).is_err());
        assert_eq!(
            command(Verb::Stop, true, false),
            Ok(ControlCommand::Dictation {
                action: CaptureAction::Stop,
                print: true,
                raw: false,
            })
        );
        assert!(command(Verb::Cancel, true, false).is_err());
    }
}
