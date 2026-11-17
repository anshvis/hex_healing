//! `hex vocabulary set --file`: reads the list here, in the client, and
//! hands it to the running Hex as one `VocabularySet`.

use std::path::PathBuf;

use clap::Args;

use super::{ControlCommand, ControlReply, Output, print, send};
use crate::control::ControlRequest;
use crate::dictation::vocabulary;
use crate::local_api::MAX_CONTROL_BYTES;

#[derive(Args, Debug)]
pub struct VocabularySetArgs {
    /// A JSON array of terms: each a string, or an object
    /// `{"term": "...", "sounds_like": ["..."]}`.
    #[arg(long, value_name = "PATH")]
    file: PathBuf,
    #[command(flatten)]
    output: Output,
}

pub fn run(VocabularySetArgs { file, output }: VocabularySetArgs) -> u8 {
    let command = match vocabulary::read_list_file(&file) {
        Ok(entries) => ControlCommand::VocabularySet { entries },
        Err(message) => return print(ControlReply::error("invalid-vocabulary", message), output),
    };
    match too_large(&command) {
        Some(message) => print(ControlReply::error("invalid-vocabulary", message), output),
        None => send(command, output),
    }
}

/// The list goes to Hex in one control request, which `/control` caps at
/// 64 KiB; refuse a bigger one here, naming why, rather than as a 413.
fn too_large(command: &ControlCommand) -> Option<String> {
    let request = ControlRequest::new(command.clone());
    let bytes = serde_json::to_vec(&request).map_or(usize::MAX, |bytes| bytes.len());
    (bytes > MAX_CONTROL_BYTES).then(|| {
        format!(
            "the list is {} KiB, over the {} KiB a control request can carry; use fewer terms or spellings",
            bytes.div_ceil(1024),
            MAX_CONTROL_BYTES / 1024
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictation::vocabulary::VocabularyEntry;

    fn list(terms: usize, spellings: usize) -> ControlCommand {
        let entries = (0..terms)
            .map(|t| VocabularyEntry {
                term: format!("term{t:0>59}"),
                sounds_like: (0..spellings)
                    .map(|s| format!("s{s:0>2}{t:0>61}"))
                    .collect(),
            })
            .collect();
        ControlCommand::VocabularySet { entries }
    }

    #[test]
    fn a_list_over_the_control_limit_is_refused_before_sending() {
        let message = too_large(&list(100, 16)).expect("100 full entries exceed 64 KiB");
        assert!(message.contains("64 KiB"), "{message}");
        assert_eq!(too_large(&list(20, 2)), None);
    }
}
