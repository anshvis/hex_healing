//! `hex vocabulary set --file`: reads the list here, in the client, and
//! hands it to the running Hex as one `VocabularySet`.

use std::path::PathBuf;

use clap::Args;

use super::{ControlCommand, ControlReply, Output, print, send};
use crate::dictation::vocabulary;

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
    match vocabulary::read_list_file(&file) {
        Ok(entries) => send(ControlCommand::VocabularySet { entries }, output),
        Err(message) => print(ControlReply::error("invalid-vocabulary", message), output),
    }
}
