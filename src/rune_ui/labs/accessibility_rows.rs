//! The Labs cards that read another app's focused text field through macOS
//! Accessibility: Rewrite reads nearby text and Learn from corrections. Each
//! says what it reads, and that password fields never are.

use rune::Icon;

use super::Row;
use crate::desktop::i18n::tr;
use crate::labs::Feature;

type MakeRow<'a> = &'a dyn Fn(Feature, Option<&'static Icon>, &'static str, &'static str) -> Row;

/// The two rows, in Labs order.
pub(super) fn rows(row: MakeRow) -> Vec<Row> {
    let book = Some(&super::BOOK);
    vec![
        Row {
            note: Some(
                tr("That text goes to the mode's OpenCode model, which may be a cloud provider. Password fields are never read.")
                    .to_owned(),
            ),
            ..row(
                Feature::CursorContext,
                book,
                tr("Rewrite reads nearby text"),
                tr("Rewrite uses the text around the cursor to spell names and terms right and fit where it pastes."),
            )
        },
        Row {
            note: Some(
                tr("Hex reads only that field, for up to 30 seconds, and keeps just the edited span in History. Password fields are never read.")
                    .to_owned(),
            ),
            ..row(
                Feature::LearnFromCorrections,
                book,
                tr("Learn from corrections"),
                tr("When you edit a dictation right after it pastes, Hex notes the change on its History entry."),
            )
        },
    ]
}
