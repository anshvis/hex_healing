//! Labs › Tidy after paste's card while on: its shortcut, off until
//! recorded, which opens the review of the last dictation.

use rune::{El, row, text};

use super::super::controls::{DESCRIPTION, Paint};
use super::super::general::keys_button;
use super::super::{ShortcutTarget, Ui};
use crate::desktop::i18n::tr;

pub(super) fn body(ui: &Ui, p: Paint) -> El {
    row()
        .key("tidy-shortcut")
        .gap(8.0)
        .center_y()
        .child(
            text(tr("Tidy shortcut"))
                .key("label")
                .font(DESCRIPTION)
                .color(p.muted())
                .grow(1.0),
        )
        .child(keys_button(ui, p, ShortcutTarget::Tidy))
}
