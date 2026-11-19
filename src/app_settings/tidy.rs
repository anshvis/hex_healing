//! Labs › Tidy after paste's settings (M8): the habits file and, beside it
//! in [`AppSettings`], the Tidy shortcut.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{AppSettings, HotkeyBinding, HotkeySlot};

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct TidySettings {
    /// The filler habits file `healing tidy habits` writes (T8.2), read at
    /// each tidy. Unset means no habits; left out of the file while unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub habits_file: Option<PathBuf>,
}

impl TidySettings {
    /// Unchanged from the default, so `settings.json` leaves `tidy` out and
    /// files stay as older builds wrote them.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

impl AppSettings {
    /// The Tidy shortcut while it takes effect: macOS, Labs › Tidy after
    /// paste on, bound, and sharing no key with another active shortcut.
    /// Otherwise Hex matches, suppresses, and reserves nothing for it, as
    /// with an unbound Dictate without processing.
    pub fn active_tidy_hotkey(&self) -> Option<&HotkeyBinding> {
        if !cfg!(target_os = "macos") || !self.labs.tidy_after_paste {
            return None;
        }
        self.tidy_hotkey
            .as_ref()
            .filter(|binding| !self.hotkey_in_use(HotkeySlot::Tidy, binding))
    }

    /// A left-click Tidy shortcut would take every click; it is cleared.
    pub(super) fn clear_left_click_tidy(&mut self) {
        if self
            .tidy_hotkey
            .as_ref()
            .is_some_and(HotkeyBinding::is_bare_primary_click)
        {
            tracing::warn!("cleared a left-click tidy shortcut");
            self.tidy_hotkey = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_settings::{HotkeyKey, HotkeyModifiers};

    fn option_t() -> HotkeyBinding {
        HotkeyBinding {
            modifiers: HotkeyModifiers::option(),
            key: Some(HotkeyKey::named("t")),
        }
    }

    /// Unbound by default, saved and read back as recorded, and active only
    /// on macOS with its lab on.
    #[test]
    fn tidy_hotkey_round_trips_and_waits_for_its_lab() {
        let mut settings = AppSettings::default();
        assert_eq!(settings.tidy_hotkey, None);
        assert_eq!(settings.active_tidy_hotkey(), None);
        assert_eq!(settings.runtime_hotkeys().tidy, None);

        settings.tidy_hotkey = Some(option_t());
        let json = serde_json::to_string(&settings).unwrap();
        let loaded: AppSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.tidy_hotkey, Some(option_t()));
        // Bound but the lab off: nothing is matched or reserved.
        assert_eq!(loaded.active_tidy_hotkey(), None);
        assert_eq!(loaded.runtime_hotkeys().tidy, None);

        settings.labs.tidy_after_paste = true;
        assert_eq!(
            settings.active_tidy_hotkey().is_some(),
            cfg!(target_os = "macos")
        );
        assert_eq!(
            settings.runtime_hotkeys().tidy.is_some(),
            cfg!(target_os = "macos")
        );
    }

    /// The Tidy shortcut follows the conflict rules of the other bindings:
    /// it never shares a key with dictation, and while it takes effect no
    /// other shortcut may take its key.
    #[test]
    fn tidy_hotkey_conflicts_like_the_others() {
        let mut settings = AppSettings::default();
        settings.labs.tidy_after_paste = true;
        let dictation = settings.dictation_hotkey.clone();
        settings.tidy_hotkey = Some(dictation.clone());
        assert!(settings.hotkey_in_use(HotkeySlot::Tidy, &dictation));
        assert_eq!(settings.active_tidy_hotkey(), None);

        settings.tidy_hotkey = Some(option_t());
        assert_eq!(
            settings.hotkey_in_use(HotkeySlot::RawDictation, &option_t()),
            cfg!(target_os = "macos")
        );
        // Off, its saved key is free for another shortcut.
        settings.labs.tidy_after_paste = false;
        assert!(!settings.hotkey_in_use(HotkeySlot::RawDictation, &option_t()));
    }
}
