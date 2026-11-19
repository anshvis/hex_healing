//! Labs › Tidy after paste's settings (M8).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
