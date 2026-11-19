//! The person's filler habits (T8.2) for tidy's rule pass, read from the
//! file `tidy.habits_file` names: `healing tidy habits` writes it from the
//! suggestions they accepted and skipped. It is read at each tidy, so a new
//! export applies to the next one.
//!
//! ```json
//! {"version": 1, "habits": [{"phrase": "basically", "accept_rate": 0.8}],
//!  "trial": ["kind of"]}
//! ```
//!
//! `habits` are phrases accepted at least 70% of the time, ranked by that
//! rate, which becomes their confidence (D-64); `trial` phrases are still
//! being learned and rank below every habit (D-65). A missing, oversized, or
//! invalid file means no habits; the log names only the reason.

use std::io::Read;
use std::path::Path;

use serde::Deserialize;

use super::{Habit, Habits};

/// The largest file read at all.
const MAX_FILE_BYTES: u64 = 64 * 1024;
/// The most phrases, habits and trial together.
const MAX_PHRASES: usize = 64;
const MAX_PHRASE_BYTES: usize = 32;
const MAX_PHRASE_WORDS: usize = 4;
const VERSION: u32 = 1;
/// A trial phrase's confidence: below every habit, which needs an accept
/// rate of at least 0.7.
pub(super) const TRIAL_CONFIDENCE: f32 = 0.5;
/// The most a habit's confidence reaches, so a built-in filler still ranks
/// with or above it.
const MAX_HABIT_CONFIDENCE: f32 = super::rules::FILLER_CONFIDENCE;
const MIN_HABIT_CONFIDENCE: f32 = 0.7;

#[derive(Deserialize)]
struct HabitsFile {
    version: u32,
    habits: Vec<Entry>,
    #[serde(default)]
    trial: Vec<String>,
}

#[derive(Deserialize)]
struct Entry {
    phrase: String,
    accept_rate: f32,
}

/// The habits in `file`, or none when it is unset, missing, too big, or not
/// a habits file Hex can read. Ranked: habits by accept rate, then trial.
pub(super) fn load_habits(file: Option<&Path>) -> Habits {
    let Some(path) = file else {
        return Habits::default();
    };
    match read(path) {
        Ok(habits) => habits,
        Err(reason) => {
            tracing::info!(reason, "tidy habits not used");
            Habits::default()
        }
    }
}

fn read(path: &Path) -> Result<Habits, &'static str> {
    let file = std::fs::File::open(path).map_err(|error| match error.kind() {
        std::io::ErrorKind::NotFound => "missing",
        _ => "unreadable",
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "unreadable")?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err("oversized");
    }
    parse(&bytes)
}

fn parse(bytes: &[u8]) -> Result<Habits, &'static str> {
    let file: HabitsFile = serde_json::from_slice(bytes).map_err(|_| "malformed")?;
    if file.version != VERSION {
        return Err("unknown version");
    }
    if file.habits.len() + file.trial.len() > MAX_PHRASES {
        return Err("too many phrases");
    }
    let mut phrases = Vec::with_capacity(file.habits.len() + file.trial.len());
    for entry in &file.habits {
        if !entry.accept_rate.is_finite() || !(0.0..=1.0).contains(&entry.accept_rate) {
            return Err("invalid accept rate");
        }
        phrases.push(Habit {
            phrase: phrase(&entry.phrase)?,
            confidence: entry
                .accept_rate
                .clamp(MIN_HABIT_CONFIDENCE, MAX_HABIT_CONFIDENCE),
        });
    }
    for trial in &file.trial {
        phrases.push(Habit {
            phrase: phrase(trial)?,
            confidence: TRIAL_CONFIDENCE,
        });
    }
    phrases.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
    Ok(Habits { phrases })
}

/// A phrase of one to four words of letters and apostrophes, at most 32
/// bytes, lowercased as the rule pass compares words.
fn phrase(raw: &str) -> Result<String, &'static str> {
    let words: Vec<&str> = raw.split_whitespace().collect();
    let plain = words.iter().all(|word| {
        word.chars()
            .all(|character| character.is_alphabetic() || super::rules::is_apostrophe(character))
    });
    if words.is_empty() || words.len() > MAX_PHRASE_WORDS || raw.len() > MAX_PHRASE_BYTES || !plain
    {
        return Err("invalid phrase");
    }
    Ok(words.join(" ").to_lowercase())
}

/// A habits file with `contents` in a folder of its own under the system's
/// temporary folder.
#[cfg(test)]
pub(super) fn scratch_file(contents: &[u8]) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "hex-tidy-habits-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("habits.json");
    std::fs::write(&path, contents).unwrap();
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phrases(habits: &Habits) -> Vec<(&str, f32)> {
        habits
            .phrases
            .iter()
            .map(|habit| (habit.phrase.as_str(), habit.confidence))
            .collect()
    }

    #[test]
    fn habits_rank_by_accept_rate_above_trial() {
        let habits = parse(
            br#"{"version": 1, "generated_at_ms": 1,
                 "habits": [{"phrase": "kind of", "accept_rate": 0.7},
                            {"phrase": "Basically", "accept_rate": 0.8},
                            {"phrase": "you know", "accept_rate": 1.0}],
                 "trial": ["I guess"]}"#,
        )
        .unwrap();
        assert_eq!(
            phrases(&habits),
            [
                ("you know", MAX_HABIT_CONFIDENCE),
                ("basically", 0.8),
                ("kind of", 0.7),
                ("i guess", TRIAL_CONFIDENCE),
            ]
        );
    }

    #[test]
    fn missing_invalid_or_oversized_file_means_no_habits() {
        assert!(load_habits(None).phrases.is_empty());
        let absent = scratch_file(b"").with_file_name("absent.json");
        assert!(load_habits(Some(&absent)).phrases.is_empty());
        let oversized = format!(
            r#"{{"version": 1, "habits": [], "trial": [], "pad": "{}"}}"#,
            " ".repeat(MAX_FILE_BYTES as usize)
        );
        let many = format!(
            r#"{{"version": 1, "habits": [], "trial": [{}]}}"#,
            vec![r#""like""#; MAX_PHRASES + 1].join(",")
        );
        for contents in [
            "not json".to_owned(),
            r#"{"version": 2, "habits": []}"#.to_owned(),
            r#"{"habits": []}"#.to_owned(),
            r#"{"version": 1, "habits": [{"phrase": "like", "accept_rate": 1.5}]}"#.to_owned(),
            r#"{"version": 1, "habits": [{"phrase": "", "accept_rate": 0.9}]}"#.to_owned(),
            r#"{"version": 1, "habits": [{"phrase": "a b c d e", "accept_rate": 0.9}]}"#.to_owned(),
            r#"{"version": 1, "habits": [{"phrase": "5 times", "accept_rate": 0.9}]}"#.to_owned(),
            r#"{"version": 1, "habits": [], "trial": ["like,"]}"#.to_owned(),
            format!(
                r#"{{"version": 1, "habits": [], "trial": ["{}"]}}"#,
                "a".repeat(MAX_PHRASE_BYTES + 1)
            ),
            oversized,
            many,
        ] {
            let path = scratch_file(contents.as_bytes());
            assert!(
                load_habits(Some(&path)).phrases.is_empty(),
                "{}",
                &contents[..contents.len().min(80)]
            );
        }
        let path = scratch_file(
            br#"{"version": 1, "habits": [{"phrase": "basically", "accept_rate": 0.8}]}"#,
        );
        assert_eq!(phrases(&load_habits(Some(&path))), [("basically", 0.8)]);
    }
}
