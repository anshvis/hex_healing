//! What the correction watcher needs from Accessibility: a bounded read of a
//! retained element's value and an `AXObserver` that reports its changes.
//! Nothing here logs or keeps text.

use std::ffi::c_void;
use std::sync::Mutex;

use super::{
    AxUiElementRef, CFGetTypeID, CFStringGetLength, CFStringGetTypeID, CfObject, CfRange,
    CfStringRef, CfTypeRef, Keep, RetainedElement, cf_string, cf_string_literal, copy_number,
    copy_range, copy_text_for_range, fetch_attribute, foreign_application, is_secure_text_field,
};
use crate::suppression::run_loop::{
    CFRunLoopAddSource, CFRunLoopGetCurrent, CFRunLoopRemoveSource, CFRunLoopRunInMode,
    kCFRunLoopDefaultMode,
};

/// Why a bounded read of the field gave no text.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReadError {
    /// The element no longer exists.
    ElementGone,
    /// A secure (password) field.
    Secure,
    /// Over the bound with no caret to window around.
    TooLarge,
    /// The app did not answer within the messaging timeout.
    Timeout,
    /// Any other Accessibility failure.
    Failed,
}

const AX_ERROR_INVALID_UI_ELEMENT: i32 = -25202;
const AX_ERROR_CANNOT_COMPLETE: i32 = -25204;
const AX_ERROR_ATTRIBUTE_UNSUPPORTED: i32 = -25205;
const AX_ERROR_NO_VALUE: i32 = -25212;

/// An attribute with its failure kept: `Ok(None)` when the element has none.
fn read_attribute(
    element: &CfObject,
    name: &std::ffi::CStr,
) -> Result<Option<CfObject>, ReadError> {
    let (status, value) = fetch_attribute(element, name).ok_or(ReadError::Failed)?;
    match status {
        0 => Ok(value),
        AX_ERROR_NO_VALUE | AX_ERROR_ATTRIBUTE_UNSUPPORTED => Ok(None),
        AX_ERROR_INVALID_UI_ELEMENT => Err(ReadError::ElementGone),
        AX_ERROR_CANNOT_COMPLETE => Err(ReadError::Timeout),
        _ => Err(ReadError::Failed),
    }
}

/// `max_chars` of a field of `characters` UTF-16 units around the caret (half
/// before, half after).
fn read_around_caret(
    element: &CfObject,
    characters: i64,
    max_chars: usize,
) -> Result<String, ReadError> {
    let caret = copy_range(element, c"AXSelectedTextRange").ok_or(ReadError::TooLarge)?;
    let start = (caret.location - (max_chars / 2) as isize).max(0);
    let length = (characters as isize - start).min(max_chars as isize);
    copy_text_for_range(
        element,
        CfRange {
            location: start,
            length,
        },
        Keep::Head,
    )
    .ok_or(ReadError::Failed)
}

/// The field's text: the whole `AXValue` when it has at most `max_chars`
/// characters, else `max_chars` around the caret. The size is known before
/// any text is copied: from `AXNumberOfCharacters`, else from the value's
/// `CFStringGetLength`.
pub fn read_value_bounded(
    element: &RetainedElement,
    max_chars: usize,
) -> Result<String, ReadError> {
    let element = &element.0;
    read_attribute(element, c"AXRole")?;
    if is_secure_text_field(element) {
        return Err(ReadError::Secure);
    }
    if let Some(characters) = copy_number(element, c"AXNumberOfCharacters")
        && characters > max_chars as i64
    {
        return read_around_caret(element, characters, max_chars);
    }
    let Some(value) = read_attribute(element, c"AXValue")? else {
        return Ok(String::new());
    };
    // SAFETY: `value` is a live Core Foundation object.
    if unsafe { CFGetTypeID(value.0) != CFStringGetTypeID() } {
        return Ok(String::new());
    }
    // SAFETY: the type was checked as a string just above.
    let length = unsafe { CFStringGetLength(value.0) }.max(0);
    if length as usize > max_chars {
        return read_around_caret(element, length as i64, max_chars);
    }
    Ok(cf_string(value.0).unwrap_or_default())
}

/// What an observer reported.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Notification {
    ValueChanged,
    FocusChanged,
}

type AxObserverRef = *const c_void;
type ObserverCallback =
    unsafe extern "C" fn(AxObserverRef, AxUiElementRef, CfStringRef, *mut c_void);

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXObserverCreate(
        application: i32,
        callback: ObserverCallback,
        observer: *mut AxObserverRef,
    ) -> i32;
    fn AXObserverAddNotification(
        observer: AxObserverRef,
        element: AxUiElementRef,
        notification: CfStringRef,
        refcon: *mut c_void,
    ) -> i32;
    fn AXObserverRemoveNotification(
        observer: AxObserverRef,
        element: AxUiElementRef,
        notification: CfStringRef,
    ) -> i32;
    fn AXObserverGetRunLoopSource(observer: AxObserverRef) -> CfTypeRef;
}

type Inbox = Mutex<Vec<Notification>>;

unsafe extern "C" fn deliver(
    _observer: AxObserverRef,
    _element: AxUiElementRef,
    notification: CfStringRef,
    refcon: *mut c_void,
) {
    // SAFETY: `refcon` is the `Inbox` the observer owns until it is dropped,
    // and the source is removed before that.
    let inbox = unsafe { &*(refcon as *const Inbox) };
    let kind = match cf_string(notification).as_deref() {
        Some("AXValueChanged") => Notification::ValueChanged,
        Some("AXFocusedUIElementChanged" | "AXApplicationDeactivated") => {
            Notification::FocusChanged
        }
        _ => return,
    };
    inbox.lock().unwrap_or_else(|e| e.into_inner()).push(kind);
}

/// An `AXObserver` on one target app for its element's value changes and
/// the app's focus changes and deactivation (switching away from the app
/// leaves its focused element as it was). Its source runs on the run loop of the thread
/// that created it; callbacks fire only while that thread runs the loop
/// ([`run_loop_once`]). Dropping it unregisters everything.
pub struct Observer {
    observer: CfObject,
    source: CfTypeRef,
    run_loop: *mut c_void,
    application: CfObject,
    element: CfObject,
    inbox: *const Inbox,
    value_registered: bool,
}

impl Observer {
    /// `None` for Hex's own pid or an app that refuses an observer.
    pub fn new(pid: i32, field: &RetainedElement) -> Option<Self> {
        let application = foreign_application(pid)?;
        let mut observer = std::ptr::null();
        // SAFETY: `observer` receives a retained observer on success.
        if unsafe { AXObserverCreate(pid, deliver, &mut observer) } != 0 || observer.is_null() {
            return None;
        }
        let observer = CfObject(observer);
        let inbox = Box::into_raw(Box::new(Inbox::default()));
        // SAFETY: the element is retained here so it outlives the observer.
        let element = CfObject(unsafe { super::CFRetain(field.0.0) });
        let mut this = Self {
            observer,
            source: std::ptr::null(),
            run_loop: std::ptr::null_mut(),
            application,
            element,
            inbox,
            value_registered: false,
        };
        // The value registration is best effort: a field that refuses it is
        // polled by the watcher ([`Observer::watches_value`]).
        let mut registered = 0;
        let mut value_registered = false;
        for (target, name) in this.registrations() {
            let name_is_value = name == c"AXValueChanged";
            let name = CfObject(cf_string_literal(name)?);
            // SAFETY: the refcon outlives every registration (see `Drop`).
            let status = unsafe {
                AXObserverAddNotification(this.observer.0, target.0, name.0, inbox as *mut _)
            };
            registered += usize::from(status == 0);
            value_registered |= name_is_value && status == 0;
        }
        this.value_registered = value_registered;
        if registered == 0 {
            return None;
        }
        // SAFETY: the source is owned by the observer and not retained here.
        let source = unsafe { AXObserverGetRunLoopSource(this.observer.0) };
        if source.is_null() {
            return None;
        }
        let run_loop = unsafe { CFRunLoopGetCurrent() };
        unsafe { CFRunLoopAddSource(run_loop, source.cast_mut(), kCFRunLoopDefaultMode) };
        this.source = source;
        this.run_loop = run_loop;
        Some(this)
    }

    /// Whether the element accepted the value-change registration. If not,
    /// the watcher polls instead.
    pub fn watches_value(&self) -> bool {
        self.value_registered
    }

    /// Each notification and the object it is registered on.
    fn registrations(&self) -> [(&CfObject, &'static std::ffi::CStr); 3] {
        [
            (&self.element, c"AXValueChanged"),
            (&self.application, c"AXFocusedUIElementChanged"),
            (&self.application, c"AXApplicationDeactivated"),
        ]
    }

    /// Everything reported since the last call.
    pub fn drain(&self) -> Vec<Notification> {
        // SAFETY: `inbox` lives until `Drop`.
        let inbox = unsafe { &*self.inbox };
        std::mem::take(&mut *inbox.lock().unwrap_or_else(|e| e.into_inner()))
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        if !self.source.is_null() {
            // SAFETY: removed from the run loop it was added to, on its thread.
            unsafe {
                CFRunLoopRemoveSource(self.run_loop, self.source.cast_mut(), kCFRunLoopDefaultMode)
            };
        }
        for (target, name) in self.registrations() {
            if let Some(name) = cf_string_literal(name).map(CfObject) {
                // SAFETY: both are live; failure (an element gone) is fine.
                unsafe { AXObserverRemoveNotification(self.observer.0, target.0, name.0) };
            }
        }
        // SAFETY: balances `Box::into_raw`; no callback can run any more.
        drop(unsafe { Box::from_raw(self.inbox.cast_mut()) });
    }
}

/// Runs the current thread's run loop for at most `seconds`, delivering
/// observer callbacks. Returns `false` at once when it has no sources.
pub fn run_loop_once(seconds: f64) -> bool {
    // SAFETY: the default mode constant is always valid.
    unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, 0) != 1 }
}
