//! What the correction watcher needs from Accessibility: a bounded read of a
//! retained element's value and an `AXObserver` that reports its changes.
//! Nothing here logs or keeps text.

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use super::{
    AxUiElementRef, CfObject, CfRange, CfStringRef, CfTypeRef, Keep, RetainedElement, cf_string,
    cf_string_literal, copy_number, copy_range, copy_text_for_range, foreign_application,
    is_secure_text_field,
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
}

const AX_ERROR_INVALID_UI_ELEMENT: i32 = -25202;
const AX_ERROR_ATTRIBUTE_UNSUPPORTED: i32 = -25205;
const AX_ERROR_NO_VALUE: i32 = -25212;

unsafe extern "C" {
    fn AXUIElementCopyAttributeValue(
        element: AxUiElementRef,
        attribute: CfStringRef,
        value: *mut CfTypeRef,
    ) -> i32;
}

/// An attribute with its failure kept: `Ok(None)` when the element has none.
fn read_attribute(
    element: &CfObject,
    name: &std::ffi::CStr,
) -> Result<Option<CfObject>, ReadError> {
    let attribute = CfObject(cf_string_literal(name).ok_or(ReadError::Timeout)?);
    let mut value = std::ptr::null();
    // SAFETY: both objects are live and `value` receives a retained object.
    let status = unsafe { AXUIElementCopyAttributeValue(element.0, attribute.0, &mut value) };
    match status {
        0 if !value.is_null() => Ok(Some(CfObject(value))),
        0 | AX_ERROR_NO_VALUE | AX_ERROR_ATTRIBUTE_UNSUPPORTED => Ok(None),
        AX_ERROR_INVALID_UI_ELEMENT => Err(ReadError::ElementGone),
        _ => Err(ReadError::Timeout),
    }
}

/// The field's text: the whole `AXValue` when it has at most `max_chars`
/// characters, else `max_chars` around the caret (half before, half after).
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
        let caret = copy_range(element, c"AXSelectedTextRange").ok_or(ReadError::TooLarge)?;
        let start = (caret.location - (max_chars / 2) as isize).max(0);
        let length = (characters as isize - start).min(max_chars as isize);
        return copy_text_for_range(
            element,
            CfRange {
                location: start,
                length,
            },
            Keep::Head,
        )
        .ok_or(ReadError::Timeout);
    }
    let Some(value) = read_attribute(element, c"AXValue")? else {
        return Ok(String::new());
    };
    let Some(text) = cf_string(value.0) else {
        return Ok(String::new());
    };
    if text.chars().count() > max_chars {
        return Err(ReadError::TooLarge);
    }
    Ok(text)
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
    fn CGEventSourceKeyState(state: u32, key: u16) -> bool;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFRunLoopDefaultMode: CfStringRef;
    fn CFRunLoopGetCurrent() -> *mut c_void;
    // `source` is `*mut` to match `suppression.rs`'s declaration of the same symbols.
    fn CFRunLoopAddSource(run_loop: *mut c_void, source: *mut c_void, mode: CfStringRef);
    fn CFRunLoopRemoveSource(run_loop: *mut c_void, source: *mut c_void, mode: CfStringRef);
    fn CFRunLoopRunInMode(mode: CfStringRef, seconds: f64, return_after_source: u8) -> i32;
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
        Some("AXFocusedUIElementChanged") => Notification::FocusChanged,
        _ => return,
    };
    inbox.lock().unwrap_or_else(|e| e.into_inner()).push(kind);
}

/// An `AXObserver` on one target app for its element's value changes and
/// the app's focus changes. Its source runs on the run loop of the thread
/// that created it; callbacks fire only while that thread runs the loop
/// ([`run_loop_once`]). Dropping it unregisters everything.
pub struct Observer {
    observer: CfObject,
    source: CfTypeRef,
    run_loop: *mut c_void,
    application: CfObject,
    element: CfObject,
    inbox: *const Inbox,
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
        let inbox = Arc::into_raw(Arc::new(Inbox::default()));
        // SAFETY: the element is retained here so it outlives the observer.
        let element = CfObject(unsafe { super::CFRetain(field.0.0) });
        let mut this = Self {
            observer,
            source: std::ptr::null(),
            run_loop: std::ptr::null_mut(),
            application,
            element,
            inbox,
        };
        let value = cf_string_literal(c"AXValueChanged")?;
        let focus = cf_string_literal(c"AXFocusedUIElementChanged")?;
        let (value, focus) = (CfObject(value), CfObject(focus));
        // SAFETY: the refcon outlives every registration (see `Drop`).
        let (added_value, added_focus) = unsafe {
            (
                AXObserverAddNotification(
                    this.observer.0,
                    this.element.0,
                    value.0,
                    inbox as *mut _,
                ),
                AXObserverAddNotification(
                    this.observer.0,
                    this.application.0,
                    focus.0,
                    inbox as *mut _,
                ),
            )
        };
        // The value registration is best effort: a field that refuses it is
        // still read on focus out, Return and the deadline.
        let _ = added_value;
        if added_focus != 0 && added_value != 0 {
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
        for (target, name) in [
            (&self.element, c"AXValueChanged"),
            (&self.application, c"AXFocusedUIElementChanged"),
        ] {
            if let Some(name) = cf_string_literal(name).map(CfObject) {
                // SAFETY: both are live; failure (an element gone) is fine.
                unsafe { AXObserverRemoveNotification(self.observer.0, target.0, name.0) };
            }
        }
        // SAFETY: balances `Arc::into_raw`; no callback can run any more.
        drop(unsafe { Arc::from_raw(self.inbox) });
    }
}

/// Runs the current thread's run loop for at most `seconds`, delivering
/// observer callbacks. Returns `false` at once when it has no sources.
pub fn run_loop_once(seconds: f64) -> bool {
    // SAFETY: the default mode constant is always valid.
    unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, seconds, 0) != 1 }
}

/// Whether Return or keypad Enter is down now. A key-state query, not an
/// event tap: it needs no Input Monitoring and sees no other key.
pub fn return_down() -> bool {
    // SAFETY: `kCGEventSourceStateCombinedSessionState` is 1; 0x24 and 0x4C
    // are Return and keypad Enter.
    unsafe { CGEventSourceKeyState(1, 0x24) || CGEventSourceKeyState(1, 0x4C) }
}
