//! What Labs › Tidy after paste needs from Accessibility to edit the field
//! Hex pasted into: the secure-field check, a bounded read of the whole
//! value in UTF-16 units, whether an attribute is settable, the selected
//! range and selected text, and whether the field still has the focus.
//! Nothing here logs or keeps text.

use std::ffi::{CStr, c_void};
use std::ops::Range;

use super::watch::{ReadError, read_attribute};
use super::{
    AX_VALUE_CF_RANGE, AXUIElementSetAttributeValue, AXValueCreate, AxUiElementRef, CFEqual,
    CFGetTypeID, CFStringGetCharacters, CFStringGetLength, CFStringGetTypeID, CfObject, CfRange,
    CfStringRef, RetainedElement, cf_string_literal, copy_element, copy_number,
    foreign_application, is_secure_text_field,
};

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXUIElementIsAttributeSettable(
        element: AxUiElementRef,
        attribute: CfStringRef,
        settable: *mut u8,
    ) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCharacters(
        allocator: *const c_void,
        characters: *const u16,
        length: isize,
    ) -> CfStringRef;
}

/// Whether the element is a secure (password) field, asked before anything
/// else is read. A gone or unanswering element is an error, not "no".
pub fn is_secure(element: &RetainedElement) -> Result<bool, ReadError> {
    // The liveness probe: a gone element answers "invalid" here.
    read_attribute(&element.0, c"AXRole")?;
    Ok(is_secure_text_field(&element.0))
}

/// The field's whole value in UTF-16 units, when it holds at most
/// `max_units`; the size is known before any text is copied. A secure field
/// is never read.
pub fn read_units(element: &RetainedElement, max_units: usize) -> Result<Vec<u16>, ReadError> {
    let element = &element.0;
    read_attribute(element, c"AXRole")?;
    if is_secure_text_field(element) {
        return Err(ReadError::Secure);
    }
    if copy_number(element, c"AXNumberOfCharacters").is_some_and(|count| count > max_units as i64) {
        return Err(ReadError::TooLarge);
    }
    let Some(value) = read_attribute(element, c"AXValue")? else {
        return Err(ReadError::Failed);
    };
    // SAFETY: `value` is a live Core Foundation object.
    if unsafe { CFGetTypeID(value.0) != CFStringGetTypeID() } {
        return Err(ReadError::Failed);
    }
    // SAFETY: checked as a string just above.
    let length = unsafe { CFStringGetLength(value.0) }.max(0);
    if length as usize > max_units {
        return Err(ReadError::TooLarge);
    }
    let mut units = vec![0_u16; length as usize];
    // SAFETY: the range is the string's whole length and `units` holds it.
    unsafe {
        CFStringGetCharacters(
            value.0,
            CfRange {
                location: 0,
                length,
            },
            units.as_mut_ptr(),
        )
    };
    Ok(units)
}

/// Whether the app lets `name` be set on the element.
pub fn is_settable(element: &RetainedElement, name: &CStr) -> bool {
    let Some(attribute) = cf_string_literal(name).map(CfObject) else {
        return false;
    };
    let mut settable = 0_u8;
    // SAFETY: a live element, a CFString name, and a Boolean out-parameter.
    let status = unsafe { AXUIElementIsAttributeSettable(element.0.0, attribute.0, &mut settable) };
    status == 0 && settable != 0
}

/// Selects `range` (UTF-16 units) in the field.
pub fn select(element: &RetainedElement, range: Range<usize>) -> bool {
    let range = CfRange {
        location: range.start as isize,
        length: range.len() as isize,
    };
    // SAFETY: AXValueCreate copies the CFRange and returns a retained object.
    let value = unsafe { AXValueCreate(AX_VALUE_CF_RANGE, (&range as *const CfRange).cast()) };
    if value.is_null() {
        return false;
    }
    set(element, c"AXSelectedTextRange", &CfObject(value))
}

/// Replaces the field's selection with `text`, as typing would: one undo
/// step in the fields that support it.
pub fn set_selected_text(element: &RetainedElement, text: &str) -> bool {
    let units: Vec<u16> = text.encode_utf16().collect();
    // SAFETY: `units` holds `units.len()` UTF-16 units; the result is retained.
    let string = unsafe {
        CFStringCreateWithCharacters(std::ptr::null(), units.as_ptr(), units.len() as isize)
    };
    if string.is_null() {
        return false;
    }
    set(element, c"AXSelectedText", &CfObject(string))
}

fn set(element: &RetainedElement, name: &CStr, value: &CfObject) -> bool {
    let Some(attribute) = cf_string_literal(name).map(CfObject) else {
        return false;
    };
    // SAFETY: a live element, a CFString name, and a live value.
    unsafe { AXUIElementSetAttributeValue(element.0.0, attribute.0, value.0) == 0 }
}

/// Whether `element` is the focused element of the app `pid`.
pub fn is_focused(pid: i32, element: &RetainedElement) -> bool {
    let Some(application) = foreign_application(pid) else {
        return false;
    };
    copy_element(&application, c"AXFocusedUIElement")
        // SAFETY: both are live AXUIElements.
        .is_some_and(|focused| unsafe { CFEqual(focused.0, element.0.0) } != 0)
}
