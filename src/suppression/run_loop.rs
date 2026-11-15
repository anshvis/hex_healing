//! The Core Foundation run-loop calls the event taps, the HID reader and the
//! Accessibility observer thread share: one declaration of each.

use std::ffi::c_void;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    pub(crate) fn CFRunLoopGetCurrent() -> *mut c_void;
    pub(crate) fn CFRunLoopAddSource(
        run_loop: *mut c_void,
        source: *mut c_void,
        mode: *const c_void,
    );
    pub(crate) fn CFRunLoopRemoveSource(
        run_loop: *mut c_void,
        source: *mut c_void,
        mode: *const c_void,
    );
    pub(crate) fn CFRunLoopRunInMode(
        mode: *const c_void,
        seconds: f64,
        return_after_source_handled: u8,
    ) -> i32;
    pub(crate) fn CFRunLoopStop(run_loop: *mut c_void);
    pub(crate) static kCFRunLoopCommonModes: *const c_void;
    pub(crate) static kCFRunLoopDefaultMode: *const c_void;
}
