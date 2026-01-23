//! APR Explorer - Interactive model format explorer in WASM
//!
//! Architecture follows probar's zero-business-logic-in-JS pattern:
//! - All parsing/display logic in pure Rust (testable via cargo test)
//! - MockDom for browser-independent testing
//! - Driver pattern for unified TUI/WASM testing
//!
//! Probar: Zero external JS dependencies for business logic

mod apr;
mod dom;
mod driver;

#[cfg(feature = "wasm")]
mod browser;

pub use apr::{
    create_test_apr1, parse_apr, AprError, AprHeader, AprModel, Compression, ModelMetadata,
    TensorInfo, APR_MAGIC_COMPRESSED, APR_MAGIC_UNCOMPRESSED,
};
pub use dom::{DomElement, DomEvent, MockDom};
pub use driver::{
    create_test_apr_bytes, run_full_specification, verify_apr_loading, verify_clear,
    verify_dom_sync, verify_error_handling, verify_tensor_info, AprExplorerDriver,
};

#[cfg(feature = "wasm")]
pub use browser::BrowserExplorer;
