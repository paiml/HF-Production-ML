//! APR Explorer Driver - Unified Testing Interface
//!
//! This module implements the driver pattern for testing,
//! enabling the same test specifications to run on both
//! native Rust tests and browser WASM.
//!
//! Probar: Balanced testing - Same tests across platforms

use crate::apr::{create_test_apr1, parse_apr, AprError, AprModel, ModelMetadata, TensorInfo};
use crate::dom::{DomElement, DomEvent, MockDom};
use serde_json::json;
use std::collections::BTreeMap;

/// APR Explorer state
#[derive(Debug)]
pub struct AprExplorerDriver {
    /// Current loaded model
    model: Option<AprModel>,
    /// Raw bytes for zero-copy access
    raw_bytes: Vec<u8>,
    /// Mock DOM for testing
    dom: MockDom,
    /// Error state
    error: Option<String>,
}

impl Default for AprExplorerDriver {
    fn default() -> Self {
        Self::new()
    }
}

impl AprExplorerDriver {
    /// Creates a new APR explorer driver
    #[must_use]
    pub fn new() -> Self {
        Self {
            model: None,
            raw_bytes: Vec::new(),
            dom: MockDom::apr_explorer(),
            error: None,
        }
    }

    /// Returns a reference to the DOM
    #[must_use]
    pub fn dom(&self) -> &MockDom {
        &self.dom
    }

    /// Returns a mutable reference to the DOM
    pub fn dom_mut(&mut self) -> &mut MockDom {
        &mut self.dom
    }

    /// Returns the currently loaded model
    #[must_use]
    pub fn model(&self) -> Option<&AprModel> {
        self.model.as_ref()
    }

    /// Returns the current error message
    #[must_use]
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    /// Simulates dropping a file on the drop zone
    pub fn drop_file(&mut self, file_name: &str, bytes: &[u8]) {
        self.dom
            .dispatch_event(DomEvent::file_drop("drop-zone", file_name, bytes.len()));

        // Store raw bytes
        self.raw_bytes = bytes.to_vec();

        // Update status
        self.dom.set_element_text("status", "Parsing APR file...");

        // Parse the file
        match parse_apr(bytes) {
            Ok(model) => {
                self.error = None;
                self.sync_dom_success(&model);
                self.model = Some(model);
            }
            Err(e) => {
                self.error = Some(e.to_string());
                self.model = None;
                self.sync_dom_error(&e);
            }
        }
    }

    /// Load APR bytes directly (for testing)
    pub fn load_bytes(&mut self, bytes: &[u8]) -> Result<&AprModel, AprError> {
        self.raw_bytes = bytes.to_vec();

        match parse_apr(bytes) {
            Ok(model) => {
                self.error = None;
                self.sync_dom_success(&model);
                self.model = Some(model);
                Ok(self.model.as_ref().unwrap())
            }
            Err(e) => {
                self.error = Some(e.to_string());
                self.model = None;
                self.sync_dom_error(&e);
                Err(e)
            }
        }
    }

    /// Clear the current model
    pub fn clear(&mut self) {
        self.model = None;
        self.raw_bytes.clear();
        self.error = None;

        // Reset DOM
        self.dom.set_element_text("status", "Ready");
        self.dom.set_element_text("load-time", "-");
        self.dom.set_element_text("file-size", "-");
        self.dom.set_element_text("tensor-count", "-");
        self.dom.set_element_text("alignment", "-");
        self.dom.set_element_text("header-info", "");
        self.dom.set_element_text("metadata-info", "");
        self.dom.clear_children("tensor-list");
    }

    /// Peek raw bytes at offset
    #[must_use]
    pub fn peek_bytes(&self, offset: usize, length: usize) -> Option<&[u8]> {
        if offset + length <= self.raw_bytes.len() {
            Some(&self.raw_bytes[offset..offset + length])
        } else {
            None
        }
    }

    /// Get status text from DOM
    #[must_use]
    pub fn status_text(&self) -> Option<&str> {
        self.dom.get_element_text("status")
    }

    /// Get tensor count from DOM
    #[must_use]
    pub fn tensor_count_text(&self) -> Option<&str> {
        self.dom.get_element_text("tensor-count")
    }

    /// Synchronize DOM state on successful parse
    fn sync_dom_success(&mut self, model: &AprModel) {
        // Update status
        self.dom.set_element_text(
            "status",
            &format!("Loaded: {} tensors", model.tensors.len()),
        );

        // Update metrics
        self.dom
            .set_element_text("load-time", &format!("{:.2}", model.parse_time_ms));
        self.dom.set_element_text(
            "file-size",
            &format!("{:.1}", model.file_size as f64 / 1024.0),
        );
        self.dom
            .set_element_text("tensor-count", &model.tensors.len().to_string());
        self.dom.set_element_text("alignment", "64B");

        // Update header info
        let compression_info = if model.header.compressed {
            format!(" ({})", model.header.compression.name())
        } else {
            String::new()
        };
        let header_html = format!(
            "Format: {}{} | Tensors: {} | Metadata: {} bytes | Index: {} bytes",
            model.header.magic,
            compression_info,
            model.header.n_tensors,
            model.header.metadata_size,
            model.header.index_size
        );
        self.dom.set_element_text("header-info", &header_html);

        // Update metadata
        let metadata_json =
            serde_json::to_string_pretty(&model.metadata).unwrap_or_else(|_| "{}".to_string());
        self.dom.set_element_text("metadata-info", &metadata_json);

        // Update tensor list
        self.dom.clear_children("tensor-list");
        for (i, tensor) in model.tensors.iter().enumerate() {
            let tensor_elem = DomElement::new("div")
                .with_id(&format!("tensor-{}", i))
                .with_class("tensor-item")
                .with_text(&format!(
                    "{}: {} [{:?}] @ {} ({})",
                    tensor.name,
                    tensor.dtype,
                    tensor.shape,
                    tensor.offset,
                    tensor.alignment_description()
                ));
            self.dom.append_child("tensor-list", tensor_elem);
        }
    }

    /// Synchronize DOM state on error
    fn sync_dom_error(&mut self, error: &AprError) {
        self.dom
            .set_element_text("status", &format!("Error: {}", error));
        self.dom.set_element_text("load-time", "-");
        self.dom.set_element_text("file-size", "-");
        self.dom.set_element_text("tensor-count", "-");
        self.dom.set_element_text("alignment", "-");
        self.dom.set_element_text("header-info", "");
        self.dom.set_element_text("metadata-info", "");
        self.dom.clear_children("tensor-list");
    }
}

// ===== Test Specifications =====
// These functions define the specification that both native and WASM tests must pass.

/// Create test APR1 bytes with standard test data
pub fn create_test_apr_bytes() -> Vec<u8> {
    let mut metadata: ModelMetadata = BTreeMap::new();
    metadata.insert("model_type".to_string(), json!("test"));
    metadata.insert("architecture".to_string(), json!("mlp"));

    let tensors = vec![TensorInfo {
        name: "w0".to_string(),
        dtype: "F32".to_string(),
        shape: vec![64, 64],
        offset: 0,
        size: 16384,
    }];

    let tensor_data = vec![0u8; 16384];

    create_test_apr1(&metadata, &tensors, &tensor_data)
}

/// Verify basic APR loading
pub fn verify_apr_loading(driver: &mut AprExplorerDriver) {
    let bytes = create_test_apr_bytes();
    let result = driver.load_bytes(&bytes);
    assert!(result.is_ok(), "Should load valid APR");

    let model = result.unwrap();
    assert_eq!(model.header.magic, "APR1");
    assert!(!model.header.compressed);
}

/// Verify tensor information
pub fn verify_tensor_info(driver: &mut AprExplorerDriver) {
    let bytes = create_test_apr_bytes();
    driver.load_bytes(&bytes).unwrap();

    let model = driver.model().unwrap();
    assert!(!model.tensors.is_empty());

    let tensor = &model.tensors[0];
    assert_eq!(tensor.name, "w0");
    assert_eq!(tensor.dtype, "F32");
    assert!(tensor.is_aligned_64() || tensor.is_aligned_32());
}

/// Verify DOM synchronization
pub fn verify_dom_sync(driver: &mut AprExplorerDriver) {
    let bytes = create_test_apr_bytes();
    driver.load_bytes(&bytes).unwrap();

    // Status should show loaded
    let status = driver.status_text().unwrap();
    assert!(status.contains("Loaded"), "Status should indicate loaded");

    // Tensor count should be updated
    let count = driver.tensor_count_text().unwrap();
    assert_eq!(count, "1", "Tensor count should be 1");
}

/// Verify error handling
pub fn verify_error_handling(driver: &mut AprExplorerDriver) {
    // Invalid magic
    let invalid_bytes = vec![0u8; 64];
    let result = driver.load_bytes(&invalid_bytes);
    assert!(result.is_err());

    // Status should show error
    let status = driver.status_text().unwrap();
    assert!(status.contains("Error"), "Status should indicate error");
}

/// Verify clear operation
pub fn verify_clear(driver: &mut AprExplorerDriver) {
    let bytes = create_test_apr_bytes();
    driver.load_bytes(&bytes).unwrap();

    driver.clear();

    assert!(driver.model().is_none());
    assert_eq!(driver.status_text(), Some("Ready"));
    assert_eq!(driver.tensor_count_text(), Some("-"));
}

/// Run the full specification
pub fn run_full_specification(driver: &mut AprExplorerDriver) {
    verify_apr_loading(driver);
    driver.clear();

    verify_tensor_info(driver);
    driver.clear();

    verify_dom_sync(driver);
    driver.clear();

    verify_error_handling(driver);
    driver.clear();

    verify_clear(driver);
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Constructor tests =====

    #[test]
    fn test_driver_new() {
        let driver = AprExplorerDriver::new();
        assert!(driver.model().is_none());
        assert!(driver.error().is_none());
    }

    #[test]
    fn test_driver_default() {
        let driver = AprExplorerDriver::default();
        assert!(driver.model().is_none());
    }

    // ===== Load tests =====

    #[test]
    fn test_load_bytes_success() {
        let mut driver = AprExplorerDriver::new();
        let bytes = create_test_apr_bytes();
        let result = driver.load_bytes(&bytes);
        assert!(result.is_ok());
        assert!(driver.model().is_some());
    }

    #[test]
    fn test_load_bytes_error() {
        let mut driver = AprExplorerDriver::new();
        let result = driver.load_bytes(&[0u8; 10]);
        assert!(result.is_err());
        assert!(driver.model().is_none());
    }

    #[test]
    fn test_drop_file_success() {
        let mut driver = AprExplorerDriver::new();
        let bytes = create_test_apr_bytes();
        driver.drop_file("test.apr", &bytes);

        assert!(driver.model().is_some());
        assert!(driver.error().is_none());

        // Check event was recorded
        let events = driver.dom().event_history();
        assert!(events
            .iter()
            .any(|e| matches!(e, DomEvent::FileDrop { .. })));
    }

    #[test]
    fn test_drop_file_error() {
        let mut driver = AprExplorerDriver::new();
        driver.drop_file("bad.apr", &[0u8; 10]);

        assert!(driver.model().is_none());
        assert!(driver.error().is_some());
    }

    // ===== DOM access tests =====

    #[test]
    fn test_dom_access() {
        let driver = AprExplorerDriver::new();
        assert!(driver.dom().get_element("status").is_some());
    }

    #[test]
    fn test_dom_mut_access() {
        let mut driver = AprExplorerDriver::new();
        driver.dom_mut().set_element_text("status", "Testing");
        assert_eq!(driver.status_text(), Some("Testing"));
    }

    // ===== Peek bytes tests =====

    #[test]
    fn test_peek_bytes() {
        let mut driver = AprExplorerDriver::new();
        let bytes = create_test_apr_bytes();
        driver.load_bytes(&bytes).unwrap();

        let header = driver.peek_bytes(0, 4).unwrap();
        assert_eq!(header, b"APR1");
    }

    #[test]
    fn test_peek_bytes_out_of_bounds() {
        let mut driver = AprExplorerDriver::new();
        let bytes = create_test_apr_bytes();
        driver.load_bytes(&bytes).unwrap();

        assert!(driver.peek_bytes(bytes.len() + 100, 10).is_none());
    }

    // ===== Clear tests =====

    #[test]
    fn test_clear() {
        let mut driver = AprExplorerDriver::new();
        let bytes = create_test_apr_bytes();
        driver.load_bytes(&bytes).unwrap();

        driver.clear();

        assert!(driver.model().is_none());
        assert_eq!(driver.status_text(), Some("Ready"));
    }

    // ===== Unified specification tests =====

    #[test]
    fn test_spec_apr_loading() {
        let mut driver = AprExplorerDriver::new();
        verify_apr_loading(&mut driver);
    }

    #[test]
    fn test_spec_tensor_info() {
        let mut driver = AprExplorerDriver::new();
        verify_tensor_info(&mut driver);
    }

    #[test]
    fn test_spec_dom_sync() {
        let mut driver = AprExplorerDriver::new();
        verify_dom_sync(&mut driver);
    }

    #[test]
    fn test_spec_error_handling() {
        let mut driver = AprExplorerDriver::new();
        verify_error_handling(&mut driver);
    }

    #[test]
    fn test_spec_clear() {
        let mut driver = AprExplorerDriver::new();
        verify_clear(&mut driver);
    }

    #[test]
    fn test_full_specification() {
        let mut driver = AprExplorerDriver::new();
        run_full_specification(&mut driver);
    }

    // ===== Model-specific tests =====

    #[test]
    fn test_model_metadata_access() {
        let mut driver = AprExplorerDriver::new();
        let bytes = create_test_apr_bytes();
        driver.load_bytes(&bytes).unwrap();

        let model = driver.model().unwrap();
        assert_eq!(model.metadata.get("model_type"), Some(&json!("test")));
        assert_eq!(model.metadata.get("architecture"), Some(&json!("mlp")));
    }

    #[test]
    fn test_model_header_info() {
        let mut driver = AprExplorerDriver::new();
        let bytes = create_test_apr_bytes();
        driver.load_bytes(&bytes).unwrap();

        let model = driver.model().unwrap();
        assert_eq!(model.header.magic, "APR1");
        assert_eq!(model.header.n_tensors, 1);
        assert!(!model.header.compressed);
    }

    #[test]
    fn test_multiple_tensors() {
        let mut metadata: ModelMetadata = BTreeMap::new();
        metadata.insert("model_type".to_string(), json!("multi-tensor"));

        let tensors = vec![
            TensorInfo {
                name: "layer.0.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![64, 32],
                offset: 0,
                size: 8192,
            },
            TensorInfo {
                name: "layer.0.bias".to_string(),
                dtype: "F32".to_string(),
                shape: vec![64],
                offset: 8192,
                size: 256,
            },
            TensorInfo {
                name: "layer.1.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![10, 64],
                offset: 8448,
                size: 2560,
            },
        ];

        let total_size: usize = tensors.iter().map(|t| t.size).sum();
        let tensor_data = vec![0u8; total_size];
        let bytes = create_test_apr1(&metadata, &tensors, &tensor_data);

        let mut driver = AprExplorerDriver::new();
        driver.load_bytes(&bytes).unwrap();

        let model = driver.model().unwrap();
        assert_eq!(model.tensors.len(), 3);
        assert_eq!(model.header.n_tensors, 3);

        // Verify DOM shows correct count
        assert_eq!(driver.tensor_count_text(), Some("3"));
    }
}
