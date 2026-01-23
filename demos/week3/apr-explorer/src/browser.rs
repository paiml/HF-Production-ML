//! Browser WASM bindings for APR Explorer
//!
//! This module provides the actual browser integration using wasm-bindgen.
//! All business logic is in the driver module - this is just the binding layer.
//!
//! Probar: Direct observation - Real browser behavior

use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::apr::parse_apr;
use crate::driver::AprExplorerDriver;

/// Browser APR Explorer - the main WASM entry point
#[derive(Debug)]
#[wasm_bindgen]
pub struct BrowserExplorer {
    driver: AprExplorerDriver,
}

#[wasm_bindgen]
impl BrowserExplorer {
    /// Create a new browser explorer
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();

        Self {
            driver: AprExplorerDriver::new(),
        }
    }

    /// Load APR bytes and return model info as JSON
    pub fn load(&mut self, bytes: &[u8]) -> Result<String, JsValue> {
        let start = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);

        match parse_apr(bytes) {
            Ok(mut model) => {
                let end = web_sys::window()
                    .and_then(|w| w.performance())
                    .map(|p| p.now())
                    .unwrap_or(0.0);

                model.parse_time_ms = end - start;

                // Store in driver
                let _ = self.driver.load_bytes(bytes);

                serde_json::to_string_pretty(&model).map_err(|e| JsValue::from_str(&e.to_string()))
            }
            Err(e) => Err(JsValue::from_str(&e.to_string())),
        }
    }

    /// Get summary as JSON
    pub fn summary(&self) -> Result<String, JsValue> {
        match self.driver.model() {
            Some(model) => {
                let summary = serde_json::json!({
                    "magic": model.header.magic,
                    "compressed": model.header.compressed,
                    "compression": model.header.compression.name(),
                    "tensor_count": model.tensors.len(),
                    "file_size_bytes": model.file_size,
                    "parse_time_ms": model.parse_time_ms,
                    "metadata_size": model.header.metadata_size,
                    "index_size": model.header.index_size,
                });
                serde_json::to_string_pretty(&summary)
                    .map_err(|e| JsValue::from_str(&e.to_string()))
            }
            None => Err(JsValue::from_str("No model loaded")),
        }
    }

    /// Get tensor list as JSON
    pub fn tensors(&self) -> Result<String, JsValue> {
        match self.driver.model() {
            Some(model) => serde_json::to_string_pretty(&model.tensors)
                .map_err(|e| JsValue::from_str(&e.to_string())),
            None => Err(JsValue::from_str("No model loaded")),
        }
    }

    /// Get metadata as JSON
    pub fn metadata(&self) -> Result<String, JsValue> {
        match self.driver.model() {
            Some(model) => serde_json::to_string_pretty(&model.metadata)
                .map_err(|e| JsValue::from_str(&e.to_string())),
            None => Err(JsValue::from_str("No model loaded")),
        }
    }

    /// Check tensor alignment
    pub fn check_alignment(&self, tensor_name: &str) -> Result<String, JsValue> {
        match self.driver.model() {
            Some(model) => {
                let tensor = model
                    .tensors
                    .iter()
                    .find(|t| t.name == tensor_name)
                    .ok_or_else(|| JsValue::from_str("Tensor not found"))?;

                let result = serde_json::json!({
                    "tensor": tensor_name,
                    "offset": tensor.offset,
                    "alignment": tensor.alignment_description(),
                    "is_simd_ready": tensor.is_aligned_64(),
                });
                Ok(result.to_string())
            }
            None => Err(JsValue::from_str("No model loaded")),
        }
    }

    /// Clear loaded model
    pub fn clear(&mut self) {
        self.driver.clear();
    }
}

impl Default for BrowserExplorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console::log_1(&"APR Explorer WASM initialized".into());
}
