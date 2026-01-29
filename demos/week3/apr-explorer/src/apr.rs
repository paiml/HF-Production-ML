//! APR Format Parser - Pure Rust, No Browser Dependencies
//!
//! Implements the real APR format from aprender:
//! - APR1: Uncompressed JSON format
//! - APR\0: Compressed format (LZ4/ZSTD)
//!
//! Format (APR1 - uncompressed):
//! ```text
//! [4-byte magic: "APR1"]
//! [4-byte metadata_len: u32 LE]
//! [JSON metadata: arbitrary key-value pairs]
//! [4-byte n_tensors: u32 LE]
//! [4-byte index_len: u32 LE]
//! [JSON tensor index: array of tensor descriptors]
//! [tensor data: raw bytes]
//! [4-byte CRC32: checksum of all preceding bytes]
//! ```
//!
//! Format (APR\0 - compressed):
//! ```text
//! [4-byte magic: "APR\0"]
//! [1-byte compression: 0=None, 1=LZ4, 2=ZSTD]
//! [4-byte uncompressed_len: u32 LE]
//! [compressed APR1 payload]
//! ```
//!
//! Probar: Direct observation - Parse real APR bytes

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Magic bytes for APR1 (uncompressed) format
pub const APR_MAGIC_UNCOMPRESSED: [u8; 4] = [b'A', b'P', b'R', b'1'];

/// Magic bytes for APR2 (compressed) format
pub const APR_MAGIC_COMPRESSED: [u8; 4] = [b'A', b'P', b'R', 0];

/// Compression type for APR files
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Compression {
    #[default]
    None = 0,
    Lz4 = 1,
    Zstd = 2,
}

impl Compression {
    /// Parse compression type from byte
    #[must_use]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            2 => Some(Self::Zstd),
            _ => None,
        }
    }

    /// Get display name
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Lz4 => "LZ4",
            Self::Zstd => "ZSTD",
        }
    }
}

/// APR file header information (for display)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AprHeader {
    /// Magic bytes as string ("APR1" or "APR\0")
    pub magic: String,
    /// Whether file is compressed
    pub compressed: bool,
    /// Compression algorithm if compressed
    pub compression: Compression,
    /// Metadata size in bytes
    pub metadata_size: u32,
    /// Number of tensors
    pub n_tensors: u32,
    /// Tensor index size in bytes
    pub index_size: u32,
}

/// Tensor descriptor from the index
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor name (e.g., "model.layers.0.weight")
    pub name: String,
    /// Data type (e.g., "F32", "I8")
    pub dtype: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Byte offset in data section
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
}

impl TensorInfo {
    /// Check if tensor data is 64-byte aligned (SIMD-ready)
    #[must_use]
    pub fn is_aligned_64(&self) -> bool {
        self.offset.is_multiple_of(64)
    }

    /// Check if tensor data is 32-byte aligned
    #[must_use]
    pub fn is_aligned_32(&self) -> bool {
        self.offset.is_multiple_of(32)
    }

    /// Get alignment description
    #[must_use]
    pub fn alignment_description(&self) -> &'static str {
        if self.offset.is_multiple_of(64) {
            "64-byte (AVX-512/SIMD128)"
        } else if self.offset.is_multiple_of(32) {
            "32-byte (AVX2)"
        } else if self.offset.is_multiple_of(16) {
            "16-byte (SSE)"
        } else {
            "Unaligned"
        }
    }

    /// Calculate total elements
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Model metadata - arbitrary JSON key-value pairs
pub type ModelMetadata = BTreeMap<String, serde_json::Value>;

/// Parsed APR model - all information needed for display
#[derive(Debug, Clone, Serialize)]
pub struct AprModel {
    /// File header info
    pub header: AprHeader,
    /// Model metadata (arbitrary JSON)
    pub metadata: ModelMetadata,
    /// Tensor descriptors
    pub tensors: Vec<TensorInfo>,
    /// Total file size in bytes
    pub file_size: usize,
    /// Time to parse in milliseconds
    pub parse_time_ms: f64,
    /// Offset to tensor data section
    pub tensor_data_offset: usize,
}

/// Error type for APR parsing
#[derive(Debug, Clone, PartialEq)]
pub enum AprError {
    /// File too small
    FileTooSmall { size: usize, required: usize },
    /// Invalid magic bytes
    InvalidMagic { found: String },
    /// Invalid compression type
    InvalidCompression { byte: u8 },
    /// Metadata parse error
    MetadataParseError { message: String },
    /// Tensor index parse error
    TensorIndexParseError { message: String },
    /// File truncated
    Truncated {
        section: String,
        needed: usize,
        available: usize,
    },
    /// Compression not supported (feature not enabled)
    CompressionNotSupported { compression: String },
    /// Decompression error
    DecompressionError { message: String },
}

impl std::fmt::Display for AprError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileTooSmall { size, required } => {
                write!(
                    f,
                    "File too small: {} bytes, need at least {}",
                    size, required
                )
            }
            Self::InvalidMagic { found } => {
                write!(
                    f,
                    "Invalid magic: expected 'APR1' or 'APR\\0', found '{}'",
                    found
                )
            }
            Self::InvalidCompression { byte } => {
                write!(f, "Invalid compression type: {}", byte)
            }
            Self::MetadataParseError { message } => {
                write!(f, "Metadata parse error: {}", message)
            }
            Self::TensorIndexParseError { message } => {
                write!(f, "Tensor index parse error: {}", message)
            }
            Self::Truncated {
                section,
                needed,
                available,
            } => {
                write!(
                    f,
                    "File truncated at {}: need {} bytes, only {} available",
                    section, needed, available
                )
            }
            Self::CompressionNotSupported { compression } => {
                write!(
                    f,
                    "Compression '{}' not supported in this build",
                    compression
                )
            }
            Self::DecompressionError { message } => {
                write!(f, "Decompression error: {}", message)
            }
        }
    }
}

impl std::error::Error for AprError {}

/// Parse APR file from raw bytes
///
/// Automatically detects APR1 (uncompressed) or APR\0 (compressed) format.
pub fn parse_apr(bytes: &[u8]) -> Result<AprModel, AprError> {
    let file_size = bytes.len();

    // Minimum size: magic (4) + metadata_len (4)
    if file_size < 8 {
        return Err(AprError::FileTooSmall {
            size: file_size,
            required: 8,
        });
    }

    let magic = &bytes[0..4];

    // Check for compressed format first
    if magic == APR_MAGIC_COMPRESSED {
        return parse_apr_compressed(bytes);
    }

    // Check for uncompressed format
    if magic == APR_MAGIC_UNCOMPRESSED {
        return parse_apr_uncompressed(bytes);
    }

    Err(AprError::InvalidMagic {
        found: String::from_utf8_lossy(magic).to_string(),
    })
}

/// Parse uncompressed APR1 format
fn parse_apr_uncompressed(bytes: &[u8]) -> Result<AprModel, AprError> {
    let file_size = bytes.len();
    let mut offset = 4; // Skip magic

    // Read metadata length
    if offset + 4 > file_size {
        return Err(AprError::Truncated {
            section: "metadata_len".to_string(),
            needed: 4,
            available: file_size - offset,
        });
    }
    let metadata_len = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]) as usize;
    offset += 4;

    // Read metadata JSON
    if offset + metadata_len > file_size {
        return Err(AprError::Truncated {
            section: "metadata".to_string(),
            needed: metadata_len,
            available: file_size - offset,
        });
    }
    let metadata: ModelMetadata = if metadata_len > 0 {
        serde_json::from_slice(&bytes[offset..offset + metadata_len]).map_err(|e| {
            AprError::MetadataParseError {
                message: e.to_string(),
            }
        })?
    } else {
        BTreeMap::new()
    };
    offset += metadata_len;

    // Read tensor count
    if offset + 4 > file_size {
        return Err(AprError::Truncated {
            section: "n_tensors".to_string(),
            needed: 4,
            available: file_size - offset,
        });
    }
    let n_tensors = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    offset += 4;

    // Read index length
    if offset + 4 > file_size {
        return Err(AprError::Truncated {
            section: "index_len".to_string(),
            needed: 4,
            available: file_size - offset,
        });
    }
    let index_len = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]) as usize;
    offset += 4;

    // Read tensor index JSON
    if offset + index_len > file_size {
        return Err(AprError::Truncated {
            section: "tensor_index".to_string(),
            needed: index_len,
            available: file_size - offset,
        });
    }
    let tensors: Vec<TensorInfo> = if n_tensors > 0 && index_len > 0 {
        serde_json::from_slice(&bytes[offset..offset + index_len]).map_err(|e| {
            AprError::TensorIndexParseError {
                message: e.to_string(),
            }
        })?
    } else {
        Vec::new()
    };
    offset += index_len;

    let tensor_data_offset = offset;

    let header = AprHeader {
        magic: "APR1".to_string(),
        compressed: false,
        compression: Compression::None,
        metadata_size: metadata_len as u32,
        n_tensors,
        index_size: index_len as u32,
    };

    Ok(AprModel {
        header,
        metadata,
        tensors,
        file_size,
        parse_time_ms: 0.0,
        tensor_data_offset,
    })
}

/// Parse compressed APR\0 format
fn parse_apr_compressed(bytes: &[u8]) -> Result<AprModel, AprError> {
    let file_size = bytes.len();

    // Minimum: magic (4) + compression (1) + uncompressed_len (4)
    if file_size < 9 {
        return Err(AprError::Truncated {
            section: "compressed_header".to_string(),
            needed: 9,
            available: file_size,
        });
    }

    let compression_byte = bytes[4];
    let compression =
        Compression::from_byte(compression_byte).ok_or(AprError::InvalidCompression {
            byte: compression_byte,
        })?;

    let uncompressed_len = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]) as usize;
    let compressed_payload = &bytes[9..];

    // Decompress based on algorithm
    let decompressed = match compression {
        Compression::None => compressed_payload.to_vec(),
        Compression::Lz4 => {
            return Err(AprError::CompressionNotSupported {
                compression: "LZ4".to_string(),
            });
        }
        Compression::Zstd => {
            // Use ruzstd for pure-Rust ZSTD decompression (works in WASM)
            let mut decoder = ruzstd::StreamingDecoder::new(compressed_payload).map_err(|e| {
                AprError::DecompressionError {
                    message: format!("ZSTD init failed: {}", e),
                }
            })?;

            let mut decompressed = Vec::with_capacity(uncompressed_len);
            std::io::Read::read_to_end(&mut decoder, &mut decompressed).map_err(|e| {
                AprError::DecompressionError {
                    message: format!("ZSTD decode failed: {}", e),
                }
            })?;
            decompressed
        }
    };

    // Parse the decompressed APR1 data
    // Note: For Compression::None, the payload should be valid APR1 data
    // starting with APR1 magic, but we need to handle it properly
    if decompressed.len() >= 4 && decompressed[0..4] == APR_MAGIC_UNCOMPRESSED {
        let mut model = parse_apr_uncompressed(&decompressed)?;
        model.header.compressed = true;
        model.header.compression = compression;
        model.file_size = file_size; // Use original file size
        Ok(model)
    } else {
        // The payload might be raw APR1 content without the magic
        // This shouldn't happen with properly formatted files
        Err(AprError::InvalidMagic {
            found: String::from_utf8_lossy(&decompressed[0..4.min(decompressed.len())]).to_string(),
        })
    }
}

/// Create APR1 bytes for testing
///
/// This creates a valid APR1 file with the given metadata and tensors.
pub fn create_test_apr1(
    metadata: &ModelMetadata,
    tensors: &[TensorInfo],
    tensor_data: &[u8],
) -> Vec<u8> {
    let mut output = Vec::new();

    // 1. Magic
    output.extend_from_slice(&APR_MAGIC_UNCOMPRESSED);

    // 2. Metadata
    let metadata_json = serde_json::to_string(metadata).unwrap_or_else(|_| "{}".to_string());
    let metadata_bytes = metadata_json.as_bytes();
    output.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(metadata_bytes);

    // 3. Tensor count
    output.extend_from_slice(&(tensors.len() as u32).to_le_bytes());

    // 4. Tensor index
    let index_json = serde_json::to_string(tensors).unwrap_or_else(|_| "[]".to_string());
    let index_bytes = index_json.as_bytes();
    output.extend_from_slice(&(index_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(index_bytes);

    // 5. Tensor data
    output.extend_from_slice(tensor_data);

    // 6. CRC32
    let crc = crc32(&output);
    output.extend_from_slice(&crc.to_le_bytes());

    output
}

/// Simple CRC32 implementation (IEEE polynomial)
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = CRC32_TABLE[idx] ^ (crc >> 8);
    }
    !crc
}

/// CRC32 lookup table (IEEE polynomial)
const CRC32_TABLE: [u32; 256] = [
    0x0000_0000,
    0x7707_3096,
    0xEE0E_612C,
    0x9909_51BA,
    0x076D_C419,
    0x706A_F48F,
    0xE963_A535,
    0x9E64_95A3,
    0x0EDB_8832,
    0x79DC_B8A4,
    0xE0D5_E91E,
    0x97D2_D988,
    0x09B6_4C2B,
    0x7EB1_7CBD,
    0xE7B8_2D07,
    0x90BF_1D91,
    0x1DB7_1064,
    0x6AB0_20F2,
    0xF3B9_7148,
    0x84BE_41DE,
    0x1ADA_D47D,
    0x6DDD_E4EB,
    0xF4D4_B551,
    0x83D3_85C7,
    0x136C_9856,
    0x646B_A8C0,
    0xFD62_F97A,
    0x8A65_C9EC,
    0x1401_5C4F,
    0x6306_6CD9,
    0xFA0F_3D63,
    0x8D08_0DF5,
    0x3B6E_20C8,
    0x4C69_105E,
    0xD560_41E4,
    0xA267_7172,
    0x3C03_E4D1,
    0x4B04_D447,
    0xD20D_85FD,
    0xA50A_B56B,
    0x35B5_A8FA,
    0x42B2_986C,
    0xDBBB_C9D6,
    0xACBC_F940,
    0x32D8_6CE3,
    0x45DF_5C75,
    0xDCD6_0DCF,
    0xABD1_3D59,
    0x26D9_30AC,
    0x51DE_003A,
    0xC8D7_5180,
    0xBFD0_6116,
    0x21B4_F4B5,
    0x56B3_C423,
    0xCFBA_9599,
    0xB8BD_A50F,
    0x2802_B89E,
    0x5F05_8808,
    0xC60C_D9B2,
    0xB10B_E924,
    0x2F6F_7C87,
    0x5868_4C11,
    0xC161_1DAB,
    0xB666_2D3D,
    0x76DC_4190,
    0x01DB_7106,
    0x98D2_20BC,
    0xEFD5_102A,
    0x71B1_8589,
    0x06B6_B51F,
    0x9FBF_E4A5,
    0xE8B8_D433,
    0x7807_C9A2,
    0x0F00_F934,
    0x9609_A88E,
    0xE10E_9818,
    0x7F6A_0DBB,
    0x086D_3D2D,
    0x9164_6C97,
    0xE663_5C01,
    0x6B6B_51F4,
    0x1C6C_6162,
    0x8565_30D8,
    0xF262_004E,
    0x6C06_95ED,
    0x1B01_A57B,
    0x8208_F4C1,
    0xF50F_C457,
    0x65B0_D9C6,
    0x12B7_E950,
    0x8BBE_B8EA,
    0xFCB9_887C,
    0x62DD_1DDF,
    0x15DA_2D49,
    0x8CD3_7CF3,
    0xFBD4_4C65,
    0x4DB2_6158,
    0x3AB5_51CE,
    0xA3BC_0074,
    0xD4BB_30E2,
    0x4ADF_A541,
    0x3DD8_95D7,
    0xA4D1_C46D,
    0xD3D6_F4FB,
    0x4369_E96A,
    0x346E_D9FC,
    0xAD67_8846,
    0xDA60_B8D0,
    0x4404_2D73,
    0x3303_1DE5,
    0xAA0A_4C5F,
    0xDD0D_7CC9,
    0x5005_713C,
    0x2702_41AA,
    0xBE0B_1010,
    0xC90C_2086,
    0x5768_B525,
    0x206F_85B3,
    0xB966_D409,
    0xCE61_E49F,
    0x5EDE_F90E,
    0x29D9_C998,
    0xB0D0_9822,
    0xC7D7_A8B4,
    0x59B3_3D17,
    0x2EB4_0D81,
    0xB7BD_5C3B,
    0xC0BA_6CAD,
    0xEDB8_8320,
    0x9ABF_B3B6,
    0x03B6_E20C,
    0x74B1_D29A,
    0xEAD5_4739,
    0x9DD2_77AF,
    0x04DB_2615,
    0x73DC_1683,
    0xE363_0B12,
    0x9464_3B84,
    0x0D6D_6A3E,
    0x7A6A_5AA8,
    0xE40E_CF0B,
    0x9309_FF9D,
    0x0A00_AE27,
    0x7D07_9EB1,
    0xF00F_9344,
    0x8708_A3D2,
    0x1E01_F268,
    0x6906_C2FE,
    0xF762_575D,
    0x8065_67CB,
    0x196C_3671,
    0x6E6B_06E7,
    0xFED4_1B76,
    0x89D3_2BE0,
    0x10DA_7A5A,
    0x67DD_4ACC,
    0xF9B9_DF6F,
    0x8EBE_EFF9,
    0x17B7_BE43,
    0x60B0_8ED5,
    0xD6D6_A3E8,
    0xA1D1_937E,
    0x38D8_C2C4,
    0x4FDF_F252,
    0xD1BB_67F1,
    0xA6BC_5767,
    0x3FB5_06DD,
    0x48B2_364B,
    0xD80D_2BDA,
    0xAF0A_1B4C,
    0x3603_4AF6,
    0x4104_7A60,
    0xDF60_EFC3,
    0xA867_DF55,
    0x316E_8EEF,
    0x4669_BE79,
    0xCB61_B38C,
    0xBC66_831A,
    0x256F_D2A0,
    0x5268_E236,
    0xCC0C_7795,
    0xBB0B_4703,
    0x2202_16B9,
    0x5505_262F,
    0xC5BA_3BBE,
    0xB2BD_0B28,
    0x2BB4_5A92,
    0x5CB3_6A04,
    0xC2D7_FFA7,
    0xB5D0_CF31,
    0x2CD9_9E8B,
    0x5BDE_AE1D,
    0x9B64_C2B0,
    0xEC63_F226,
    0x756A_A39C,
    0x026D_930A,
    0x9C09_06A9,
    0xEB0E_363F,
    0x7207_6785,
    0x0500_5713,
    0x95BF_4A82,
    0xE2B8_7A14,
    0x7BB1_2BAE,
    0x0CB6_1B38,
    0x92D2_8E9B,
    0xE5D5_BE0D,
    0x7CDC_EFB7,
    0x0BDB_DF21,
    0x86D3_D2D4,
    0xF1D4_E242,
    0x68DD_B3F8,
    0x1FDA_836E,
    0x81BE_16CD,
    0xF6B9_265B,
    0x6FB0_77E1,
    0x18B7_4777,
    0x8808_5AE6,
    0xFF0F_6A70,
    0x6606_3BCA,
    0x1101_0B5C,
    0x8F65_9EFF,
    0xF862_AE69,
    0x616B_FFD3,
    0x166C_CF45,
    0xA00A_E278,
    0xD70D_D2EE,
    0x4E04_8354,
    0x3903_B3C2,
    0xA767_2661,
    0xD060_16F7,
    0x4969_474D,
    0x3E6E_77DB,
    0xAED1_6A4A,
    0xD9D6_5ADC,
    0x40DF_0B66,
    0x37D8_3BF0,
    0xA9BC_AE53,
    0xDEBB_9EC5,
    0x47B2_CF7F,
    0x30B5_FFE9,
    0xBDBD_F21C,
    0xCABA_C28A,
    0x53B3_9330,
    0x24B4_A3A6,
    0xBAD0_3605,
    0xCDD7_0693,
    0x54DE_5729,
    0x23D9_67BF,
    0xB366_7A2E,
    0xC461_4AB8,
    0x5D68_1B02,
    0x2A6F_2B94,
    0xB40B_BE37,
    0xC30C_8EA1,
    0x5A05_DF1B,
    0x2D02_EF8D,
];

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ===== Magic bytes tests =====

    #[test]
    fn test_apr_magic_uncompressed() {
        assert_eq!(APR_MAGIC_UNCOMPRESSED, [b'A', b'P', b'R', b'1']);
    }

    #[test]
    fn test_apr_magic_compressed() {
        assert_eq!(APR_MAGIC_COMPRESSED, [b'A', b'P', b'R', 0]);
    }

    #[test]
    fn test_magics_distinct() {
        assert_ne!(APR_MAGIC_UNCOMPRESSED, APR_MAGIC_COMPRESSED);
    }

    // ===== Compression tests =====

    #[test]
    fn test_compression_from_byte() {
        assert_eq!(Compression::from_byte(0), Some(Compression::None));
        assert_eq!(Compression::from_byte(1), Some(Compression::Lz4));
        assert_eq!(Compression::from_byte(2), Some(Compression::Zstd));
        assert_eq!(Compression::from_byte(255), None);
    }

    #[test]
    fn test_compression_name() {
        assert_eq!(Compression::None.name(), "None");
        assert_eq!(Compression::Lz4.name(), "LZ4");
        assert_eq!(Compression::Zstd.name(), "ZSTD");
    }

    // ===== TensorInfo tests =====

    #[test]
    fn test_tensor_alignment_64() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![64, 64],
            offset: 128,
            size: 16384,
        };
        assert!(tensor.is_aligned_64());
        assert!(tensor.is_aligned_32());
        assert_eq!(tensor.alignment_description(), "64-byte (AVX-512/SIMD128)");
    }

    #[test]
    fn test_tensor_alignment_32() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![64, 64],
            offset: 96,
            size: 16384,
        };
        assert!(!tensor.is_aligned_64());
        assert!(tensor.is_aligned_32());
        assert_eq!(tensor.alignment_description(), "32-byte (AVX2)");
    }

    #[test]
    fn test_tensor_alignment_16() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![64, 64],
            offset: 48,
            size: 16384,
        };
        assert!(!tensor.is_aligned_64());
        assert!(!tensor.is_aligned_32());
        assert_eq!(tensor.alignment_description(), "16-byte (SSE)");
    }

    #[test]
    fn test_tensor_unaligned() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![64, 64],
            offset: 7,
            size: 16384,
        };
        assert!(!tensor.is_aligned_64());
        assert!(!tensor.is_aligned_32());
        assert_eq!(tensor.alignment_description(), "Unaligned");
    }

    #[test]
    fn test_tensor_num_elements() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            dtype: "F32".to_string(),
            shape: vec![3, 4, 5],
            offset: 0,
            size: 240,
        };
        assert_eq!(tensor.num_elements(), 60);
    }

    #[test]
    fn test_tensor_num_elements_empty_shape() {
        let tensor = TensorInfo {
            name: "scalar".to_string(),
            dtype: "F32".to_string(),
            shape: vec![],
            offset: 0,
            size: 4,
        };
        assert_eq!(tensor.num_elements(), 1); // Product of empty = 1
    }

    // ===== CRC32 tests =====

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(&[]), 0x0000_0000);
    }

    #[test]
    fn test_crc32_hello() {
        // Known CRC32 for "hello" (IEEE polynomial)
        let crc = crc32(b"hello");
        assert_eq!(crc, 0x3610_A686);
    }

    // ===== create_test_apr1 tests =====

    #[test]
    fn test_create_test_apr1_empty() {
        let metadata = BTreeMap::new();
        let tensors = Vec::new();
        let data = Vec::new();

        let bytes = create_test_apr1(&metadata, &tensors, &data);

        // Should have magic
        assert_eq!(&bytes[0..4], &APR_MAGIC_UNCOMPRESSED);

        // Parse it back
        let model = parse_apr(&bytes).unwrap();
        assert_eq!(model.header.magic, "APR1");
        assert!(!model.header.compressed);
        assert!(model.metadata.is_empty());
        assert!(model.tensors.is_empty());
    }

    #[test]
    fn test_create_test_apr1_with_metadata() {
        let mut metadata = BTreeMap::new();
        metadata.insert("model_type".to_string(), json!("test"));
        metadata.insert("version".to_string(), json!(1));

        let tensors = Vec::new();
        let data = Vec::new();

        let bytes = create_test_apr1(&metadata, &tensors, &data);
        let model = parse_apr(&bytes).unwrap();

        assert_eq!(model.metadata.get("model_type"), Some(&json!("test")));
        assert_eq!(model.metadata.get("version"), Some(&json!(1)));
    }

    #[test]
    fn test_create_test_apr1_with_tensors() {
        let metadata = BTreeMap::new();
        let tensors = vec![
            TensorInfo {
                name: "weights".to_string(),
                dtype: "F32".to_string(),
                shape: vec![2, 3],
                offset: 0,
                size: 24,
            },
            TensorInfo {
                name: "bias".to_string(),
                dtype: "F32".to_string(),
                shape: vec![3],
                offset: 24,
                size: 12,
            },
        ];
        let data = vec![0u8; 36];

        let bytes = create_test_apr1(&metadata, &tensors, &data);
        let model = parse_apr(&bytes).unwrap();

        assert_eq!(model.tensors.len(), 2);
        assert_eq!(model.tensors[0].name, "weights");
        assert_eq!(model.tensors[0].shape, vec![2, 3]);
        assert_eq!(model.tensors[1].name, "bias");
    }

    #[test]
    fn test_create_test_apr1_roundtrip() {
        let mut metadata = BTreeMap::new();
        metadata.insert("model_type".to_string(), json!("whisper-tiny"));
        metadata.insert("n_vocab".to_string(), json!(51865));

        let tensors = vec![TensorInfo {
            name: "encoder.embed".to_string(),
            dtype: "F32".to_string(),
            shape: vec![384, 80],
            offset: 0,
            size: 122880,
        }];
        let data = vec![0u8; 122880];

        let bytes = create_test_apr1(&metadata, &tensors, &data);
        let model = parse_apr(&bytes).unwrap();

        assert_eq!(model.header.magic, "APR1");
        assert_eq!(model.header.n_tensors, 1);
        assert_eq!(
            model.metadata.get("model_type"),
            Some(&json!("whisper-tiny"))
        );
        assert_eq!(model.tensors[0].name, "encoder.embed");
        assert_eq!(model.tensors[0].shape, vec![384, 80]);
    }

    // ===== parse_apr error tests =====

    #[test]
    fn test_parse_file_too_small() {
        let bytes = vec![0u8; 4];
        let result = parse_apr(&bytes);
        assert!(matches!(result, Err(AprError::FileTooSmall { .. })));
    }

    #[test]
    fn test_parse_invalid_magic() {
        let bytes = vec![b'X', b'Y', b'Z', b'1', 0, 0, 0, 0];
        let result = parse_apr(&bytes);
        assert!(matches!(result, Err(AprError::InvalidMagic { .. })));
    }

    #[test]
    fn test_parse_truncated_metadata() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&APR_MAGIC_UNCOMPRESSED);
        bytes.extend_from_slice(&100u32.to_le_bytes()); // metadata_len = 100
                                                        // But file ends here

        let result = parse_apr(&bytes);
        assert!(
            matches!(result, Err(AprError::Truncated { section, .. }) if section == "metadata")
        );
    }

    #[test]
    fn test_parse_invalid_metadata_json() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&APR_MAGIC_UNCOMPRESSED);
        bytes.extend_from_slice(&10u32.to_le_bytes()); // metadata_len = 10
        bytes.extend_from_slice(b"not json!!"); // Invalid JSON
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_tensors = 0
        bytes.extend_from_slice(&0u32.to_le_bytes()); // index_len = 0
        bytes.extend_from_slice(&0u32.to_le_bytes()); // CRC32

        let result = parse_apr(&bytes);
        assert!(matches!(result, Err(AprError::MetadataParseError { .. })));
    }

    #[test]
    fn test_parse_compressed_not_supported() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&APR_MAGIC_COMPRESSED);
        bytes.push(1); // LZ4 compression
        bytes.extend_from_slice(&100u32.to_le_bytes()); // uncompressed_len
        bytes.extend_from_slice(&[0u8; 50]); // Dummy compressed payload

        let result = parse_apr(&bytes);
        assert!(matches!(
            result,
            Err(AprError::CompressionNotSupported { .. })
        ));
    }

    #[test]
    fn test_parse_invalid_compression_byte() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&APR_MAGIC_COMPRESSED);
        bytes.push(99); // Invalid compression byte
        bytes.extend_from_slice(&100u32.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 50]);

        let result = parse_apr(&bytes);
        assert!(matches!(
            result,
            Err(AprError::InvalidCompression { byte: 99 })
        ));
    }

    // ===== AprError display tests =====

    #[test]
    fn test_error_display_file_too_small() {
        let err = AprError::FileTooSmall {
            size: 4,
            required: 8,
        };
        assert!(err.to_string().contains("too small"));
    }

    #[test]
    fn test_error_display_invalid_magic() {
        let err = AprError::InvalidMagic {
            found: "XYZ1".to_string(),
        };
        assert!(err.to_string().contains("Invalid magic"));
        assert!(err.to_string().contains("XYZ1"));
    }

    #[test]
    fn test_error_display_truncated() {
        let err = AprError::Truncated {
            section: "metadata".to_string(),
            needed: 100,
            available: 50,
        };
        assert!(err.to_string().contains("truncated"));
        assert!(err.to_string().contains("metadata"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let err = AprError::InvalidMagic {
            found: "test".to_string(),
        };
        let _: &dyn std::error::Error = &err;
    }

    // ===== Full integration tests =====

    #[test]
    fn test_full_model_roundtrip() {
        let mut metadata = BTreeMap::new();
        metadata.insert("model_type".to_string(), json!("demo"));
        metadata.insert("architecture".to_string(), json!("mlp"));
        metadata.insert("hidden_size".to_string(), json!(128));

        let tensors = vec![
            TensorInfo {
                name: "layer.0.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![128, 64],
                offset: 0,
                size: 32768,
            },
            TensorInfo {
                name: "layer.0.bias".to_string(),
                dtype: "F32".to_string(),
                shape: vec![128],
                offset: 32768,
                size: 512,
            },
            TensorInfo {
                name: "layer.1.weight".to_string(),
                dtype: "F32".to_string(),
                shape: vec![10, 128],
                offset: 33280,
                size: 5120,
            },
        ];

        // Create dummy tensor data
        let total_size: usize = tensors.iter().map(|t| t.size).sum();
        let data = vec![0u8; total_size];

        let bytes = create_test_apr1(&metadata, &tensors, &data);
        let model = parse_apr(&bytes).unwrap();

        // Verify header
        assert_eq!(model.header.magic, "APR1");
        assert!(!model.header.compressed);
        assert_eq!(model.header.n_tensors, 3);

        // Verify metadata
        assert_eq!(model.metadata.get("model_type"), Some(&json!("demo")));
        assert_eq!(model.metadata.get("hidden_size"), Some(&json!(128)));

        // Verify tensors
        assert_eq!(model.tensors.len(), 3);
        assert_eq!(model.tensors[0].name, "layer.0.weight");
        assert_eq!(model.tensors[0].shape, vec![128, 64]);
        assert_eq!(model.tensors[1].name, "layer.0.bias");
        assert_eq!(model.tensors[2].name, "layer.1.weight");

        // Verify file size
        assert_eq!(model.file_size, bytes.len());
    }

    #[test]
    fn test_apr_header_serialization() {
        let header = AprHeader {
            magic: "APR1".to_string(),
            compressed: false,
            compression: Compression::None,
            metadata_size: 42,
            n_tensors: 5,
            index_size: 256,
        };

        let json = serde_json::to_string(&header).unwrap();
        let parsed: AprHeader = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.magic, "APR1");
        assert_eq!(parsed.n_tensors, 5);
    }
}
