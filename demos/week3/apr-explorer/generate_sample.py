#!/usr/bin/env python3
"""Generate sample APR1 files for testing."""

import struct
import json
import os

def crc32_ieee(data: bytes) -> int:
    """CRC32 with IEEE polynomial."""
    crc = 0xFFFFFFFF
    table = []
    for i in range(256):
        crc_val = i
        for _ in range(8):
            if crc_val & 1:
                crc_val = 0xEDB88320 ^ (crc_val >> 1)
            else:
                crc_val >>= 1
        table.append(crc_val)

    for byte in data:
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF

def create_apr1(metadata: dict, tensors: list, tensor_data: bytes) -> bytes:
    """Create APR1 format file."""
    output = bytearray()

    # 1. Magic: "APR1"
    output.extend(b"APR1")

    # 2. Metadata
    metadata_json = json.dumps(metadata, separators=(',', ':')).encode('utf-8')
    output.extend(struct.pack('<I', len(metadata_json)))
    output.extend(metadata_json)

    # 3. Tensor count
    output.extend(struct.pack('<I', len(tensors)))

    # 4. Tensor index
    index_json = json.dumps(tensors, separators=(',', ':')).encode('utf-8')
    output.extend(struct.pack('<I', len(index_json)))
    output.extend(index_json)

    # 5. Tensor data
    output.extend(tensor_data)

    # 6. CRC32
    crc = crc32_ieee(bytes(output))
    output.extend(struct.pack('<I', crc))

    return bytes(output)

def main():
    # Sample 1: Simple MLP model
    metadata1 = {
        "model_type": "mlp",
        "architecture": "feedforward",
        "hidden_size": 128,
        "num_layers": 2,
        "created_by": "apr-explorer-demo"
    }

    tensors1 = [
        {"name": "layer.0.weight", "dtype": "F32", "shape": [128, 64], "offset": 0, "size": 32768},
        {"name": "layer.0.bias", "dtype": "F32", "shape": [128], "offset": 32768, "size": 512},
        {"name": "layer.1.weight", "dtype": "F32", "shape": [10, 128], "offset": 33280, "size": 5120},
        {"name": "layer.1.bias", "dtype": "F32", "shape": [10], "offset": 38400, "size": 40},
    ]

    # Create random tensor data
    total_size1 = sum(t["size"] for t in tensors1)
    tensor_data1 = os.urandom(total_size1)

    apr1_bytes = create_apr1(metadata1, tensors1, tensor_data1)

    with open("www/sample-mlp.apr", "wb") as f:
        f.write(apr1_bytes)

    print(f"Created www/sample-mlp.apr ({len(apr1_bytes)} bytes)")
    print(f"  Metadata: {len(json.dumps(metadata1))} bytes")
    print(f"  Tensors: {len(tensors1)}")
    print(f"  Data: {total_size1} bytes")

    # Sample 2: Transformer-like model
    metadata2 = {
        "model_type": "transformer",
        "architecture": "encoder-only",
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_layers": 2,
        "vocab_size": 1000,
        "max_position_embeddings": 512
    }

    tensors2 = [
        {"name": "embed.weight", "dtype": "F32", "shape": [1000, 256], "offset": 0, "size": 1024000},
        {"name": "layers.0.attn.q.weight", "dtype": "F32", "shape": [256, 256], "offset": 1024000, "size": 262144},
        {"name": "layers.0.attn.k.weight", "dtype": "F32", "shape": [256, 256], "offset": 1286144, "size": 262144},
        {"name": "layers.0.attn.v.weight", "dtype": "F32", "shape": [256, 256], "offset": 1548288, "size": 262144},
        {"name": "layers.0.attn.o.weight", "dtype": "F32", "shape": [256, 256], "offset": 1810432, "size": 262144},
        {"name": "layers.0.ffn.up.weight", "dtype": "F32", "shape": [1024, 256], "offset": 2072576, "size": 1048576},
        {"name": "layers.0.ffn.down.weight", "dtype": "F32", "shape": [256, 1024], "offset": 3121152, "size": 1048576},
    ]

    total_size2 = sum(t["size"] for t in tensors2)
    tensor_data2 = os.urandom(total_size2)

    apr2_bytes = create_apr1(metadata2, tensors2, tensor_data2)

    with open("www/sample-transformer.apr", "wb") as f:
        f.write(apr2_bytes)

    print(f"\nCreated www/sample-transformer.apr ({len(apr2_bytes)} bytes)")
    print(f"  Metadata: {len(json.dumps(metadata2))} bytes")
    print(f"  Tensors: {len(tensors2)}")
    print(f"  Data: {total_size2} bytes")

if __name__ == "__main__":
    main()
