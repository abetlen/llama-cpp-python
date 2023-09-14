import operator
import sys
from functools import reduce
from io import SEEK_CUR
from typing import Dict, Any, Tuple, Union, BinaryIO
import os
import struct
from enum import IntEnum
import pprint

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 2
GGUF_DEFAULT_ALIGNMENT = 32

KEY_GENERAL_ALIGNMENT = "general.alignment"


GGML_TYPE_MAP = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
}

# todo: probably 4.1 for the _1's ?
GGML_SIZE_MAP = {
    0: 32,
    1: 16,
    2: 4,
    3: 4,
    6: 5,
    7: 5,
    8: 8,
    9: 8,
    10: 2,
    11: 3,
    12: 4,
    13: 5,
    14: 6,
    15: 8
}


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val):
        if isinstance(val, str) or isinstance(val, bytes) or isinstance(val, bytearray):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            raise ValueError("Unknown type: " + str(type(val)))


GGUFValuePacking = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
    GGUFValueType.BOOL: "?",
}


class GGUFReader:
    fin: BinaryIO
    gguf_version: int
    ti_data_count: int
    kv_data_count: int
    kv_data: Dict[str, Union[int, float, str, bool, list[Any]]]
    data_alignment = GGUF_DEFAULT_ALIGNMENT

    def __init__(self, path: os.PathLike[str] | str):
        self.ti_info = []
        self.fin = open(path, "rb")
        self.read_header()
        self.read_kv_data()
        self.read_tensor_info()
        self.skip_padding(self.fin, self.fin.tell())

    @staticmethod
    def ggml_pad(x: int, n: int) -> int:
        return ((x + n - 1) // n) * n

    def skip_padding(self, fp: BinaryIO, n: int, align: int | None = None):
        pad = GGUFReader.ggml_pad(n, align if align is not None else self.data_alignment) - n
        if pad != 0:
            fp.seek(pad, SEEK_CUR)

    def read_header(self):
        magic = struct.unpack("<I", self.fin.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError("Invalid GGUF file magic number")

        self.gguf_version = struct.unpack("<I", self.fin.read(4))[0]
        self.ti_data_count = struct.unpack("<Q", self.fin.read(8))[0]
        self.kv_data_count = struct.unpack("<Q", self.fin.read(8))[0]

    def read_kv_data(self):
        self.kv_data = {}
        for _ in range(self.kv_data_count):
            key, val = self.read_kv_pair()
            if key == KEY_GENERAL_ALIGNMENT:
                self.data_alignment = val
            self.kv_data[key] = val

    def read_tensor_info(self):
        for i in range(self.ti_data_count):
            tensor_info = {}

            # Read tensor name length and the name itself
            name_length_bytes = self.fin.read(8)
            name_length = struct.unpack("<Q", name_length_bytes)[0]
            encoded_name = self.fin.read(name_length)
            tensor_info['name'] = encoded_name.decode("utf-8")

            # Read tensor shape (n_dims first, then the actual dimensions)
            n_dims_bytes = self.fin.read(4)
            n_dims = struct.unpack("<I", n_dims_bytes)[0]
            shape = []
            for _ in range(n_dims):
                dim_bytes = self.fin.read(8)
                dim = struct.unpack("<Q", dim_bytes)[0]
                shape.append(dim)
            tensor_info['shape'] = tuple(reversed(shape))

            # Read data type
            dtype_bytes = self.fin.read(4)
            dtype = struct.unpack("<I", dtype_bytes)[0]
            tensor_info['dtype'] = dtype  # You might want to map this back to a human-readable string

            # Read tensor offset
            offset_bytes = self.fin.read(8)
            offset_tensor = struct.unpack("<Q", offset_bytes)[0]
            tensor_info['offset_tensor'] = offset_tensor

            # Append to tensor info list
            self.ti_info.append(tensor_info)

    def read_val(self, vtype) -> Any:
        if vtype == GGUFValueType.STRING:
            val = self.read_string()
        elif vtype in GGUFValuePacking:
            fmt = GGUFValuePacking[vtype]
            val = struct.unpack(fmt, self.fin.read(struct.calcsize(fmt)))[0]
        elif vtype == GGUFValueType.ARRAY:
            ltype = GGUFValueType(struct.unpack("<I", self.fin.read(4))[0])
            length = struct.unpack("<Q", self.fin.read(8))[0]
            val = [self.read_val(ltype) for _ in range(length)]
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")
        return val

    def read_kv_pair(self) -> Tuple[str, Any]:
        key = self.read_string()
        vtype = GGUFValueType(struct.unpack("<I", self.fin.read(4))[0])

        val = self.read_val(vtype)
        return key, val

    def read_string(self) -> str:
        length = struct.unpack("<Q", self.fin.read(8))[0]
        return self.fin.read(length).decode("utf8")

    def close(self):
        self.fin.close()

    def quantization_name(self):
        kvd = self.kv_data
        if qv:= kvd.get("general.quantization_version"):
            return GGML_TYPE_MAP.get(qv)
        return None

    def value_size(self):
        num_size = 32
        kvd = self.kv_data
        if qv:= kvd.get("general.quantization_version"):
            return GGML_SIZE_MAP.get(qv, 16)
        return 16

    def vram_estimate(self):
        num_size = self.value_size()

        tot_size = 0
        for ent in self.ti_info:
            size = reduce(operator.mul, ent["shape"], 1)
            tot_size += size

        return int(tot_size * num_size / 8)

    def layers(self):
        # todo: no idea if this is right
        if blks := self.kv_data.get("llama.block_count"):
            return blks + 3
        return None

    def summary(self) -> Dict[str, Any]:
        kvd = self.kv_data
        tokens = kvd.pop("tokenizer.ggml.tokens", None)
        kvd.pop("tokenizer.ggml.scores", None)
        kvd.pop("tokenizer.ggml.token_type", None)
        if tokens:
            kvd["tokenizer.ggml.token_count"] = len(tokens)

        vram_est = self.vram_estimate()
        num_size = self.value_size()
        qname = self.quantization_name() or "unknown"

        return {
            "gguf.version": self.gguf_version,
            "kv.data": kvd,
            "tensor.count": len(self.ti_info),
            "tensor.q_size": num_size,
            "tensor.q_name": qname,
            "tensor.layers": self.layers(),
            "tensor.ram_estimate": vram_est,
            "tensor.data": self.ti_info,
        }


# Make sure to define GGUF_MAGIC and GGUFValueType somewhere above this code or import it from where it is defined.

__all__ = GGUFReader

# Example usage
if __name__ == "__main__":
    reader = GGUFReader(sys.argv[1])
    rs = reader.summary()
    rs.pop("tensor.data")
    pprint.pprint(rs, indent=4, sort_dicts=False)
    reader.close()
