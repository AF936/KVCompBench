"""Full BF16 benchmark with configured blocks and batched GPU decoding."""
import argparse
import configparser
import csv
import json
from pathlib import Path
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from threading import Lock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "compression_framework"))
from gpu_codec import discover_gpu_codecs
from benchmark_routes import discover_routes
from codec_timing import TIMING_VERSION, encode_measured


def cpu_threads(path):
    config = configparser.ConfigParser(interpolation=None)
    config.read(path, encoding='utf-8')
    workers = config.getint('benchmark', 'cpu_encode_threads', fallback=1)
    if not 1 <= workers <= 256:
        raise ValueError('cpu_encode_threads must be between 1 and 256')
    return workers


def decode_batch_blocks(path):
    config = configparser.ConfigParser(interpolation=None)
    config.read(path, encoding='utf-8')
    value = config.getint('benchmark', 'decode_batch_blocks', fallback=256)
    if not 1 <= value <= 4096:
        raise ValueError('decode_batch_blocks must be between 1 and 4096')
    return value


def encoded_blocks(data, block_bytes, codec, completed, dataset, workers):
    # ZipNN's cached native engine is mutable; serialize that encoder.
    lock = Lock() if codec.name == 'zipnn' else None
    if codec.declaration['type'] == 'module':
        codec._load_module()
    def encode(item):
        index, raw = item
        if lock:
            with lock:
                frame,seconds,started,ended = encode_measured(codec,raw,warmup=False)
        else:
            frame,seconds,started,ended = encode_measured(codec,raw,warmup=codec.mode=='GPU-GPU')
        return index, raw, frame, seconds, started, ended
    pending = ((i, start) for i, start in enumerate(range(0, len(data), block_bytes))
               if (dataset, codec.name, i) not in completed)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        warmed=False
        while True:
            batch = list(islice(pending, workers * 2))
            if not batch:
                break
            prepared=[(index,data[start:start+block_bytes].tobytes()) for index,start in batch]
            if not warmed and codec.mode!='GPU-GPU':
                encode_measured(codec,prepared[0][1]);warmed=True
            results = list(pool.map(encode, prepared))
            if codec.mode!='GPU-GPU' and workers>1:
                elapsed=max(r[5] for r in results)-min(r[4] for r in results)
                total=sum(len(r[1]) for r in results)
                results=[(i,raw,frame,elapsed*len(raw)/total,start,end) for i,raw,frame,seconds,start,end in results]
            for index, raw, frame, seconds, started, ended in results:
                yield index,raw,frame,seconds


def load_tensor(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    tensor = value["data"] if isinstance(value, dict) else value
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
        raise ValueError(f"{path.name}: expected a contiguous BF16 tensor")
    return tensor.reshape(-1).view(torch.uint8).numpy()


def read_progress(path):
    completed = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            completed[(record["dataset"], record["codec"], record["block"])] = record
    return completed


def dataset_blocks(settings_path):
    config = configparser.ConfigParser(interpolation=None)
    with settings_path.open(encoding="utf-8") as handle:
        config.read_file(handle)
    if not config.has_section("benchmark_datasets") or not config["benchmark_datasets"]:
        raise ValueError("settings.txt requires a nonempty [benchmark_datasets] section")
    blocks = {}
    for filename, value in config["benchmark_datasets"].items():
        if Path(filename).name != filename or not filename.endswith(".pt"):
            raise ValueError(f"Invalid benchmark dataset filename: {filename}")
        try:
            block_bytes = int(value)
        except ValueError as error:
            raise ValueError(f"Invalid block size for {filename}: {value}") from error
        if block_bytes < 1:
            raise ValueError(f"Block size must be positive: {filename}")
        blocks[filename] = block_bytes
    return blocks


def write_png(path, rows, codec_names):
    # Dependency-free PNG heatmap: rows=datasets, columns=codecs, green=larger ratio.
    margin, cell_w, cell_h = 220, 210, 95
    width, height = max(980, margin + len(codec_names) * cell_w + 30), 340
    pixels = bytearray([13, 20, 31, 255]) * (width * height)
    font = {
        'C':('01111','10000','10000','10000','10000','10000','01111'),
        'U':('10001','10001','10001','10001','10001','10001','01110'),
        'O':('01110','10001','10001','10001','10001','10001','01110'),
        'B':('11110','10001','10001','11110','10001','10001','11110'),
        'M':('10001','11011','10101','10101','10001','10001','10001'),
        'V':('10001','10001','10001','10001','10001','01010','00100'),
        'H':('10001','10001','10001','11111','10001','10001','10001'),
        '-':('00000','00000','00000','11111','00000','00000','00000'),
        ':':('00000','00100','00100','00000','00100','00100','00000'),
        "A":("01110","10001","10001","11111","10001","10001","10001"), "D":("11110","10001","10001","10001","10001","10001","11110"),
        "E":("11111","10000","10000","11110","10000","10000","11111"), "F":("11111","10000","10000","11110","10000","10000","10000"),
        "G":("01111","10000","10000","10111","10001","10001","01110"), "I":("11111","00100","00100","00100","00100","00100","11111"),
        "K":("10001","10010","10100","11000","10100","10010","10001"), "L":("10000","10000","10000","10000","10000","10000","11111"),
        "N":("10001","11001","10101","10011","10001","10001","10001"), "P":("11110","10001","10001","11110","10000","10000","10000"),
        "Q":("01110","10001","10001","10001","10101","10010","01101"), "R":("11110","10001","10001","11110","10100","10010","10001"),
        "S":("01111","10000","10000","01110","00001","00001","11110"), "T":("11111","00100","00100","00100","00100","00100","00100"),
        "W":("10001","10001","10001","10101","10101","10101","01010"), "Y":("10001","10001","01010","00100","00100","00100","00100"),
        "Z":("11111","00001","00010","00100","01000","10000","11111"), "0":("01110","10001","10011","10101","11001","10001","01110"),
        "1":("00100","01100","00100","00100","00100","00100","01110"), "2":("01110","10001","00001","00010","00100","01000","11111"),
        "3":("11110","00001","00001","01110","00001","00001","11110"), "4":("00010","00110","01010","10010","11111","00010","00010"),
        "5":("11111","10000","10000","11110","00001","00001","11110"), "6":("01110","10000","10000","11110","10001","10001","01110"),
        "7":("11111","00001","00010","00100","01000","01000","01000"), "8":("01110","10001","10001","01110","10001","10001","01110"),
        "9":("01110","10001","10001","01111","00001","00001","01110"), ".":("00000","00000","00000","00000","00000","00110","00110"), " ":("00000",)*7,
    }
    def pixel(x, y, color):
        if 0 <= x < width and 0 <= y < height:
            pos = 4 * (y * width + x); pixels[pos:pos + 4] = bytes(color)
    def text(x, y, value, color=(222, 235, 247, 255), scale=2):
        for char in value.upper():
            for ry, row in enumerate(font.get(char, font[" "])):
                for rx, bit in enumerate(row):
                    if bit == "1":
                        for dy in range(scale):
                            for dx in range(scale): pixel(x + rx * scale + dx, y + ry * scale + dy, color)
            x += 6 * scale
    text(margin, 12, 'CODEC PATH', scale=2)
    maximum = max(1.0, *(row["compression_ratio"] for row in rows))
    for y, dataset in enumerate(sorted({row["dataset"] for row in rows})):
        for x, codec in enumerate(codec_names):
            row = next((r for r in rows if r["dataset"] == dataset and r["codec"] == codec), None)
            if not row:
                continue
            scale = min(1.0, row["compression_ratio"] / maximum)
            color = (25, int(70 + 155 * scale), int(95 + 90 * scale), 255)
            x0, y0 = margin + x * cell_w, 95 + y * cell_h
            for py in range(y0, y0 + cell_h - 8):
                for px in range(x0, x0 + cell_w - 8):
                    offset = 4 * (py * width + px)
                    pixels[offset:offset + 4] = bytes(color)
            text(x0 + 12, y0 + 20, f"{row['compression_ratio']:.2f}", scale=3)
    headers = {"zstd": "ZSTD", "snappy": "SNAPPY", "gzip": "GZIP", "lz4_block": "LZ4", "raw_deflate": "RAW DEFLATE"}
    for index, codec in enumerate(codec_names):
        text(margin + index * cell_w, 47, headers.get(codec, codec.upper()), scale=1)
        mode = next(r['execution_path'] for r in rows if r['codec'] == codec)
        text(margin + index * cell_w, 67, mode, scale=2)
    for index, dataset in enumerate(sorted({row["dataset"] for row in rows})): text(22, 125 + index * cell_h, dataset.replace("_bf16", ""), scale=2)
    raw = b"".join(b"\0" + pixels[y * width * 4:(y + 1) * width * 4] for y in range(height))
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(png)
