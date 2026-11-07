"""Opt-in durable prefix snapshots for the MLX engine.

A snapshot stores only the prompt prefix that was actually prefilled, not the
unused KV capacity. Loading is guarded by a model fingerprint, exact token IDs,
a payload checksum and shape checks. A failed validation is a cache miss; it
never installs partially validated state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .prefix import Checkpoint, PromptCache

FORMAT_VERSION = 1
MAGIC = "rec-prefix-snapshot"
CHUNK = 1 << 20


class SnapshotError(RuntimeError):
    pass


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _token_bytes(tokens) -> bytes:
    return struct.pack(f"<{len(tokens)}i", *tokens)


def token_digest(tokens) -> str:
    return _sha_bytes(_token_bytes(tokens))


def model_fingerprint(identity, startup) -> str:
    fields = {
        "engine": "mlx",
        "identity": identity,
        "kv_format": startup.get("kv_format"),
        "weight_format": startup.get("weight_format"),
        "chat_template_sha256": startup.get("chat_template_sha256"),
    }
    return _sha_bytes(json.dumps(fields, sort_keys=True, default=str).encode())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _eval(arrays) -> None:
    mx.eval(list(arrays))
    mx.synchronize()


def _array_meta(array) -> dict:
    return {"shape": list(array.shape), "dtype": str(array.dtype), "nbytes": array.nbytes}


@dataclass
class SnapshotEntry:
    key: str
    path: Path
    token_count: int
    token_sha256: str
    size: int
    last_used: float


class SnapshotStore:
    def __init__(
        self,
        directory,
        fingerprint,
        *,
        storage_factory,
        layer_factory,
        max_bytes=0,
        min_tokens=4096,
    ):
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self.storage_factory = storage_factory
        self.layer_factory = layer_factory
        self.max_bytes = int(max_bytes or 0)
        self.min_tokens = int(min_tokens)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass
        self._entries: dict[str, SnapshotEntry] = {}
        self._scan()

    def _manifest_path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def _payload_path(self, key: str) -> Path:
        return self.directory / f"{key}.safetensors"

    def _scan(self) -> None:
        for manifest_path in sorted(self.directory.glob("*.json")):
            try:
                manifest = json.loads(manifest_path.read_text())
                if (
                    manifest.get("magic") != MAGIC
                    or manifest.get("format_version") != FORMAT_VERSION
                    or manifest.get("model_fingerprint") != self.fingerprint
                ):
                    continue
                key = manifest["key"]
                payload = self._payload_path(key)
                if not payload.is_file():
                    continue
                self._entries[key] = SnapshotEntry(
                    key=key,
                    path=payload,
                    token_count=int(manifest["token_count"]),
                    token_sha256=manifest["token_sha256"],
                    size=payload.stat().st_size,
                    last_used=time.time(),
                )
            except Exception as error:
                logging.warning("Ignoring invalid prefix snapshot manifest %s: %s", manifest_path, error)

    def _total_bytes(self) -> int:
        return sum(entry.size for entry in self._entries.values())

    def _remove(self, entry: SnapshotEntry) -> None:
        self._entries.pop(entry.key, None)
        for path in (entry.path, self._manifest_path(entry.key)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _evict(self, keep: str | None = None) -> None:
        if self.max_bytes <= 0:
            return
        for entry in sorted(self._entries.values(), key=lambda item: item.last_used):
            if self._total_bytes() <= self.max_bytes:
                break
            if entry.key == keep:
                continue
            self._remove(entry)

    def _match(self, tokens) -> tuple[SnapshotEntry | None, list[int] | None]:
        digests: dict[int, str] = {}
        for entry in sorted(
            self._entries.values(), key=lambda item: item.token_count, reverse=True
        ):
            if entry.token_count > len(tokens):
                continue
            digest = digests.get(entry.token_count)
            if digest is None:
                digest = token_digest(tokens[: entry.token_count])
                digests[entry.token_count] = digest
            if digest == entry.token_sha256:
                return entry, list(tokens[: entry.token_count])
        return None, None

    def best_prefix_length(self, tokens) -> int:
        entry, _ = self._match(tokens)
        return 0 if entry is None else entry.token_count

    def load(self, tokens) -> PromptCache | None:
        if not self._entries:
            return None
        entry, prefix = self._match(tokens)
        if entry is None:
            return None
        try:
            manifest = json.loads(self._manifest_path(entry.key).read_text())
            if (
                manifest.get("magic") != MAGIC
                or manifest.get("format_version") != FORMAT_VERSION
                or manifest.get("model_fingerprint") != self.fingerprint
                or manifest.get("token_count") != entry.token_count
                or manifest.get("token_sha256") != entry.token_sha256
            ):
                raise SnapshotError("manifest does not match the index entry")
            if _file_sha256(entry.path) != manifest.get("payload_sha256"):
                raise SnapshotError("payload checksum mismatch")
            arrays = mx.load(str(entry.path))
            cache = self._restore(manifest, arrays, prefix)
            entry.last_used = time.time()
            return cache
        except Exception as error:
            logging.warning("Discarding prefix snapshot %s: %s", entry.key[:12], error)
            self._remove(entry)
            return None

    def _restore(self, manifest, arrays, tokens) -> PromptCache:
        count = int(manifest["token_count"])
        storage = self.storage_factory(manifest["rows"], count, manifest["block_tokens"])
        global_layers = manifest["global_layers"]
        for layer in global_layers:
            index = layer["index"]
            keys = arrays[f"g{index}_k"]
            values = arrays[f"g{index}_v"]
            if list(keys.shape) != layer["k_shape"] or list(values.shape) != layer["v_shape"]:
                raise SnapshotError(f"shape mismatch for global layer {index}")
            storage.write(index, 0, keys, values)
        layers = self.layer_factory(count, storage)
        for layer in global_layers:
            layers[layer["index"]].offset = count
        for saved in manifest["sliding_layers"]:
            index = saved["index"]
            layer = layers[index]
            layer.key_buffers = (arrays[f"s{index}_k0"], arrays[f"s{index}_k1"])
            layer.value_buffers = (arrays[f"s{index}_v0"], arrays[f"s{index}_v1"])
            layer.capacity = saved["capacity"]
            layer.offset = saved["offset"]
            layer._valid = saved["valid"]
            layer._slot = saved["slot"]
            layer._ring_index = saved["ring_index"]
            layer.window_size = saved["window_size"]
        logits = arrays["logits"]
        if list(logits.shape) != manifest["logits"]["shape"]:
            raise SnapshotError("logits shape mismatch")
        checkpoints = []
        for index, saved in enumerate(manifest["checkpoints"]):
            sliding = []
            for position, entry in enumerate(saved["sliding"]):
                sliding.append(
                    (
                        arrays[f"cp{index}_s{position}_k"],
                        arrays[f"cp{index}_s{position}_v"],
                        entry["valid"],
                        entry["ring_index"],
                    )
                )
            checkpoints.append(
                Checkpoint(saved["position"], sliding, arrays[f"cp{index}_logits"])
            )
        return PromptCache(list(tokens), layers, logits, storage, checkpoints)

    def save(self, cache: PromptCache, tokens) -> Path | None:
        count = len(tokens)
        if count < self.min_tokens:
            return None
        key = _sha_bytes((self.fingerprint + token_digest(tokens)).encode())
        if key in self._entries:
            self._entries[key].last_used = time.time()
            return self._entries[key].path
        storage = cache.storage
        rows = storage.rows
        block_tokens = getattr(storage, "block_tokens", 65536)
        arrays = {"tokens": mx.array(tokens, dtype=mx.int32)}
        global_layers = []
        for index, row in enumerate(rows[:49]):
            if not row[1]:
                continue
            data = storage.layer(index, count)
            keys = data[0:1]
            values = data[1:2]
            _eval([keys, values])
            arrays[f"g{index}_k"] = keys
            arrays[f"g{index}_v"] = values
            global_layers.append(
                {
                    "index": index,
                    "k_shape": list(keys.shape),
                    "v_shape": list(values.shape),
                }
            )
        sliding_layers = []
        for index, layer in enumerate(cache.layers):
            if not hasattr(layer, "_ring_index"):
                continue
            key0, key1 = layer.key_buffers
            value0, value1 = layer.value_buffers
            _eval([key0, key1, value0, value1])
            arrays[f"s{index}_k0"] = key0
            arrays[f"s{index}_k1"] = key1
            arrays[f"s{index}_v0"] = value0
            arrays[f"s{index}_v1"] = value1
            sliding_layers.append(
                {
                    "index": index,
                    "capacity": layer.capacity,
                    "offset": layer.offset,
                    "valid": layer._valid,
                    "slot": layer._slot,
                    "ring_index": layer._ring_index,
                    "window_size": layer.window_size,
                }
            )
        if cache.logits is None:
            return None
        _eval([cache.logits])
        arrays["logits"] = cache.logits
        checkpoints = []
        for checkpoint_index, checkpoint in enumerate(cache.checkpoints):
            sliding = []
            for position, (keys, values, valid, ring_index) in enumerate(checkpoint.sliding):
                _eval([keys, values])
                arrays[f"cp{checkpoint_index}_s{position}_k"] = keys
                arrays[f"cp{checkpoint_index}_s{position}_v"] = values
                sliding.append({"valid": valid, "ring_index": ring_index})
            _eval([checkpoint.logits])
            arrays[f"cp{checkpoint_index}_logits"] = checkpoint.logits
            checkpoints.append({"position": checkpoint.position, "sliding": sliding})
        manifest = {
            "magic": MAGIC,
            "format_version": FORMAT_VERSION,
            "key": key,
            "model_fingerprint": self.fingerprint,
            "token_count": count,
            "token_sha256": token_digest(tokens),
            "block_tokens": block_tokens,
            "rows": rows,
            "global_layers": global_layers,
            "sliding_layers": sliding_layers,
            "checkpoints": checkpoints,
            "logits": _array_meta(cache.logits),
            "payload_sha256": "",
            "created_at": time.time(),
        }
        payload = self._persist(arrays, manifest, count)
        if payload is not None and cache.checkpoints:
            checkpoint = cache.checkpoints[-1]
            if self.min_tokens <= checkpoint.position < count:
                self._save_checkpoint(
                    tokens, checkpoint, arrays, global_layers, sliding_layers, rows, block_tokens
                )
        return payload

    def _save_checkpoint(
        self, tokens, checkpoint, full_arrays, global_layers, sliding_layers, rows, block_tokens
    ):
        count = checkpoint.position
        key = _sha_bytes((self.fingerprint + token_digest(tokens[:count])).encode())
        if key in self._entries:
            return
        arrays = {"tokens": mx.array(tokens[:count], dtype=mx.int32)}
        saved_global = []
        for layer in global_layers:
            index = layer["index"]
            keys = full_arrays[f"g{index}_k"][..., :count, :]
            values = full_arrays[f"g{index}_v"][..., :count, :]
            arrays[f"g{index}_k"] = keys
            arrays[f"g{index}_v"] = values
            saved_global.append(
                {
                    "index": index,
                    "k_shape": list(keys.shape),
                    "v_shape": list(values.shape),
                }
            )
        saved_sliding = []
        for position, (keys, values, valid, ring_index) in enumerate(checkpoint.sliding):
            if position >= len(sliding_layers):
                break
            layer = sliding_layers[position]
            index = layer["index"]
            arrays[f"s{index}_k0"] = keys
            arrays[f"s{index}_k1"] = mx.zeros_like(keys)
            arrays[f"s{index}_v0"] = values
            arrays[f"s{index}_v1"] = mx.zeros_like(values)
            saved_sliding.append(
                {
                    "index": index,
                    "capacity": keys.shape[-2],
                    "offset": count,
                    "valid": valid,
                    "slot": 0,
                    "ring_index": ring_index,
                    "window_size": layer["window_size"],
                }
            )
        arrays["logits"] = checkpoint.logits
        _eval(list(arrays.values()))
        manifest = {
            "magic": MAGIC,
            "format_version": FORMAT_VERSION,
            "key": key,
            "model_fingerprint": self.fingerprint,
            "token_count": count,
            "token_sha256": token_digest(tokens[:count]),
            "block_tokens": block_tokens,
            "rows": rows,
            "global_layers": saved_global,
            "sliding_layers": saved_sliding,
            "checkpoints": [],
            "logits": _array_meta(checkpoint.logits),
            "payload_sha256": "",
            "created_at": time.time(),
        }
        self._persist(arrays, manifest, count)

    def _persist(self, arrays, manifest, count):
        key = manifest["key"]
        stamp = f"{os.getpid()}.{time.time_ns()}"
        payload_tmp = self.directory / f".{key}.{stamp}.payload.safetensors"
        manifest_tmp = self.directory / f".{key}.{stamp}.json.tmp"
        try:
            mx.save_safetensors(str(payload_tmp), arrays)
            os.chmod(payload_tmp, 0o600)
            _fsync_file(payload_tmp)
            size = payload_tmp.stat().st_size
            if 0 < self.max_bytes < size:
                logging.warning(
                    "Prefix snapshot needs %d bytes, above the %d-byte budget", size, self.max_bytes
                )
                payload_tmp.unlink()
                return None
            manifest["payload_sha256"] = _file_sha256(payload_tmp)
            payload = self._payload_path(key)
            os.replace(payload_tmp, payload)
            _fsync_dir(self.directory)
            manifest_tmp.write_text(json.dumps(manifest) + "\n")
            os.chmod(manifest_tmp, 0o600)
            _fsync_file(manifest_tmp)
            os.replace(manifest_tmp, self._manifest_path(key))
            _fsync_dir(self.directory)
            self._entries[key] = SnapshotEntry(
                key=key,
                path=payload,
                token_count=count,
                token_sha256=manifest["token_sha256"],
                size=size,
                last_used=time.time(),
            )
            self._evict(keep=key)
            logging.info(
                "Saved prefix snapshot: tokens=%d bytes=%d total=%d",
                count,
                size,
                self._total_bytes(),
            )
            return payload
        except Exception as error:
            logging.warning("Could not save prefix snapshot: %s", error)
            for path in (payload_tmp, manifest_tmp):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
            return None
