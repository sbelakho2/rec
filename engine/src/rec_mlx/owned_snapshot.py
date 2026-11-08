"""Off-by-default owned in-memory prefix snapshots.

A snapshot owns independent copies of the used global-KV range, the full sliding
ring state, logits, token IDs and checkpoint state. It never aliases the source
cache's Metal banks. This is the M4 branch-isolation candidate; it is not
enabled in default serving.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx

from .cache import copy_array
from .prefix import Checkpoint, PromptCache


class SnapshotIdentityError(ValueError):
    pass


@dataclass(frozen=True)
class OwnedPrefixSnapshot:
    model_fingerprint: str
    tokens: tuple[int, ...]
    rows: tuple[tuple[int, ...], ...]
    block_tokens: int
    global_layers: tuple[tuple[int, Any, Any], ...]
    sliding_layers: tuple[tuple, ...]
    checkpoints: tuple[tuple[int, tuple[tuple[int, Any, Any, int, int], ...], Any], ...]
    logits: Any
    nbytes: int


def _owned(array):
    copied = copy_array(array)
    mx.eval(copied)
    return copied


def capture(cache: PromptCache, tokens, model_fingerprint: str) -> OwnedPrefixSnapshot:
    tokens = tuple(int(token) for token in tokens)
    if len(tokens) != len(cache.tokens) or list(cache.tokens) != list(tokens):
        raise SnapshotIdentityError("snapshot tokens must equal the cached prompt tokens")
    if cache.logits is None:
        raise SnapshotIdentityError("cache has no logits")
    storage = cache.storage
    rows = tuple(tuple(int(value) for value in row) for row in storage.rows)
    block_tokens = int(getattr(storage, "block_tokens", 65536))
    global_layers = []
    for index, row in enumerate(rows[:49]):
        if not row[1]:
            continue
        data = storage.layer(index, len(tokens))
        global_layers.append((index, _owned(data[0:1]), _owned(data[1:2])))
    sliding_layers = []
    sliding_indices = []
    for index, layer in enumerate(cache.layers):
        if not hasattr(layer, "_ring_index"):
            continue
        sliding_indices.append(index)
        key0, key1 = layer.key_buffers
        value0, value1 = layer.value_buffers
        sliding_layers.append(
            (
                index,
                _owned(key0),
                _owned(key1),
                _owned(value0),
                _owned(value1),
                int(layer.capacity),
                int(layer.offset),
                int(layer._valid),
                int(layer._slot),
                int(layer._ring_index),
                int(layer.window_size),
            )
        )
    checkpoints = []
    for checkpoint in cache.checkpoints:
        sliding = []
        for position, (keys, values, valid, ring_index) in enumerate(checkpoint.sliding):
            if position >= len(sliding_indices):
                raise SnapshotIdentityError("checkpoint has more sliding states than layers")
            sliding.append(
                (
                    sliding_indices[position],
                    _owned(keys),
                    _owned(values),
                    int(valid),
                    int(ring_index),
                )
            )
        checkpoints.append(
            (int(checkpoint.position), tuple(sliding), _owned(checkpoint.logits))
        )
    logits = _owned(cache.logits)
    mx.synchronize()
    nbytes = sum(int(array.nbytes) for _, keys, values in global_layers for array in (keys, values))
    nbytes += sum(
        int(item[1].nbytes) + int(item[2].nbytes) + int(item[3].nbytes) + int(item[4].nbytes)
        for item in sliding_layers
    )
    nbytes += int(logits.nbytes)
    nbytes += sum(
        sum(int(keys.nbytes) + int(values.nbytes) for _, keys, values, _, _ in sliding)
        + int(checkpoint_logits.nbytes)
        for _, sliding, checkpoint_logits in checkpoints
    )
    return OwnedPrefixSnapshot(
        model_fingerprint=model_fingerprint,
        tokens=tokens,
        rows=rows,
        block_tokens=block_tokens,
        global_layers=tuple(global_layers),
        sliding_layers=tuple(sliding_layers),
        checkpoints=tuple(checkpoints),
        logits=logits,
        nbytes=nbytes,
    )


def restore(
    snapshot: OwnedPrefixSnapshot,
    model_fingerprint: str,
    storage_factory: Callable[[tuple, int, int], Any],
    layer_factory: Callable[[int, Any], list],
) -> PromptCache:
    if snapshot.model_fingerprint != model_fingerprint:
        raise SnapshotIdentityError("model/tokenizer identity changed")
    count = len(snapshot.tokens)
    storage = storage_factory(snapshot.rows, count, snapshot.block_tokens)
    for index, keys, values in snapshot.global_layers:
        storage.write(index, 0, keys, values)
    layers = layer_factory(count, storage)
    for index, _, _ in snapshot.global_layers:
        layers[index].offset = count
    for item in snapshot.sliding_layers:
        index, key0, key1, value0, value1, capacity, offset, valid, slot, ring_index, window = item
        layer = layers[index]
        layer.key_buffers = (key0, key1)
        layer.value_buffers = (value0, value1)
        layer.capacity = capacity
        layer.offset = offset
        layer._valid = valid
        layer._slot = slot
        layer._ring_index = ring_index
        layer.window_size = window
    checkpoints = []
    for position, sliding, logits in snapshot.checkpoints:
        checkpoints.append(
            Checkpoint(
                position,
                [(keys, values, valid, ring_index) for _, keys, values, valid, ring_index in sliding],
                logits,
            )
        )
    return PromptCache(list(snapshot.tokens), layers, snapshot.logits, storage, checkpoints)
