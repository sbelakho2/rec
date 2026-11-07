import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from rec_mlx.cache import FixedKVCache, FixedSlidingKVCache, copy_array
from rec_mlx.m4_max import KVStorage
from rec_mlx.prefix import PromptCache
from rec_mlx.snapshots import SnapshotStore, model_fingerprint, token_digest

TOKENS = list(range(8))


def metadata():
    rows = [[0] * 64 for _ in range(50)]
    rows[0][1] = 1
    rows[1][1] = 0
    return rows


def make_layers(count, storage):
    ring = FixedSlidingKVCache(
        max_sequence_tokens=count, num_kv_heads=4, head_dim=128, window_size=count
    )
    return [FixedKVCache(storage, 0), ring]


def make_cache():
    storage = KVStorage(metadata(), 8, block_tokens=16)
    keys = mx.arange(4 * 8 * 128, dtype=mx.float32).astype(mx.bfloat16).reshape(1, 4, 8, 128)
    values = -keys
    storage.write(0, 0, keys, values)
    layers = make_layers(8, storage)
    layers[0].offset = 8
    for index in range(8):
        layer_keys = mx.full((1, 4, 1, 128), float(index), dtype=mx.bfloat16)
        layer_values = -layer_keys
        layers[1].update_and_fetch(layer_keys, layer_values)
    layers[1].offset = 8
    logits = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4)
    return PromptCache(TOKENS[:], layers, logits, storage)


def store(directory, fingerprint="test-model", **kwargs):
    kwargs.setdefault("min_tokens", 1)
    return SnapshotStore(
        directory,
        fingerprint,
        storage_factory=lambda rows, capacity, block_tokens: KVStorage(
            rows, capacity, block_tokens=block_tokens
        ),
        layer_factory=make_layers,
        **kwargs,
    )


class SnapshotTests(unittest.TestCase):
    def test_digest_and_fingerprint_are_stable(self):
        self.assertEqual(token_digest(TOKENS), token_digest(TOKENS[:]))
        self.assertNotEqual(token_digest(TOKENS), token_digest([0, 1, 2]))
        one = model_fingerprint({"model_id": "flint"}, {"kv_format": "bf16"})
        two = model_fingerprint({"model_id": "flint"}, {"kv_format": "bf16"})
        other = model_fingerprint({"model_id": "flint"}, {"kv_format": "int8"})
        self.assertEqual(one, two)
        self.assertNotEqual(one, other)

    def test_roundtrip_restores_global_sliding_logits_and_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = make_cache()
            saved_global = copy_array(cache.storage.layer(0, 8))
            saved_sliding = copy_array(cache.layers[1]._prior()[0])
            saved_logits = mx.array(cache.logits)
            repository = store(directory)
            path = repository.save(cache, TOKENS)
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())
            self.assertEqual((path.stat().st_mode & 0o777), 0o600)

            restored = repository.load(TOKENS)
            self.assertIsNotNone(restored)
            self.assertEqual(restored.tokens, TOKENS)
            self.assertTrue(mx.array_equal(restored.storage.layer(0, 8), saved_global).item())
            self.assertTrue(mx.array_equal(restored.layers[1]._prior()[0], saved_sliding).item())
            self.assertTrue(mx.array_equal(restored.logits, saved_logits).item())
            self.assertEqual(restored.layers[0].offset, 8)

    def test_snapshot_is_immutable_after_source_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = make_cache()
            saved_global = copy_array(cache.storage.layer(0, 8))
            saved_sliding = copy_array(cache.layers[1]._prior()[0])
            repository = store(directory)
            self.assertIsNotNone(repository.save(cache, TOKENS))
            cache.storage.write(
                0,
                0,
                mx.zeros((1, 4, 8, 128), dtype=mx.bfloat16),
                mx.zeros((1, 4, 8, 128), dtype=mx.bfloat16),
            )
            cache.layers[1].update_and_fetch(
                mx.full((1, 4, 1, 128), 99, dtype=mx.bfloat16),
                mx.full((1, 4, 1, 128), 99, dtype=mx.bfloat16),
            )
            restored = repository.load(TOKENS)
            self.assertTrue(mx.array_equal(restored.storage.layer(0, 8), saved_global).item())
            self.assertTrue(mx.array_equal(restored.layers[1]._prior()[0], saved_sliding).item())

    def test_prefix_match_and_token_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = store(directory)
            self.assertIsNotNone(repository.save(make_cache(), TOKENS))
            restored = repository.load(TOKENS + [123])
            self.assertIsNotNone(restored)
            self.assertEqual(restored.tokens, TOKENS)
            self.assertIsNone(repository.load([0, 1, 2, 99, 4, 5, 6, 7]))
            self.assertIsNone(repository.load([0, 1, 2]))

    def test_fingerprint_mismatch_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNotNone(store(directory, "model-a").save(make_cache(), TOKENS))
            self.assertIsNone(store(directory, "model-b").load(TOKENS))

    def test_corrupt_payload_is_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = store(directory)
            path = repository.save(make_cache(), TOKENS)
            data = bytearray(path.read_bytes())
            data[len(data) // 2] ^= 1
            path.write_bytes(data)
            self.assertIsNone(repository.load(TOKENS))
            self.assertFalse(path.exists())

    def test_budget_evicts_oldest_and_rejects_oversized_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = store(directory, max_bytes=60_000, min_tokens=1)
            first = repository.save(make_cache(), [1] * 8)
            self.assertIsNotNone(first)
            second = repository.save(make_cache(), [2] * 8)
            self.assertIsNotNone(second)
            self.assertLessEqual(repository._total_bytes(), 60_000)
            self.assertEqual(len(repository._entries), 1)
            self.assertFalse(first.exists())
            self.assertIsNone(repository.load([1] * 8))
            self.assertIsNotNone(repository.load([2] * 8))

            with tempfile.TemporaryDirectory() as other:
                small = store(other, max_bytes=1, min_tokens=1)
                self.assertIsNone(small.save(make_cache(), TOKENS))
                self.assertEqual(list(Path(other).glob("*.safetensors")), [])


if __name__ == "__main__":
    unittest.main()
