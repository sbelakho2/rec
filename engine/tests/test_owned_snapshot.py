import unittest

import mlx.core as mx

from rec_mlx.cache import FixedKVCache, FixedSlidingKVCache, copy_array
from rec_mlx.m4_max import KVStorage
from rec_mlx.owned_snapshot import SnapshotIdentityError, capture, restore
from rec_mlx.prefix import Checkpoint, PromptCache


def metadata():
    rows = [[0] * 64 for _ in range(50)]
    rows[0][1] = 1
    rows[1][1] = 0
    return rows


def make_layers(count, storage, window):
    ring = FixedSlidingKVCache(
        max_sequence_tokens=count, num_kv_heads=4, head_dim=128, window_size=window
    )
    return [FixedKVCache(storage, 0), ring]


def make_cache(count=40, block_tokens=65536, window=513):
    storage = KVStorage(metadata(), count, block_tokens=block_tokens)
    keys = mx.arange(4 * count * 128, dtype=mx.float32).astype(mx.bfloat16).reshape(1, 4, count, 128)
    values = -keys
    storage.write(0, 0, keys, values)
    layers = make_layers(count, storage, window)
    layers[0].offset = count
    ring = layers[1]
    half = max(1, count // 2)
    checkpoint = None
    for index in range(count):
        key = mx.full((1, 4, 1, 128), float(index), dtype=mx.bfloat16)
        value = -key
        ring.update_and_fetch(key, value)
        if index + 1 == half:
            checkpoint = Checkpoint(
                half,
                [(copy_array(ring.key_buffers[ring._slot]), copy_array(ring.value_buffers[ring._slot]), ring._valid, ring._ring_index)],
                mx.arange(8, dtype=mx.float32).reshape(1, 1, 8),
            )
    ring.offset = count
    cache = PromptCache(
        list(range(count)), layers, mx.arange(8, dtype=mx.float32).reshape(1, 1, 8), storage
    )
    if checkpoint is not None:
        cache.checkpoints = [checkpoint]
    return cache


def storage_factory(rows, capacity, block_tokens):
    return KVStorage(rows, capacity, block_tokens=block_tokens)


class OwnedSnapshotTests(unittest.TestCase):
    def test_a_to_b_to_a_owns_global_and_sliding_state(self):
        cache = make_cache()
        saved_global = copy_array(cache.storage.layer(0, 40))
        saved_sliding = copy_array(cache.layers[1]._prior()[0])
        saved_logits = copy_array(cache.logits)
        snapshot = capture(cache, cache.tokens, "identity-a")
        cache.storage.write(
            0,
            0,
            mx.zeros((1, 4, 40, 128), dtype=mx.bfloat16),
            mx.zeros((1, 4, 40, 128), dtype=mx.bfloat16),
        )
        cache.layers[1].update_and_fetch(
            mx.full((1, 4, 1, 128), 77, dtype=mx.bfloat16),
            mx.full((1, 4, 1, 128), 77, dtype=mx.bfloat16),
        )
        restored = restore(snapshot, "identity-a", storage_factory, lambda n, s: make_layers(n, s, 513))
        self.assertEqual(restored.tokens, list(range(40)))
        self.assertTrue(mx.array_equal(restored.storage.layer(0, 40), saved_global).item())
        self.assertTrue(mx.array_equal(restored.layers[1]._prior()[0], saved_sliding).item())
        self.assertTrue(mx.array_equal(restored.logits, saved_logits).item())
        self.assertEqual(restored.layers[0].offset, 40)

    def test_identity_mismatch_refused(self):
        snapshot = capture(make_cache(), list(range(40)), "identity-a")
        with self.assertRaises(SnapshotIdentityError):
            restore(snapshot, "identity-b", storage_factory, lambda n, s: make_layers(n, s, 513))

    def test_checkpoint_state_is_owned(self):
        cache = make_cache()
        checkpoint = cache.checkpoints[0]
        saved_keys = copy_array(checkpoint.sliding[0][0])
        snapshot = capture(cache, cache.tokens, "identity-a")
        zero = mx.zeros_like(saved_keys)
        cache.checkpoints[0] = Checkpoint(
            20,
            [(zero, mx.zeros_like(checkpoint.sliding[0][1]), checkpoint.sliding[0][2], checkpoint.sliding[0][3])],
            mx.zeros_like(checkpoint.logits),
        )
        restored = restore(snapshot, "identity-a", storage_factory, lambda n, s: make_layers(n, s, 513))
        self.assertEqual(restored.checkpoints[0].position, 20)
        self.assertTrue(mx.array_equal(restored.checkpoints[0].sliding[0][0], saved_keys).item())

    def test_prefix_lengths_are_isolated(self):
        for count in (1, 20, 39):
            cache = make_cache(count=count)
            saved = copy_array(cache.storage.layer(0, count))
            snapshot = capture(cache, cache.tokens, "identity-a")
            cache.storage.write(
                0,
                0,
                mx.zeros((1, 4, count, 128), dtype=mx.bfloat16),
                mx.zeros((1, 4, count, 128), dtype=mx.bfloat16),
            )
            restored = restore(snapshot, "identity-a", storage_factory, lambda n, s: make_layers(n, s, 513))
            self.assertEqual(restored.tokens, list(range(count)))
            self.assertTrue(mx.array_equal(restored.storage.layer(0, count), saved).item())

    def test_cross_65536_block_boundary(self):
        cache = make_cache(count=66000)
        saved = copy_array(cache.storage.layer(0, 66000))
        snapshot = capture(cache, cache.tokens, "identity-a")
        restored = restore(snapshot, "identity-a", storage_factory, lambda n, s: make_layers(n, s, 513))
        self.assertTrue(mx.array_equal(restored.storage.layer(0, 66000), saved).item())

    def test_sliding_ring_wrap_restores_exactly(self):
        cache = make_cache(count=600, window=513)
        saved = copy_array(cache.layers[1]._prior()[0])
        snapshot = capture(cache, cache.tokens, "identity-a")
        cache.layers[1].update_and_fetch(
            mx.full((1, 4, 1, 128), 99, dtype=mx.bfloat16),
            mx.full((1, 4, 1, 128), 99, dtype=mx.bfloat16),
        )
        restored = restore(snapshot, "identity-a", storage_factory, lambda n, s: make_layers(n, s, 513))
        self.assertTrue(mx.array_equal(restored.layers[1]._prior()[0], saved).item())
        self.assertEqual(restored.layers[1]._valid, 513)

    def test_payload_is_bounded_by_used_prefix(self):
        cache = make_cache(count=4, window=4)
        snapshot = capture(cache, cache.tokens, "identity-a")
        storage_bytes = sum(int(bank.nbytes) for bank in cache.storage.banks)
        self.assertLess(snapshot.nbytes, storage_bytes)
        self.assertLess(snapshot.nbytes, storage_bytes // 2)


if __name__ == "__main__":
    unittest.main()
