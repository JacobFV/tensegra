import torch
from topoformer.return_crossdelay import pooled_indices, feature_batch
from topoformer.return_memory import ReturnMemoryModel
from topoformer.retention_data import make_batch


def test_shared_rows_are_balanced_unique_and_cover_every_event():
    pairs = pooled_indices(2048, [0,1,2,4,8,16], 8192)
    assert len(pairs) == len(set(pairs)) == 8192
    assert len({index for _, index in pairs}) == 2048
    counts = [sum(d == delay for d, _ in pairs) for delay in [0,1,2,4,8,16]]
    assert max(counts) - min(counts) == 1
    assert set(d for d, _ in pairs) == {0,1,2,4,8,16}


def test_source_pool_uses_only_its_declared_delay():
    pairs = pooled_indices(512, [4], 512)
    assert set(pairs) == {(4, index) for index in range(512)}


def test_feature_capture_matches_original_and_pairs_distractors():
    # Width24 is only a mechanical fixture; all experimental checkpoints are1024.
    torch.manual_seed(7)
    model = ReturnMemoryModel(width=24, heads=4).eval()
    first = feature_batch(model, 101, 4, 2, [0,1,2], 2, 'cpu')
    second = feature_batch(model, 101, 4, 8, [0,1,2], 2, 'cpu')
    assert first['event_sha256'] == second['event_sha256']
    public = make_batch(101, 4, distractors=2)['public']
    with torch.no_grad():
        for delay in [0,1,2]:
            expected = model.norm(model(public, delay)['state'])[:,0]
            torch.testing.assert_close(first['features'][delay], expected)
