import time
from full_evaluation import all_jobs,Q
from evaluation import sha,write
assert 'All 38 full-retained adapters passed' in (Q/'full_adapter_test.log').read_text()
rows=all_jobs()
assert not (Q/'FULL_PROTOCOL_LOCK.json').exists()
write(Q/'FULL_PROTOCOL_LOCK.json',dict(created=time.time(),jobs=rows,checkpoints={j['id']:j['sha256'] for j in rows},source_sha256=sha(Q/'full_evaluation.py'),evaluation_source_sha256=sha(Q/'evaluation.py'),severity='medium',split='test',sampling='seed42 filename-derived permutation; disjoint groups of8192; last group cyclic padding from its own valid points; discard padding predictions',input_support='O-sampled constructs maps only from unique records in each group; others retain full-block context as trained',metrics='All retained records count exactly once; all classes; not official whole-scene benchmark',prediction_cache='uint8 prediction array in original retained-record order; no smoothing, interpolation, propagation or invented points',validated_adapters=38,reference_order='Base/O first, then new controls, then original F controls'))
print('38-model full-retained protocol locked.')
