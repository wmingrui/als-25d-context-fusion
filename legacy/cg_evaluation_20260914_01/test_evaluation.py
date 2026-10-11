"""Acceptance test: reproduce selected validation metrics before accessing test."""
from evaluation import evaluate, jobs
for job in jobs():
    result=evaluate(job,'val','medium')
    assert abs(result['metrics']['mIoU']-job['best_validation_mIoU'])<1e-6,(job['path'],result['metrics'])
print('All 14 selected validation metrics reproduced.')
