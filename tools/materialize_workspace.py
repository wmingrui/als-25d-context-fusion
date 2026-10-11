"""Create a separate runnable copy of the archived directory layout."""
import argparse
import hashlib
import json
from pathlib import Path
import re


OLD_ROOT = '/root/autodl-tmp'


def materialize(source, target):
    source = Path(source).resolve()
    target = Path(target).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target == source.parent or source.parent in target.parents:
        raise ValueError('The workspace must be outside the repository')
    if target.exists():
        raise FileExistsError(target)
    if any(c in target.as_posix() for c in "'\"\n\r"):
        raise ValueError('Workspace path must not contain quotes or newlines')
    files = [p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.json')]
    if any(p.is_symlink() for p in files):
        raise ValueError('Source files must not be symbolic links')
    target.mkdir(parents=True, exist_ok=False)
    pattern = re.compile(re.escape(OLD_ROOT) + r'''(?=/|$|["'])''')
    protected = {'normalization.json', 'shared_25d_norm_v3.json'}
    audit = {'source':str(source), 'workspace':str(target), 'source_hash_updates':[], 'files':[]}
    for p in files:
        rel = p.relative_to(source)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()
        if p.name not in protected:
            data = pattern.sub(lambda match: target.as_posix(), data.decode('utf-8-sig')).encode('utf-8')
        dest.write_bytes(data)
        audit['files'].append({'path':rel.as_posix(), 'repository_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})

    for p in target.rglob('config.json'):
        cfg = json.loads(p.read_text(encoding='utf-8'))
        changed = False
        for name, expected in cfg.get('source_hashes', {}).items():
            file = Path(name)
            if file.is_file() and target in file.resolve().parents:
                digest = hashlib.sha256(file.read_bytes()).hexdigest()
                audit['source_hash_updates'].append({'config':p.relative_to(target).as_posix(), 'file':name, 'original':expected, 'runtime':digest})
                cfg['source_hashes'][name] = digest
                changed = True
        if 'source_sha256' in cfg:
            for parent in p.parents:
                if parent == target:
                    break
                trainer = parent / 'train_complete.py'
                if trainer.is_file():
                    digest = hashlib.sha256(trainer.read_bytes()).hexdigest()
                    audit['source_hash_updates'].append({'config':p.relative_to(target).as_posix(), 'file':str(trainer), 'original':cfg['source_sha256'], 'runtime':digest})
                    cfg['source_sha256'] = digest
                    changed = True
                    break
        if changed:
            p.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    for item in audit['files']:
        item['runtime_sha256'] = hashlib.sha256((target/item['path']).read_bytes()).hexdigest()
    (target/'WORKSPACE_PROVENANCE.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'legacy'
    result = materialize(source,args.workspace)
    print(json.dumps({'workspace':result['workspace'],'files':len(result['files']),'runtime_source_hash_updates':len(result['source_hash_updates'])}))


if __name__ == '__main__':
    main()
