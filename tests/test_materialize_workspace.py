import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / 'tools/materialize_workspace.py'


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.exists(), 'Workspace materialization has not been implemented')
        spec = importlib.util.spec_from_file_location('materialize', TOOL)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'repository/legacy'
        self.source.mkdir(parents=True)
        self.target = self.root / 'workspace with spaces'

    def write(self, name, value):
        path = self.source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.encode('utf-8'))
        return path

    def test_relocates_paths_without_changing_numerical_settings(self):
        self.write('project/train.py', "ROOT='/root/autodl-tmp/project'\nLR=0.001\n")
        self.write('project/config.json', json.dumps({'block_dir':'/root/autodl-tmp/data','seed':43,'lr':0.001}))
        self.module.materialize(self.source, self.target)
        cfg=json.loads((self.target/'project/config.json').read_text())
        self.assertEqual(cfg, {'block_dir':self.target.as_posix()+'/data','seed':43,'lr':0.001})
        self.assertIn('LR=0.001', (self.target/'project/train.py').read_text())

    def test_rejects_existing_target_without_touching_it(self):
        self.target.mkdir()
        marker=self.target/'keep';marker.write_text('original')
        with self.assertRaises(FileExistsError):self.module.materialize(self.source,self.target)
        self.assertEqual(marker.read_text(),'original')

    def test_rejects_target_within_source(self):
        with self.assertRaises(ValueError):self.module.materialize(self.source,self.source/'nested')

    def test_preserves_normalization_bytes_and_source_files(self):
        original='{"source":"/root/autodl-tmp/data", "stats_sha256":"fixed", "value":1.25}\n'
        paths=[self.write('project/normalization.json',original),self.write('project/shared_25d_norm_v3.json',original)]
        self.module.materialize(self.source,self.target)
        for p in paths:
            self.assertEqual(p.read_bytes(),original.encode())
            self.assertEqual((self.target/p.relative_to(self.source)).read_bytes(),original.encode())

    def test_reseals_runtime_source_hashes_and_records_old_values(self):
        self.write('project/train.py', "ROOT='/root/autodl-tmp/project'\n")
        self.write('project/config.json',json.dumps({'source_hashes':{'/root/autodl-tmp/project/train.py':'old'},'seed':43}))
        self.module.materialize(self.source,self.target)
        cfg=json.loads((self.target/'project/config.json').read_text())
        key=(self.target/'project/train.py').as_posix()
        self.assertEqual(cfg['source_hashes'][key],hashlib.sha256((self.target/'project/train.py').read_bytes()).hexdigest())
        audit=json.loads((self.target/'WORKSPACE_PROVENANCE.json').read_text())
        self.assertEqual(audit['source_hash_updates'][0]['original'],'old')

    def test_external_reference_hash_matches_relocated_trainer(self):
        self.write('whu/train_complete.py', "ROOT='/root/autodl-tmp/whu'\n")
        self.write('whu/train/s43_observation/config.json',json.dumps({'source_sha256':'old','batch':2}))
        self.module.materialize(self.source,self.target)
        cfg=json.loads((self.target/'whu/train/s43_observation/config.json').read_text())
        self.assertEqual(cfg['source_sha256'],hashlib.sha256((self.target/'whu/train_complete.py').read_bytes()).hexdigest())
        self.assertEqual(cfg['batch'],2)

    def test_does_not_replace_similarly_named_unrelated_path(self):
        self.write('config.json',json.dumps({'other':'/root/autodl-tmp-other/data'}))
        self.module.materialize(self.source,self.target)
        self.assertEqual(json.loads((self.target/'config.json').read_text())['other'],'/root/autodl-tmp-other/data')


if __name__ == '__main__':
    unittest.main()
