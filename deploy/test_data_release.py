import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from data_release import build_shard, restore_shard, safe_path


class DataReleaseTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        path = source / 'data/accepted/episode.hdf5'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'fixture source bytes\0' * 100)
        row = dict(path='data/accepted/episode.hdf5', bytes=path.stat().st_size,
                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return source, dict(name='fixture.tar', files=[row])

    def test_roundtrip_and_idempotent_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, package = self.fixture(root)
            archive = root / package['name']
            result = build_shard(source, package, archive)
            self.assertEqual(result['sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
            target = root / 'target'
            restore_shard(archive, package, target)
            restore_shard(archive, package, target)
            row = package['files'][0]
            self.assertEqual((source / row['path']).read_bytes(), (target / row['path']).read_bytes())

    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, package = self.fixture(root)
            path = source / package['files'][0]['path']
            path.write_bytes(b'x' * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                build_shard(source, package, root / package['name'])

    def test_existing_user_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, package = self.fixture(root)
            archive = root / package['name']
            build_shard(source, package, archive)
            target = root / 'target'
            file = target / package['files'][0]['path']
            file.parent.mkdir(parents=True)
            file.write_bytes(b'user data')
            with self.assertRaisesRegex(ValueError, 'overwrite'):
                restore_shard(archive, package, target)
            self.assertEqual(file.read_bytes(), b'user data')

    def test_traversal_and_symlink_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('../secret', '/data/file', 'data/../../secret', 'checkpoints/model'):
                with self.assertRaises(ValueError):
                    safe_path(root, name)
            (root / 'data').symlink_to('/tmp')
            with self.assertRaises(ValueError):
                safe_path(root, 'data/escape')

    def test_unlisted_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, package = self.fixture(root)
            archive = root / package['name']
            with tarfile.open(archive, 'w') as stream:
                item = tarfile.TarInfo('data/unlisted')
                item.size = 1
                stream.addfile(item, io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError, 'Unexpected archive member'):
                restore_shard(archive, package, root / 'target')


if __name__ == '__main__':
    unittest.main()
