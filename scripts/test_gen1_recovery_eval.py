"""CPU regression coverage for the resource calculation behind the aborted run."""
import tempfile
from pathlib import Path
import unittest
import gen1_recovery_eval as recovery


class MemoryAccountingTest(unittest.TestCase):
    def snapshot(self, inactive, dirty, writeback=0, shmem=0):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'memory.current').write_text('95')
            (root/'memory.max').write_text('100')
            (root/'memory.stat').write_text(
                f'inactive_file {inactive}\nactive_file 40\nfile_dirty {dirty}\n'
                f'file_writeback {writeback}\nshmem {shmem}\n')
            (root/'memory.events').write_text('oom 0\noom_kill 0\n')
            return recovery.memory_status(root)

    def test_counts_only_inactive_clean_cache(self):
        self.assertEqual(self.snapshot(20,3,2,1)['estimated_available'],19)

    def test_dirty_pages_cannot_create_negative_reclaimable(self):
        self.assertEqual(self.snapshot(2,5)['estimated_available'],5)

    def test_active_cache_is_not_discounted(self):
        self.assertEqual(self.snapshot(0,0)['estimated_available'],5)


if __name__=='__main__':unittest.main()
