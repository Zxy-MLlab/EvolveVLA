"""CPU tests for real suffix windows, native transforms and typed Orbax restore."""
import argparse
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import gen1_recovery_smallfit_20261020 as smallfit
from test_gen1_recovery_suffix_dataset_20261020 import make_case


class SmallfitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        smallfit.environment(False)

    def test_ready_dataset_windows_are_next_capture_targets_and_chw_rgb(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=root/'raw'
            _,states=make_case(inputs)
            smallfit.suffix.process(inputs,root/'built','original','build')
            data=smallfit.SuffixDataset(root/'built/manifest.json')
            raw=data.raw(0,2)
            np.testing.assert_array_equal(raw['observation.state'],states[2])
            np.testing.assert_array_equal(raw['action'],states[3:53])
            self.assertEqual(raw['observation.images.cam_high'].shape,(3,3,4))
            self.assertTrue(np.all(raw['observation.images.cam_high']==2))
            with self.assertRaises(IndexError):
                data.raw(0,3)
            for _ in range(10):
                episode,start=data.draw(random.Random(_))
                self.assertEqual(episode,0)
                self.assertLess(start,3)

    def test_native_14_to_32_delta_quantile_roundtrip_on_suffix_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            inputs=root/'raw'
            make_case(inputs)
            smallfit.suffix.process(inputs,root/'built','original','build')
            data=smallfit.SuffixDataset(root/'built/manifest.json')
            args=argparse.Namespace(checkpoint_dir=smallfit.ROOT/'checkpoints/cpu_smallfit_test',batch_size=8)
            cfg=smallfit.make_config(args,smallfit.StrictFullLoraLoader(smallfit.INITIAL))
            forward,inverse=smallfit.transformations(cfg)
            anchors=smallfit.anchors_and_preflight(data,forward,inverse)
            self.assertEqual(len(anchors),3)
            self.assertEqual(anchors[0]['transformed']['actions'].shape,(50,32))
            self.assertEqual(cfg.model.action_horizon,50)
            self.assertFalse(cfg.data.adapt_to_pi)
            self.assertTrue(cfg.data.use_delta_joint_actions)

    def test_typed_full_restore_preserves_nonzero_b_and_declared_dtypes(self):
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            values=dict(backbone=dict(kernel=np.array([1,2],dtype=jnp.bfloat16)),
                adapter=dict(lora_a=np.array([[.5,.25]],dtype=jnp.bfloat16),
                             lora_b=np.array([[.125],[.75]],dtype=jnp.bfloat16)))
            template=dict(backbone=dict(kernel=jax.ShapeDtypeStruct((2,),jnp.bfloat16)),
                adapter=dict(lora_a=jax.ShapeDtypeStruct((1,2),jnp.float32),
                             lora_b=jax.ShapeDtypeStruct((2,1),jnp.float32)))
            with ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) as saver:
                saver.save(root/'params',dict(params=values))
            loader=smallfit.StrictFullLoraLoader(root)
            restored=loader.load(template)
            np.testing.assert_array_equal(restored['adapter']['lora_b'],np.asarray(values['adapter']['lora_b'],dtype=np.float32))
            self.assertEqual(restored['adapter']['lora_b'].dtype,np.float32)
            self.assertEqual(restored['backbone']['kernel'].dtype,jnp.bfloat16)
            self.assertFalse(loader.report['reset_lora_b'])
            self.assertTrue(loader.report['typed_restore'])
            self.assertEqual(loader.report['nonzero_lora_b_leaves'],1)
            template['adapter']['missing_lora_b']=jax.ShapeDtypeStruct((1,),jnp.float32)
            with self.assertRaisesRegex(ValueError,'exact keys'):
                loader.load(template)

    def test_unready_manifest_is_rejected_before_initializer_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json'
            p.write_text('{"format":"recovery_suffix_npz_v1","dataset_ready":false}')
            with patch.object(smallfit.StrictFullLoraLoader,'load') as restore:
                with self.assertRaises(ValueError):
                    smallfit.SuffixDataset(p)
                restore.assert_not_called()

    def test_manifest_cannot_relabel_source_split_or_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_case(root/'raw')
            smallfit.suffix.process(root/'raw',root/'built','original','build')
            path=root/'built/manifest.json'
            original=path.read_text()
            manifest=json.loads(original)
            group=manifest['episodes'][0]['parent_episode_uid']
            manifest['source_episode_splits'][group]='val'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'split mapping'):
                smallfit.SuffixDataset(path)
            manifest=json.loads(original)
            manifest['episodes'][0]['prompt']='An unrelated task instruction'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'upstream verification'):
                smallfit.SuffixDataset(path)


if __name__=='__main__':
    unittest.main()
