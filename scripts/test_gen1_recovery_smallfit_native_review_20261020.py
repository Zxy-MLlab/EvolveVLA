"""Independent native-init/update CPU test with a tiny model, no3B/GPU load."""
import argparse
import dataclasses
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
import numpy as np
import gen1_recovery_smallfit_20261020 as smallfit


class NativeResumeReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):smallfit.environment(False)

    def test_native_init_preserves_trained_adapters_and_native_update_freezes_base(self):
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        import orbax.checkpoint as ocp
        from openpi.models import model as model_lib
        from openpi.training import sharding
        import gen1_replay_lora_train as lora
        class Tiny(model_lib.BaseModel):
            def __init__(self):
                super().__init__(14,50,0)
                self.kernel=nnx.Param(jnp.ones((2,2),jnp.float32))
                self.lora_a=nnx.Param(jnp.zeros((2,1),jnp.float32))
                self.lora_b=nnx.Param(jnp.zeros((1,2),jnp.float32))
            def compute_loss(self,rng,observation,actions,*,train=False):
                prediction=jnp.sum(self.kernel.value.astype(jnp.float32))+jnp.sum(self.lora_a.value@self.lora_b.value)
                return (prediction-jnp.mean(actions,axis=-1))**2
            def sample_actions(self,rng,observation,**kwargs):
                return jnp.zeros((observation.state.shape[0],50,14))
        spec=importlib.util.spec_from_file_location('independent_smallfit_native',smallfit.OPENPI/'scripts/train.py')
        native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            values=dict(kernel=np.array([[1,2],[3,4]],dtype=jnp.bfloat16),
                lora_a=np.array([[.5],[.25]],dtype=jnp.bfloat16),
                lora_b=np.array([[.125,.75]],dtype=jnp.bfloat16))
            with ocp.Checkpointer(ocp.PyTreeCheckpointHandler()) as saver:saver.save(root/'params',dict(params=values))
            loader=smallfit.StrictFullLoraLoader(root)
            cfg=smallfit.make_config(argparse.Namespace(checkpoint_dir=root/'train',batch_size=2),loader)
            cfg=dataclasses.replace(cfg,model=types.SimpleNamespace(create=lambda rng:Tiny()))
            mesh=sharding.make_mesh(1)
            state,_=native.init_train_state(cfg,jax.random.key(0),mesh,resume=False)
            jax.block_until_ready(state)
            actual={k:smallfit.hash_array(v) for k,v in lora.flat(state.params).items()}
            self.assertEqual(actual,loader.expected_hashes)
            np.testing.assert_array_equal(lora.flat(state.params)['lora_b'],values['lora_b'].astype(np.float32))
            self.assertEqual(lora.flat(state.params)['lora_b'].dtype,jnp.float32)
            self.assertEqual(lora.flat(state.params)['kernel'].dtype,jnp.bfloat16)
            frozen=lora.hashes(state.params);initial=lora.hashes(state.params,True)
            batch=(model_lib.Observation(images={},image_masks={},state=jnp.zeros((2,14),jnp.float32)),jnp.zeros((2,50,14),jnp.float32))
            for _ in range(3):state,info=native.train_step(cfg,jax.random.key(1),state,batch)
            jax.block_until_ready(state)
            self.assertEqual(lora.hashes(state.params),frozen)
            self.assertNotEqual(lora.hashes(state.params,True),initial)
            self.assertEqual(int(state.step),3)
            self.assertTrue(np.isfinite(float(info['loss'])))
            self.assertTrue(all(device.platform=='cpu' for device in jax.devices()))


if __name__=='__main__':unittest.main()
