# 24,000-step LoRA adapter

The public release `model-gen1-lora-b64-24000-20261021` contains the Orbax `adapter_state` and `provenance.json` from the batch-64, 24,000-update run. It is a diagnostic candidate only: the independent paired evaluation did not pass the routing gate, so deployment defaults to the original Myungkyu checkpoint.

The archive is `gen1_replay_expanded_lora_b64_20261015_step24000_adapter.tar`. Extract its contents into a checkpoint directory and verify every file against `step24000_adapter_manifest.json`. The base model and normalization stats still come from the fixed Hugging Face revision in `deploy/model_reference/`.
