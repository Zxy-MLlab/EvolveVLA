# pi0.5 RoboCasa Measurement Gate

## Status

**BLOCKED: C007 is NOT cleared.** As of 2026-09-30 19:56:08 UTC, the
`google/paligemma-3b-pt-224` tokenizer cache is absent and the tokenizer source
is undecided pending the user's A/B decision:

- A: accept the Gemma license and provide an HF token for the official tokenizer.
- B: explicitly approve the ungated `unsloth/gemma-2b` surrogate, then verify
  token-id equivalence before measurement.

The launchers now set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, verify
tokenizer provenance locally, print the `C007 GATE` source header, and exit
non-zero before evaluation or training if the tokenizer is absent or unverified.
Firing them now is expected to refuse loudly. No surrogate has been wired in.

## Guarded launchers

```bash
bash /root/autodl-tmp/EvolveVLA/run_pi05_competence.sh
bash /root/autodl-tmp/EvolveVLA/run_pi05_base_competence.sh
bash /root/autodl-tmp/EvolveVLA/run_lora_stability_gate.sh
```

These commands are not ready-to-fire until C007 is cleared. The guard is an
intentional preflight gate and must remain in place for every measurement path.
