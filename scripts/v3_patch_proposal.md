# V3 Harness Patch Proposal (UNAPPLIED)

This is a text-only proposal. It is not applied to
`scripts/robotwin_eval_harness.py`.

The anchors below were read from the current harness on 2026-10-02:

- import block: `scripts/robotwin_eval_harness.py:20-26`
- `load_policy`: `scripts/robotwin_eval_harness.py:141`
- parent per-episode print: `scripts/robotwin_eval_harness.py:371`
- parent result epilogue: `scripts/robotwin_eval_harness.py:384`
- simulator `chdir`: `scripts/robotwin_eval_harness.py:505`
- simulator protocol setup/import boundary: `scripts/robotwin_eval_harness.py:509-512`
- simulator per-episode step loop: `scripts/robotwin_eval_harness.py:593`
- simulator step increment: `scripts/robotwin_eval_harness.py:604`

## Hunk (a) - LOG-MTIME HEARTBEAT WATCHDOG

**Anchor file:line:** `scripts/robotwin_eval_harness.py:86-93` for the helper,
`scripts/robotwin_eval_harness.py:509-512` for log-path setup, and
`scripts/robotwin_eval_harness.py:593-604` for the simulator-loop call site.

The watched file is the run's own stdout/stderr log, for example
`/root/autodl-tmp/EvolveVLA/results/gate3_skip_expert_5ep.log`. The launcher
must pass its absolute path through `ROBOTWIN_HEARTBEAT_LOG`; the path must be
absolute because `run_simulator` changes directory at line 505. The child
inherits this environment through the existing `child_env = os.environ.copy()`.

```diff
--- a/scripts/robotwin_eval_harness.py
+++ b/scripts/robotwin_eval_harness.py
@@ -20,7 +20,8 @@
 import pickle
 import struct
 import subprocess
 import sys
+import time
 from pathlib import Path
 from typing import Any, BinaryIO
 
@@ -86,8 +87,28 @@
 def receive_message(stream: BinaryIO) -> Any:
     header = _read_exact(stream, 8)
     (size,) = struct.unpack("!Q", header)
     payload = _read_exact(stream, size)
     return pickle.loads(payload)
 
 
+def _check_log_heartbeat(
+    path: str | Path,
+    last_mtime: float | None,
+    stall_seconds: float = 120,
+) -> float:
+    log_path = Path(path)
+    try:
+        current_mtime = os.stat(log_path).st_mtime
+    except FileNotFoundError:
+        print(
+            f"WATCHDOG_TIMEOUT missing_log={log_path}",
+            file=sys.stderr,
+            flush=True,
+        )
+        raise SystemExit(124)
+    if last_mtime is None or current_mtime > last_mtime:
+        return current_mtime
+    age = time.time() - current_mtime
+    if age > stall_seconds:
+        print(
+            f"WATCHDOG_TIMEOUT log={log_path} age={age:.1f}s",
+            file=sys.stderr,
+            flush=True,
+        )
+        raise SystemExit(124)
+    return last_mtime
+
+
 def parse_args() -> argparse.Namespace:
```

```diff
--- a/scripts/robotwin_eval_harness.py
+++ b/scripts/robotwin_eval_harness.py
@@ -509,6 +509,13 @@
     protocol_fd = os.dup(1)
     os.dup2(2, 1)
     protocol_out = os.fdopen(protocol_fd, "wb")
+    heartbeat_log = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
+    if not heartbeat_log:
+        raise RuntimeError(
+            "ROBOTWIN_HEARTBEAT_LOG must point to the run's absolute stdout log"
+        )
+    log_path = Path(heartbeat_log)
+    if not log_path.is_absolute():
+        raise RuntimeError("ROBOTWIN_HEARTBEAT_LOG must be an absolute path")
+    last_log_mtime: float | None = None
     eval_module = importlib.import_module("eval_policy_xpolicylab")
```

```diff
--- a/scripts/robotwin_eval_harness.py
+++ b/scripts/robotwin_eval_harness.py
@@ -593,6 +593,9 @@
                     if command.get("cmd") != "action":
                         raise RuntimeError(f"unexpected parent command: {command}")
                     action = np.asarray(command["action"], dtype=np.float32).reshape(-1)
                     assert action.shape == (ACTION_DIM,), (
                         f"parent action must be 14-D, got {action.shape}"
                     )
                     task_env.take_action(action, action_type="qpos")
                     step += 1
+                    last_log_mtime = _check_log_heartbeat(
+                        log_path, last_log_mtime, stall_seconds=120
+                    )
                     is_success = bool(getattr(task_env, "eval_success", False))
```

`WATCHDOG_TIMEOUT` is the required machine-searchable failure marker. The
check is performed in the SIM child after each completed simulator step. A
stale or missing run log exits the child with status 124, which makes the
parent's existing non-zero child handling fail the run rather than treating a
live PID as proof of progress.

## Hunk (b) - RENDER BACKEND FIX (LAUNCHER ENVIRONMENT ONLY)

**Anchor file:line:** `scripts/robotwin_eval_harness.py:505` is the
`os.chdir(robotwin_root)` boundary, and `:512` is the first RoboTwin module
import. The fix is not a harness edit; it must be applied by the launcher
before the harness/SAPIEN import path starts.

The launcher prefix is:

```bash
bash -c 'export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; exec "$@"' -- <existing-managed-harness-command> <existing-args>
```

For the watchdog proposal, the same launcher should also pass the run log
path, without hardcoding it in the harness:

```bash
bash -c 'export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; export ROBOTWIN_HEARTBEAT_LOG=/root/autodl-tmp/EvolveVLA/results/gate3_skip_expert_5ep.log; exec "$@"' -- <existing-managed-harness-command> <existing-args>
```

This export must occur before the simulator child imports
`eval_policy_xpolicylab` at line 512, because that import reaches SAPIEN. The
root cause is that SAPIEN's `_vulkan_tricks._ensure_vulkan_icd` checks
`/usr/share/vulkan/icd.d/nvidia_icd.json`, while the actual ICD is at
`/etc/vulkan/icd.d/nvidia_icd.json`. Per the current code path,
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` is sufficient by itself
for `_ensure_vulkan_icd` to return early.

The separate `_ensure_libvulkan_linux` warning is still expected: the system
does not provide `libvulkan.so.1`, while SAPIEN bundles
`libvulkan.so.1.3.224`. The `VK_ICD_FILENAMES` export does not remove that
warning; it is separate from ICD selection and is not the Vulkan-ICD failure
described above.

## Hunk (c) - PER-N-STEP FD / VRAM INSTRUMENTATION

**Anchor file:line:** `scripts/robotwin_eval_harness.py:20-26` for the
`threading` import and `:604` immediately after `step += 1` in the SIM
child's step loop.

```diff
--- a/scripts/robotwin_eval_harness.py
+++ b/scripts/robotwin_eval_harness.py
@@ -20,7 +20,8 @@
 import argparse
 import importlib
 import os
 import pickle
 import struct
 import subprocess
 import sys
+import threading
 from pathlib import Path
 from typing import Any, BinaryIO
```

```diff
--- a/scripts/robotwin_eval_harness.py
+++ b/scripts/robotwin_eval_harness.py
@@ -603,6 +603,14 @@
                     task_env.take_action(action, action_type="qpos")
                     step += 1
+                    if step % 25 == 0:
+                        print(
+                            f"STEP_METRICS step={step} "
+                            f"fds={len(os.listdir('/proc/self/fd'))} "
+                            f"threads={threading.active_count()} "
+                            "vram_note=SIM_CHILD_RENDERER_EXTERNAL_NVIDIA_SMI",
+                            file=sys.stderr,
+                            flush=True,
+                        )
                     is_success = bool(getattr(task_env, "eval_success", False))
```

This instrumentation belongs in the SIM child process, not the policy parent:
the SIM child is the Python 3.10 process that owns SAPIEN and the renderer.
The existing `os.dup2(2, 1)` at line 510 sends native fd 1 output to stderr,
so explicitly printing to `sys.stderr` keeps these diagnostics in the mirrored
run log without corrupting the length-prefixed protocol stream. The
`vram_note` is deliberately a note rather than a fabricated byte count;
renderer VRAM attribution must be taken externally for the SIM child.
