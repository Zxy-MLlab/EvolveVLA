#!/usr/bin/env python3
"""Process-local qpos duration interventions; the RoboTwin source stays read-only.

The original TOPP arrays emit one command per advancement of that arm. Treat
their positions as local planning-clock boundaries 1..N and prepend the current joint command at
boundary 0. Resample at k*N/M (k=1..M), and sample TOPP velocity at the same
coordinates, multiplying it by N/M. These are executed-command coordinates,
not TOPP's returned timestamps (which the original controller ignores). The
original controller interleaves both arms over max(N_left,N_right) scene ticks,
so a shorter arm's local planning clock differs from its actual emission times.

fixed15 changes every qpos action to 15 ticks for BOTH arms. opening15 does so
only during a command-driven opening window. A normal TOPP path is resampled;
a failed TOPP keeps its false flag and never gains an arm setter call. The
original gripper interpolation, control loop and success early return remain.
"""
import ast
import copy
import hashlib
from pathlib import Path
import types

import numpy as np

MODES = ('original', 'fixed15', 'opening15')
CLOSED = .2
OPEN = .8
START_DELTA = .01
START_TARGET = .05
MONOTONIC_EPS = 1e-6


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OpeningGate:
    """State for one arm, using drive targets and model commands only.

    The initial observed drive target may establish closure. After a consumed
    opening or reversal, BOTH a new closed command and an observed closed
    drive target are required to rearm. No object/contact/source-index inputs.
    """
    def __init__(self):
        self.armed = False
        self.active = False
        self.require_new_close = False
        self.previous_target = None

    def observe(self, current, target):
        current, raw_target = float(current), float(target)
        if not np.isfinite([current, raw_target]).all():
            raise ValueError('Nonfinite gripper gate input')
        target = float(np.clip(raw_target, 0., 1.))
        before = dict(armed=self.armed, active=self.active,
                      require_new_close=self.require_new_close,
                      previous_target=self.previous_target)
        event, enabled = 'inactive', False
        if self.active:
            monotonic = self.previous_target is None or target >= self.previous_target - MONOTONIC_EPS
            if not monotonic or target < current - MONOTONIC_EPS:
                self.active = self.armed = False
                self.require_new_close = True
                event = 'opening_reversed'
            else:
                enabled, event = True, 'opening_continues'
        if not self.active and not enabled:
            # A newly closed command cannot immediately start an opening. A
            # later call sees its resulting drive target, then may open.
            observed_closed = current <= CLOSED
            closed_command = target <= CLOSED
            if observed_closed and (not self.require_new_close or closed_command):
                self.armed = True
                if closed_command:
                    self.require_new_close = False
            monotonic_start = self.previous_target is None or target >= self.previous_target - MONOTONIC_EPS
            if (self.armed and event != 'opening_reversed' and monotonic_start
                    and target - current >= START_DELTA and target > START_TARGET):
                self.active = True
                enabled, event = True, 'opening_started'
        if enabled and target >= OPEN:
            self.active = self.armed = False
            self.require_new_close = True
            event = 'opening_completed_including_this_action'
        self.previous_target = target
        return dict(enabled=enabled, event=event, drive_target=current,
                    raw_model_target=raw_target, model_target=target,
                    target_minus_drive=target-current, before=before,
                    after=dict(armed=self.armed, active=self.active,
                               require_new_close=self.require_new_close))


def resample_path(result, current, old_ticks, new_ticks, timestep):
    """Retain endpoint and path ordering; scale velocities with executed time.

    Supplied TOPP velocities are interpolated independently of position. Their
    time scaling is consistent, but they need not equal the derivative of the
    piecewise linear position interpolation. No extra zeroing or clipping is
    performed, and compression can exceed the original planner velocity limit.
    """
    position, velocity = np.asarray(result['position']), np.asarray(result['velocity'])
    current = np.asarray(current)
    if (position.ndim != 2 or position.shape != velocity.shape
            or position.shape[0] != old_ticks or current.shape != position.shape[1:]
            or old_ticks < 1 or new_ticks < 1 or timestep <= 0
            or not all(np.isfinite(a).all() for a in (position, velocity, current))):
        raise ValueError('Invalid TOPP arrays or resampling duration')
    ratio = old_ticks / new_ticks
    old_boundaries = np.arange(old_ticks + 1, dtype=float)
    sample_at = np.arange(1, new_ticks + 1, dtype=float) * ratio
    sample_at[-1] = old_ticks
    pos_boundaries = np.vstack((current, position))
    # The old controller supplies no initial measured joint velocity. Extend
    # its first commanded velocity to boundary zero; do not invent zero speed.
    vel_boundaries = np.vstack((velocity[0], velocity))
    resample = lambda values: np.column_stack([
        np.interp(sample_at, old_boundaries, values[:, j])
        for j in range(values.shape[1])])
    new_position = resample(pos_boundaries)
    new_velocity = resample(vel_boundaries) * ratio
    new_position[-1] = position[-1]
    changed = dict(result, position=new_position, velocity=new_velocity)
    finite_difference = np.diff(np.vstack((current, new_position)), axis=0) / timestep
    return changed, dict(old_ticks=old_ticks, new_ticks=new_ticks, velocity_scale=ratio,
        old_execution_seconds=old_ticks*timestep, new_execution_seconds=new_ticks*timestep,
        position_endpoint_max_abs=float(np.max(np.abs(new_position[-1]-position[-1]))),
        original_max_abs_command_velocity=float(np.max(np.abs(velocity))),
        new_max_abs_command_velocity=float(np.max(np.abs(new_velocity))),
        new_max_abs_position_difference_velocity=float(np.max(np.abs(finite_difference))),
        max_abs_first_command_shift=float(np.max(np.abs(new_position[0]-position[0]))),
        all_finite=bool(np.isfinite(new_position).all() and np.isfinite(new_velocity).all()))


def patch_function(source):
    """Insert one qpos hook; no original AST node or branch is rewritten."""
    source = Path(source).resolve()
    tree = ast.parse(source.read_text(), filename=str(source))
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'take_action']
    if len(candidates) != 1:
        raise ValueError('Expected exactly one take_action')
    original = candidates[0]
    modified = copy.deepcopy(original)
    branches = [n for n in modified.body if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
        and n.test.left.id == 'action_type' and len(n.test.ops) == 1
        and isinstance(n.test.ops[0], ast.Eq) and len(n.test.comparators) == 1
        and isinstance(n.test.comparators[0], ast.Constant)
        and n.test.comparators[0].value == 'qpos']
    if len(branches) != 1:
        raise ValueError('Expected exactly one qpos branch')
    branch = branches[0]
    hook = ast.parse('left_result, right_result, left_n_step, right_n_step = '
        '__gen1_duration_apply(self, left_result if topp_left_flag else None, '
        'right_result if topp_right_flag else None, left_n_step, right_n_step, '
        'topp_left_flag, topp_right_flag, left_current_qpos, right_current_qpos, '
        'left_gripper_actions[0], right_gripper_actions[0])').body[0]
    # Place the hook on the blank line after the qpos branch so passive source
    # tracing cannot mistake it for one of the four fallback assignments.
    hook_line = branch.body[-1].end_lineno + 1
    ast.increment_lineno(hook, hook_line - 1)
    branch.body.append(hook)
    restored = copy.deepcopy(modified)
    restored_branch = next(n for n in restored.body
                           if isinstance(n, ast.If) and n.lineno == branch.lineno)
    restored_branch.body.pop()
    if ast.dump(restored, include_attributes=True) != ast.dump(original, include_attributes=True):
        raise ValueError('Modification exceeded the single qpos hook')
    return modified, dict(source=str(source), source_sha256=_sha(source), hook_line=hook_line,
        only_one_qpos_hook_inserted=True, original_nodes_preserved=True,
        compilation_preserves_source_filename_and_original_line_numbers=True)


class DurationController:
    def __init__(self, original_take, mode, ticks=15):
        if mode not in MODES or ticks != 15:
            raise ValueError('Supported modes are original/fixed15/opening15 with ticks=15')
        if not isinstance(original_take, types.MethodType):
            raise TypeError('A bound original take_action method is required')
        self.mode, self.ticks = mode, ticks
        self.gates = {arm: OpeningGate() for arm in ('left', 'right')}
        self.last_record = None
        source = Path(original_take.__func__.__code__.co_filename).resolve()
        function, patch = patch_function(source)
        namespace = dict(original_take.__func__.__globals__)
        namespace['__gen1_duration_apply'] = self._apply
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        self._method = types.MethodType(namespace['take_action'], original_take.__self__)
        self.metadata = dict(mode=mode, requested_ticks=ticks, patch=patch,
            controller_source_sha256=_sha(__file__), qpos_only=True,
            timing_coordinate='emitted command boundaries: current qpos at0, old positions at1..N; sample k*N/15',
            velocity_rule='interpolate same boundary coordinates; multiply by old_N/15; no clipping or extra zeroing',
            initial_velocity_rule='extend first TOPP commanded velocity to boundary0; no measured velocity added',
            fallback_rule='both counts15 when active; preserve false TOPP flag and no arm setter',
            synchronized_both_arms=True, gripper_interpolation_unchanged=True,
            early_success_unchanged=True, normal_topp_returned_times_ignored_as_in_original=True,
            two_arm_clock_semantics='original shorter-arm commands are interleaved over max(N_left,N_right) scene ticks; N_arm/15 rescales the local planning clock, not the full original physical timeline',
            opening_gate=dict(current='robot.get_normal_real_gripper_val reads normalized drive targets, not physical aperture',
                target='clip current model gripper target to0..1 as original setter does',
                closed_max=CLOSED, start_delta_min=START_DELTA, start_target_strict_min=START_TARGET,
                monotonic_epsilon=MONOTONIC_EPS, terminal_target_min=OPEN,
                end='include first target>=.8; reversal ends before action; consumed gate needs new closed command and drive',
                activation='either arm enables duration change for both arms',
                no_source_indices_object_state_or_expert_labels=True),
            limitations=['The gate thresholds are fixed heuristics, not a contact detector.',
                'Retiming changes both arm command speeds and gripper duration; it is not gripper-only.',
                'Short-arm local TOPP array time differs from original interleaved scene time; N_arm/15 is not complete physical timing equivalence.',
                'Compression may exceed original TOPP velocity limits; values are recorded and not silently clipped.',
                'Interpolated velocity is not forced to equal finite differences of resampled positions.',
                'The source capture cadence varies; fixed15 is an explicit ablation, not exact dense replay.'])

    def take_action(self, action, *args, **kwargs):
        self.last_record = dict(mode=self.mode, qpos_hook_reached=False, active=False)
        return self._method(action, *args, **kwargs)

    def _apply(self, task, left, right, nl, nr, fl, fr, ql, qr, gl, gr):
        gates = {}
        if self.mode == 'opening15':
            current = task.robot.get_normal_real_gripper_val()
            if len(current) != 2:
                raise ValueError('Expected two normalized gripper drive targets')
            gates = {arm: self.gates[arm].observe(c, target)
                for arm, c, target in zip(('left', 'right'), current, (gl, gr), strict=True)}
        selected = [arm for arm, gate in gates.items() if gate['enabled']]
        active = self.mode == 'fixed15' or bool(selected)
        dt = float(task.scene.get_timestep())
        arms, results = {}, []
        for arm, result, count, flag, current in zip(
                ('left', 'right'), (left, right), (nl, nr), (fl, fr), (ql, qr), strict=True):
            row = dict(topp_success=bool(flag), original_ticks=int(count),
                       scheduled_ticks=self.ticks if active else int(count),
                       retimed=active, arm_path_resampled=bool(active and flag),
                       arm_setter_preserved=bool(flag))
            if active and flag:
                result, metrics = resample_path(result, current, int(count), self.ticks, dt)
                row.update(metrics)
            arms[arm] = row
            results.append(result)
        self.last_record = dict(mode=self.mode, qpos_hook_reached=True, active=active,
            gate_selected_arms=selected, gates=gates, arms=arms,
            both_arm_durations_changed_together=active,
            original_scheduled_physics_ticks=max(int(nl), int(nr)),
            scheduled_physics_ticks=self.ticks if active else max(int(nl), int(nr)),
            timestep=dt, actual_ticks_require_passive_scene_counter=True)
        return results[0], results[1], self.ticks if active else nl, self.ticks if active else nr


def create(original_take, mode, ticks=15):
    """Return (bound callable, JSON-compatible metadata, controller)."""
    controller = DurationController(original_take, mode, ticks)
    return controller.take_action, controller.metadata, controller
