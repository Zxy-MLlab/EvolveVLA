#!/usr/bin/env python3
"""Task routing with an isolated legacy harness argument compatibility shim.

The simulator now expects episode_index, which the public harness parser does
not provide. Supply the historical default None in this process; leave the
frozen harness source, original controller, and expert admission unchanged.
"""
import contextlib
import json
import os
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
import gen1_task_routing_20261020 as routing

OLD_SOURCES = routing.required_sources
OLD_COMMAND = routing.command_for


def required_sources():
    return OLD_SOURCES() + [Path(routing.__file__).resolve()]


def command_for(plan, output, seed, episode_steps=2000):
    command = OLD_COMMAND(plan, output, seed, episode_steps)
    if command[4] != str(SCRIPT):
        command[4] = str(SCRIPT)
        command.insert(5, '_runtime')
    return command


def load_runtime_receipt():
    path = Path(os.environ[routing.RECEIPT_ENV])
    if routing.digest(path) != os.environ[routing.RECEIPT_SHA_ENV]:
        raise ValueError('Routing receipt changed')
    receipt = routing.read_json(path)
    routing.verify_file_map(receipt['runtime_sources_sha256'])
    config_path = Path(receipt['config'])
    if routing.digest(config_path) != receipt['config_sha256']:
        raise ValueError('Routing configuration changed')
    route = routing.select_route(routing.read_json(config_path), receipt['task_id'])
    if (route['role'] != receipt['selected_role'] or route['controller_mode'] != receipt['controller_mode']
            or route['model'] != receipt['model_spec']):
        raise ValueError('Selected task/model/controller differs from the receipt')
    routing.verify_deployment_evidence(route)
    return receipt


def internal_runtime():
    receipt = load_runtime_receipt()
    sys.path[:0] = [str(SCRIPT.parent), str(routing.OPENPI/'src'),
                    str(routing.OPENPI/'packages/openpi-client/src')]
    import robotwin_public_openpi_harness as harness
    args = harness.parse_args()
    if not hasattr(args, 'episode_index'):
        args.episode_index = None
    if args.task != receipt['task_id'] or Path(args.robotwin_root).resolve() != routing.ROBOTWIN:
        raise ValueError('Runtime arguments do not match routing receipt')
    if args.role == 'sim':
        hook = (routing.simulator_controller_hook(receipt)
                if receipt['controller_mode'] != 'original' else contextlib.nullcontext())
        with hook:
            return harness.run_simulator(args)
    if Path(args.checkpoint).resolve() != Path(receipt['checkpoint']).resolve():
        raise ValueError('Runtime checkpoint differs from routing receipt')
    routing.verify_checkpoint(receipt['model_spec'])
    harness.__file__ = str(SCRIPT)
    return harness.run_parent(args)


def install():
    routing.SCRIPT = SCRIPT
    routing.required_sources = required_sources
    routing.command_for = command_for
    routing.load_runtime_receipt = load_runtime_receipt
    routing.internal_runtime = internal_runtime


if __name__ == '__main__':
    install()
    raise SystemExit(routing.main())
