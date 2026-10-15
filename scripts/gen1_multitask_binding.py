"""CPU-only scene-aware seen-prompt binding for future broad Clean checks.

Uses the official field eligibility and replacement semantics, with local hash
selection instead of consuming simulator RNG. Not installed in any active job.
"""
import hashlib
import json
from pathlib import Path
import re
import string


def choose(task, seed, field, count):
    if type(seed) is not int or seed < 0 or count < 1:
        raise ValueError('Invalid deterministic selection')
    value = hashlib.sha256(f'gen1-multitask-seen-v1:{task}:{seed}:{field}'.encode()).digest()
    return int.from_bytes(value, 'big') % count


def placeholders(template):
    result = set()
    for _, name, spec, conversion in string.Formatter().parse(template):
        if name is not None:
            if not re.fullmatch('[A-Za-z]', name) or spec or conversion:
                raise ValueError('Unsupported official placeholder')
            result.add(name)
    return result


def eligible(template, fields):
    names = placeholders(template)
    arms = {k for k in fields if len(k) == 1 and 'a' <= k <= 'z'}
    return names == fields or bool(arms and names | arms == fields and not names & arms)


def read_pool(path):
    data = path.read_bytes()
    pool = json.loads(data)['seen']
    if not isinstance(pool, list) or not pool or not all(isinstance(s, str) and s for s in pool):
        raise ValueError('Invalid official seen pool')
    return pool, hashlib.sha256(data).hexdigest()


def bind(task, seed, episode_info, description_root):
    if not re.fullmatch('[a-z][a-z0-9_]*', task):
        raise ValueError('Unsafe task')
    params = episode_info.get('info')
    if not isinstance(params, dict):
        raise ValueError('Missing expert fields')
    fields = {}
    for key, value in params.items():
        if not re.fullmatch(r'{[A-Za-z]}', key) or not isinstance(value, str) or not value or '{' in value or '}' in value:
            raise ValueError('Invalid expert field')
        fields[key[1:-1]] = value
    root = Path(description_root).resolve()
    source = root / 'task_instruction' / (task + '.json')
    pool, sha = read_pool(source)
    indices = [i for i, s in enumerate(pool) if eligible(s, set(fields))]
    if not indices:
        raise ValueError('No eligible official template; no fallback')
    index = indices[choose(task, seed, 'template', len(indices))]
    template = pool[index]
    substitutions, objects = {}, {}
    object_root = root / 'objects_description'
    for key, value in sorted(fields.items()):
        path = (object_root / (value + '.json')).resolve()
        if '\\' in value or not path.is_relative_to(object_root):
            raise ValueError('Object path escapes description root')
        if path.is_file():
            descriptions, obj_sha = read_pool(path)
            selected = choose(task, seed, key, len(descriptions))
            substitutions[key] = 'the ' + descriptions[selected]
            objects[key] = dict(source=str(path), sha256=obj_sha, index=selected)
        elif '/' in value:
            raise ValueError('Missing official object description')
        elif key.islower():
            if value not in ('left', 'right', 'dual'):
                raise ValueError('Invalid expert arm')
            substitutions[key] = f'the {value} arm'
        else:
            substitutions[key] = value
    prompt = template.format_map(substitutions)
    if '{' in prompt or '}' in prompt:
        raise ValueError('Unbound placeholder')
    return dict(task=task, seed=seed, prompt=prompt, fields=dict(params),
                source=str(source), source_sha256=sha, index=index, template=template,
                eligible_indices=indices, objects=objects,
                rule='Official field eligibility; hash selection; expert fields only; no simulator RNG consumed')
