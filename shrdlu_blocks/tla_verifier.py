"""TLA+ trace verification for SHRDLU AP traces against LTL properties."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from shrdlu_blocks.property_verifier import ACTIVE_PROPERTY_IDS

__all__ = ['build_tla_spec', 'run_tlc', 'verify_ap_trace']

_PROPERTIES_FILE = Path(__file__).resolve().parent.parent / 'SHRDLU_PROPERTIES_AST.json'
_TLA2TOOLS_SEARCH = [
    '/usr/local/lib/tla2tools.jar',
    '/usr/lib/tla2tools.jar',
    str(Path.home() / 'tla2tools.jar'),
    str(Path(__file__).resolve().parent.parent / 'tla2tools.jar'),
]


def _load_properties() -> List[Dict]:
    payload = json.loads(_PROPERTIES_FILE.read_text(encoding='utf-8'))
    return [
        prop
        for prop in payload['properties']
        if prop['id'] in ACTIVE_PROPERTY_IDS
    ]


def _ast_to_tla(node: Dict, ap_names: List[str]) -> str:
    t = node['type']
    if t == 'ap':
        name = node['name']
        if name not in ap_names:
            raise ValueError('Unknown AP in property AST: %s' % name)
        return name
    if t == 'not':
        return '~(%s)' % _ast_to_tla(node['operand'], ap_names)
    if t == 'and':
        parts = [_ast_to_tla(a, ap_names) for a in node['args']]
        return '(%s)' % ' /\\ '.join(parts)
    if t == 'or':
        parts = [_ast_to_tla(a, ap_names) for a in node['args']]
        return '(%s)' % ' \\/ '.join(parts)
    if t == 'implies':
        l = _ast_to_tla(node['left'], ap_names)
        r = _ast_to_tla(node['right'], ap_names)
        return '(%s => %s)' % (l, r)
    if t == 'globally':
        inner = node['operand']
        # G(A => X(P)) must be encoded as [][A => P']_vars in TLA+ (primes not allowed in []).
        if inner['type'] == 'implies' and inner['right']['type'] == 'next':
            antecedent = _ast_to_tla(inner['left'], ap_names)
            consequent = _ast_to_tla(inner['right']['operand'], ap_names)
            return "[][((%s) => %s')]_<<%s>>" % (antecedent, consequent, ', '.join(ap_names))
        return '[](%s)' % _ast_to_tla(inner, ap_names)
    if t == 'eventually':
        return '<>(%s)' % _ast_to_tla(node['operand'], ap_names)
    if t == 'next':
        raise ValueError('Bare next operator outside G(A => X(P)) is not supported by TLC')
    raise ValueError('Unsupported AST node type: %s' % t)


def build_tla_spec(
    ap_trace: List[Dict[str, bool]],
    ap_names: List[str],
    properties: Optional[List[Dict]] = None,
    module_name: str = 'ShrdluTrace',
) -> str:
    """Return a TLA+ module string encoding the trace and all LTL properties."""
    if not ap_trace:
        raise ValueError('ap_trace must be non-empty')
    if properties is None:
        properties = _load_properties()

    n = len(ap_trace)

    def bool_tla(v: bool) -> str:
        return 'TRUE' if v else 'FALSE'

    def state_conjunct(state: Dict[str, bool]) -> str:
        return ' /\\ '.join(
            '%s = %s' % (ap, bool_tla(state.get(ap, False)))
            for ap in ap_names
        )

    def primed_state_conjunct(state: Dict[str, bool]) -> str:
        return ' /\\ '.join(
            "%s' = %s" % (ap, bool_tla(state.get(ap, False)))
            for ap in ap_names
        )

    lines = []
    lines.append('---- MODULE %s ----' % module_name)
    lines.append('EXTENDS Naturals')
    lines.append('')
    lines.append('VARIABLES %s, step' % ', '.join(ap_names))
    lines.append('')

    # Named state predicates
    for i, state in enumerate(ap_trace):
        lines.append('State_%d == %s' % (i, state_conjunct(state)))
    lines.append('')

    # Init
    lines.append('Init ==')
    lines.append('  /\\ State_0')
    lines.append('  /\\ step = 0')
    lines.append('')

    # Next: one disjunct per transition, plus stuttering at last state
    lines.append('Next ==')
    for i in range(1, n):
        lines.append('  \\/ /\\ step = %d' % (i - 1))
        lines.append("     /\\ step' = %d" % i)
        lines.append('     /\\ %s' % primed_state_conjunct(ap_trace[i]))
    lines.append('  \\/ /\\ step = %d' % (n - 1))
    lines.append("     /\\ step' = step")
    for ap in ap_names:
        lines.append("     /\\ %s' = %s" % (ap, ap))
    lines.append('')

    all_vars = '<<step, %s>>' % ', '.join(ap_names)
    lines.append('Spec == Init /\\ [][Next]_%s' % all_vars)
    lines.append('')

    # Properties: temporal properties (no primes) as PROPERTY,
    # next-operator properties encoded as action invariants [][P => Q']_vars.
    for prop in properties:
        prop_id = prop['id'].replace('.', '_').replace('-', '_')
        tla_formula = _ast_to_tla(prop['ast'], ap_names)
        lines.append('(* %s: %s *)' % (prop['id'], prop.get('natural_language', '')))
        lines.append('Property_%s == %s' % (prop_id, tla_formula))
        lines.append('')

    lines.append('=' * 20)
    return '\n'.join(lines)


def build_tla_cfg(
    properties: Optional[List[Dict]] = None,
    module_name: str = 'ShrdluTrace',
) -> str:
    """Return a TLC .cfg file referencing all properties."""
    if properties is None:
        properties = _load_properties()
    lines = ['SPECIFICATION Spec', '']
    for prop in properties:
        prop_id = prop['id'].replace('.', '_').replace('-', '_')
        lines.append('PROPERTY Property_%s' % prop_id)
    return '\n'.join(lines)


def _find_tla2tools() -> Optional[str]:
    for path in _TLA2TOOLS_SEARCH:
        if os.path.isfile(path):
            return path
    env_path = os.environ.get('TLA2TOOLS_JAR')
    if env_path and os.path.isfile(env_path):
        return env_path
    return None


def run_tlc(
    tla_spec: str,
    cfg: str,
    module_name: str = 'ShrdluTrace',
    timeout: int = 60,
) -> Dict:
    """Run TLC on the given spec and cfg strings. Returns a result dict."""
    jar = _find_tla2tools()
    if jar is None:
        return {
            'success': False,
            'skipped': True,
            'reason': 'tla2tools.jar not found; set TLA2TOOLS_JAR env var or place it at %s'
                      % _TLA2TOOLS_SEARCH[-1],
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        tla_path = os.path.join(tmpdir, '%s.tla' % module_name)
        cfg_path = os.path.join(tmpdir, '%s.cfg' % module_name)
        with open(tla_path, 'w') as f:
            f.write(tla_spec)
        with open(cfg_path, 'w') as f:
            f.write(cfg)

        try:
            proc = subprocess.run(
                ['java', '-jar', jar, '-config', cfg_path, tla_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return {
                'success': False,
                'skipped': True,
                'reason': 'java not found in PATH',
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'skipped': False,
                'reason': 'TLC timed out after %ds' % timeout,
            }

        stdout = proc.stdout
        stderr = proc.stderr
        passed = 'No error has been found' in stdout
        violations = [
            line.strip() for line in stdout.splitlines()
            if line.strip().startswith('Error') or 'violated' in line.lower()
        ]
        return {
            'success': passed,
            'skipped': False,
            'returncode': proc.returncode,
            'stdout': stdout,
            'stderr': stderr,
            'violations': violations,
        }


def _has_eventually_consequence(ast: Dict) -> bool:
    """Return True if the property has the shape G(A => <>P) or G(<>P)."""
    if ast['type'] != 'globally':
        return False
    inner = ast['operand']
    if inner['type'] == 'eventually':
        return True
    if inner['type'] == 'implies' and inner['right']['type'] == 'eventually':
        return True
    return False


def verify_ap_trace(
    ap_trace: List[Dict[str, bool]],
    ap_names: List[str],
    properties: Optional[List[Dict]] = None,
    module_name: str = 'ShrdluTrace',
    timeout: int = 60,
    is_complete_trace: bool = True,
) -> Dict:
    """Build and run TLC on an AP trace. Returns combined result.

    When is_complete_trace=False (intermediate planning rollout), properties
    with an eventually-consequence (G(A => <>P)) are skipped — they cannot
    be satisfied by a partial trace that stutters at the final state.
    """
    if properties is None:
        properties = _load_properties()
    if not is_complete_trace:
        properties = [p for p in properties if not _has_eventually_consequence(p['ast'])]
    tla_spec = build_tla_spec(ap_trace, ap_names, properties, module_name)
    cfg = build_tla_cfg(properties, module_name)
    tlc_result = run_tlc(tla_spec, cfg, module_name, timeout)
    return {
        'tla_spec': tla_spec,
        'tla_cfg': cfg,
        'tlc_result': tlc_result,
        'trace_length': len(ap_trace),
        'properties_checked': [p['id'] for p in properties],
    }
