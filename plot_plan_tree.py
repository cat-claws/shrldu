#!/usr/bin/env python3
"""
Render a SHRDLU agent trace as an HTML page with an inline SVG plan tree.

Graph semantics:
  - Each DOT node = one action proposed by the planner (one attempt)
  - Color: green = TLC PASS, red = TLC FAIL (property violated)
  - Multiple attempts at the same tree state are siblings (same parent state node)
  - Accepted action continues to the next state; failed actions are dead-end leaves
  - Edge label: generation sequence index (child_index)
  - Node label: action + AP state (true predicates only)
  - Dead-leaf attempts from child_failure (backtrack branches) are also plotted

Usage:
  python plot_plan_tree.py  <trace.json>  [out.html]
"""

import json
import re
import subprocess
import sys
from pathlib import Path

PASS_FILL  = "#2ecc71"
FAIL_FILL  = "#e74c3c"
ERR_FILL   = "#e67e22"   # planning_error dead-end
ROOT_FILL  = "#2980b9"
WHITE      = "white"
GREY_BG    = "#f8f9fa"

ACTION_ABBREV = {
    "move_grasper":  "move",
    "lower_grasper": "lower",
    "raise_grasper": "raise",
    "close_grasper": "close",
    "open_grasper":  "open",
}
_PROP_RE = re.compile(r"Property_prop_(\w+)")


# ── helpers ────────────────────────────────────────────────────────────────────

def wrap(s: str, width: int = 20) -> str:
    s = s.replace("_", " ")
    words, lines, cur = s.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\\n".join(lines)


def fmt_action(action: dict) -> str:
    if not action:
        return "?"
    name  = action.get("name", "?")
    short = ACTION_ABBREV.get(name, name)
    args  = action.get("args") or {}
    if args:
        return short + "(" + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"
    return short


def ap_true_lines(ap_state: dict) -> str:
    """Only true predicates, abbreviated, as dot \\n-joined string."""
    true_keys = [k for k, v in ap_state.items() if v]
    if not true_keys:
        return "(none)"
    out = []
    for k in true_keys:
        s = (k
             .replace("some_object_resting_on_", "∃on_")
             .replace("_resting_on_", "→")
             .replace("object_", "o")
             .replace("grasper_", "g_"))
        out.append(s)
    return "\\n".join(out)


def violation_props(attempt: dict) -> str:
    fail = (attempt.get("step_verification") or {}).get("failure") or {}
    if fail.get("type") == "tla_property_violation":
        viols = fail.get("violations", [])
        if viols:
            props = _PROP_RE.findall(viols[0])
            return " & ".join(props) if props else "prop violated"
    return ""


def _viol_props_from_failure(f: dict) -> str:
    """Extract violation property names from a child_failure entry."""
    viols = f.get("violations", [])
    if viols:
        props = _PROP_RE.findall(viols[0])
        return " & ".join(props) if props else "prop violated"
    return ""


def collect_dead_leaves(tree_nodes: list) -> dict:
    """Return {node_id: [(dot_id, dot_label, fill, estyle, ecolor, pw), ...]}
    for every dead-leaf attempt found inside child_failure trees.

    Each branch_exhausted entry carries a node_id that matches a real state
    node.  Its failed_attempts are the direct attempts at that state — we draw
    tla_property_violation and planning_error ones as red/orange dead-end
    nodes.  We deduplicate by (node_id, type, action) so the same failure
    isn't added multiple times from different ancestor child_failure chains.
    """
    seq: dict[int, int] = {}
    seen: set[tuple] = set()
    result: dict[int, list] = {}

    def process_branch(be: dict):
        nid = be.get("node_id")
        if nid is None:
            return
        for fa in be.get("failed_attempts", []):
            ftype = fa.get("type", "")
            if ftype == "branch_exhausted":
                process_branch(fa)
            elif ftype in ("tla_property_violation", "planning_error"):
                action = fa.get("action") or {}
                aname  = action.get("name", "") if isinstance(action, dict) else ""
                aargs  = action.get("args", {}) if isinstance(action, dict) else {}
                key    = (nid, ftype, aname, str(sorted((aargs or {}).items())))
                if key in seen:
                    continue
                seen.add(key)
                idx = seq.get(nid, 0)
                seq[nid] = idx + 1

                dot_id = f"dead_{nid}_{idx}"
                if ftype == "tla_property_violation":
                    viol     = _viol_props_from_failure(fa)
                    act_lbl  = wrap(fmt_action(action))
                    viol_txt = dot_esc(f"✗ {viol}") if viol else ""
                    parts    = [dot_esc(act_lbl)]
                    if viol_txt:
                        parts.append(viol_txt)
                    full_lbl = "\\n".join(parts)
                    fill     = FAIL_FILL
                    estyle   = "filled,dashed"
                    ecolor   = "#c0392b"
                    pw       = "1.4"
                else:  # planning_error
                    msg      = fa.get("message", "plan error")[:40]
                    full_lbl = dot_esc(f"plan_err\\n{msg}")
                    fill     = ERR_FILL
                    estyle   = "filled,dashed"
                    ecolor   = "#d35400"
                    pw       = "1.4"

                result.setdefault(nid, []).append(
                    (dot_id, full_lbl, fill, estyle, ecolor, pw)
                )
            # max_tries: budget-exceeded noise, skip

    for n in tree_nodes:
        for att in n.get("attempts", []):
            cf = att.get("child_failure")
            if cf and cf.get("type") == "branch_exhausted":
                process_branch(cf)

    return result


def dot_esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ── DOT builder ────────────────────────────────────────────────────────────────

def build_dot(data: dict) -> str:
    request   = data.get("request", "")
    model     = data.get("model", "")
    status    = data.get("status", "")
    timestamp = data.get("timestamp_utc", "")[:19].replace("T", " ")
    tree_nodes = data["planning_tree"]["nodes"]

    title = dot_esc(f'{request}  [{model}  {timestamp} UTC  status:{status}]')

    lines = [
        'digraph PlanTree {',
        f'  graph [label="{title}" labelloc=t fontsize=11 fontname=Helvetica',
        f'         rankdir=TB splines=ortho nodesep=0.5 ranksep=0.6 bgcolor="{GREY_BG}"];',
        f'  node  [fontname=Helvetica fontsize=9 style=filled shape=box margin="0.15,0.08"];',
        f'  edge  [fontname=Helvetica fontsize=8];',
        # ROOT = initial state (node 0, no accepted steps)
        f'  s0 [label="ROOT" shape=ellipse fillcolor="{ROOT_FILL}" fontcolor=white fontsize=10];',
    ]

    # Build a lookup: tree_node_id → node data
    by_id = {n["node_id"]: n for n in tree_nodes}

    # Pre-collect dead-leaf attempts from child_failure trees
    dead_leaves = collect_dead_leaves(tree_nodes)

    # We emit two kinds of DOT nodes:
    #   s{tree_node_id}  — state node (invisible, just a routing point); we reuse ROOT for s0
    #   a{tree_node_id}_{child_index} — action node (one per attempt)
    #
    # Edges:
    #   s{parent} -> a{node}_{child_index}   [label="#{child_index}"]
    #   a{node}_{child_index} -> s{node}     (only for the accepted/passing attempt
    #                                          that leads to the child state)

    # State nodes (intermediate, minimal box — just shows the AP state)
    for n in tree_nodes:
        nid = n["node_id"]
        if nid == 0:
            continue  # ROOT already emitted
        ap = n.get("current_ap_state", {})
        ap_txt = dot_esc(ap_true_lines(ap))
        lines.append(
            f'  s{nid} [label="{ap_txt}" fillcolor="#ecf0f1" fontcolor="#2c3e50"'
            f' color="#bdc3c7" penwidth=1 fontsize=7 shape=box style="filled,rounded"];'
        )

    # Action nodes (one per attempt)
    for n in tree_nodes:
        nid      = n["node_id"]
        parent_id = n.get("parent_node_id")
        attempts  = n.get("attempts", [])

        # The action that was actually accepted for this node =
        # the last entry of accepted_steps (if any), recorded on THIS node.
        # There may be NO attempts if the planner went straight through.
        accepted_steps = n.get("accepted_steps", [])
        this_step = accepted_steps[-1] if accepted_steps else None

        if nid == 0:
            # Root has no incoming action
            continue

        # The parent state
        p_state = f"s{parent_id}" if parent_id != 0 else "s0"

        if not attempts:
            # No retries — action was accepted without a failed attempt recorded.
            # Draw a single green action node for the accepted step.
            action   = this_step["action"] if this_step else {}
            ap_after = this_step.get("ap_state_after", {}) if this_step else {}
            act_lbl  = wrap(fmt_action(action))
            ap_txt   = dot_esc(ap_true_lines(ap_after))
            full_lbl = dot_esc(act_lbl) + "\\n—\\n" + ap_txt
            dot_id   = f"a{nid}_acc"
            lines.append(
                f'  {dot_id} [label="{full_lbl}" fillcolor="{PASS_FILL}" fontcolor="{WHITE}"];'
            )
            lines.append(f'  {p_state} -> {dot_id} [color="#27ae60" penwidth=2];')
            lines.append(f'  {dot_id} -> s{nid} [color="#27ae60" penwidth=2];')
        else:
            # One or more attempts recorded at the parent state before arriving here.
            for att in attempts:
                ci       = att.get("child_index", 0)
                action   = att.get("action", {})
                passed   = att.get("step_verification", {}).get("passed", False)
                ap_after = att.get("step_verification", {}).get("predicted_ap_state") or {}
                viol     = violation_props(att)

                act_lbl  = wrap(fmt_action(action))
                ap_txt   = dot_esc(ap_true_lines(ap_after))
                viol_txt = dot_esc(f"✗ {viol}") if viol else ""

                parts = [dot_esc(act_lbl), "—", ap_txt]
                if viol_txt:
                    parts.insert(1, viol_txt)
                full_lbl = "\\n".join(parts)

                dot_id = f"a{nid}_{ci}"

                if passed:
                    fill   = PASS_FILL
                    estyle = "filled"
                    ecolor = "#27ae60"
                    pw     = "2"
                else:
                    fill   = FAIL_FILL
                    estyle = "filled,dashed"
                    ecolor = "#c0392b"
                    pw     = "1.4"

                lines.append(
                    f'  {dot_id} [label="{full_lbl}" fillcolor="{fill}" fontcolor="{WHITE}" style="{estyle}"];'
                )
                # edge from parent state → action node, labelled with generation index
                lines.append(
                    f'  {p_state} -> {dot_id} [label="#{ci}" color="{ecolor}" penwidth={pw}];'
                )
                # if passed → edge from action node → next state
                if passed:
                    lines.append(
                        f'  {dot_id} -> s{nid} [color="{ecolor}" penwidth={pw}];'
                    )

    # Dead-leaf attempts from child_failure (TLC violations & planning errors)
    for nid, leaves in dead_leaves.items():
        p_state = "s0" if nid == 0 else f"s{nid}"
        for (dot_id, full_lbl, fill, estyle, ecolor, pw) in leaves:
            lines.append(
                f'  {dot_id} [label="{full_lbl}" fillcolor="{fill}" fontcolor="{WHITE}" style="{estyle}"];'
            )
            lines.append(
                f'  {p_state} -> {dot_id} [color="{ecolor}" penwidth={pw} style=dashed];'
            )

    lines.append("}")
    return "\n".join(lines)


def dot_to_svg(dot_src: str) -> str:
    r = subprocess.run(["dot", "-Tsvg"], input=dot_src, capture_output=True, text=True)
    if r.returncode != 0:
        return f"<pre>dot error: {r.stderr[:200]}</pre>"
    svg = r.stdout
    svg = "\n".join(l for l in svg.splitlines()
                    if not l.startswith("<?xml") and not l.startswith("<!DOCTYPE"))
    return svg.strip()


# ── HTML ───────────────────────────────────────────────────────────────────────

def build_html(data: dict, svg: str) -> str:
    request   = data.get("request", "")
    model     = data.get("model", "")
    status    = data.get("status", "")
    timestamp = data.get("timestamp_utc", "")[:19].replace("T", " ")
    nodes     = data["planning_tree"]["nodes"]

    total_attempts = sum(len(n.get("attempts", [])) for n in nodes)
    n_fail = sum(
        1 for n in nodes
        for att in n.get("attempts", [])
        if not att.get("step_verification", {}).get("passed", True)
    )
    n_pass = total_attempts - n_fail
    n_no_att = sum(1 for n in nodes if not n.get("attempts") and n["node_id"] != 0)
    dead = collect_dead_leaves(nodes)
    n_dead = sum(len(v) for v in dead.values())

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Plan Tree — {request}</title>
<style>
  body {{ font-family: Helvetica, Arial, sans-serif; background: #f0f2f5; margin: 0; padding: 0; }}
  h1   {{ text-align: center; padding: 24px 0 8px; color: #2c3e50; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 16px; font-size: 14px; }}
  .stats {{
    display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;
    max-width: 1100px; margin: 0 auto 20px; padding: 0 16px;
  }}
  .stat-box {{
    background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
    padding: 14px 24px; text-align: center; min-width: 160px;
  }}
  .stat-label {{ font-size: 12px; color: #888; margin-bottom: 4px; }}
  .stat-value {{ font-size: 32px; font-weight: bold; color: #2c3e50; }}
  .stat-sub   {{ font-size: 12px; color: #aaa; margin-top: 4px; }}
  .legend {{ display: flex; gap: 20px; justify-content: center; padding: 10px; font-size: 13px; color: #555; flex-wrap: wrap; }}
  .legend span {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
  .tree-card {{
    max-width: 98vw; margin: 0 auto 24px;
    background: white; border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.12); overflow: hidden;
  }}
  .tree-header {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; background: #2c3e50; color: white;
  }}
  .qtitle {{ font-size: 13px; opacity: .9; }}
  .svg-wrap {{
    padding: 12px; background: #f8f9fa;
    overflow: hidden; cursor: grab; user-select: none;
    display: flex; justify-content: center;
  }}
  .svg-wrap.dragging {{ cursor: grabbing; }}
  .svg-wrap svg {{ display: block; }}
</style>
</head>
<body>
<h1>SHRDLU Plan Tree</h1>
<div class="subtitle">
  {model} &nbsp;|&nbsp; {timestamp} UTC &nbsp;|&nbsp; status: <b>{status}</b>
</div>

<div class="stats">
  <div class="stat-box">
    <div class="stat-label">Actions (no retry)</div>
    <div class="stat-value">{n_no_att}</div>
    <div class="stat-sub">accepted first try</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Attempts — PASS</div>
    <div class="stat-value" style="color:{PASS_FILL}">{n_pass}</div>
    <div class="stat-sub">TLC verified</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Attempts — FAIL</div>
    <div class="stat-value" style="color:{FAIL_FILL}">{n_fail}</div>
    <div class="stat-sub">property violated</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Dead leaves (backtrack)</div>
    <div class="stat-value" style="color:{FAIL_FILL}">{n_dead}</div>
    <div class="stat-sub">from child_failure</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Total attempts</div>
    <div class="stat-value">{total_attempts}</div>
    <div class="stat-sub">across all states</div>
  </div>
</div>

<div class="legend">
  <span><span class="dot" style="background:{ROOT_FILL}"></span> ROOT / state node</span>
  <span><span class="dot" style="background:{PASS_FILL}"></span> action — TLC PASS</span>
  <span><span class="dot" style="background:{FAIL_FILL}; border: 2px dashed #c0392b"></span> action — TLC FAIL (dashed)</span>
  <span><span class="dot" style="background:{ERR_FILL}; border: 2px dashed #d35400"></span> planning error (dashed)</span>
  <span style="margin-left:8px">edge #N = generation order &nbsp;|&nbsp; scroll to zoom &nbsp;|&nbsp; drag to pan &nbsp;|&nbsp; double-click to reset</span>
</div>

<div class="tree-card">
  <div class="tree-header">
    <span class="qtitle">{request}</span>
  </div>
  <div class="svg-wrap" id="svg-wrap">{svg}</div>
</div>

<script>
(function() {{
  var wrap = document.getElementById('svg-wrap');
  if (!wrap) return;
  var svg  = wrap.querySelector('svg');
  if (!svg) return;

  // Initialise viewBox from the SVG's natural size
  var vb = svg.viewBox.baseVal;
  if (!vb || vb.width === 0) {{
    var w = parseFloat(svg.getAttribute('width'))  || 800;
    var h = parseFloat(svg.getAttribute('height')) || 600;
    svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
    vb = svg.viewBox.baseVal;
  }}
  svg.removeAttribute('width');
  svg.removeAttribute('height');
  svg.style.width  = '100%';
  svg.style.height = '80vh';

  var scale = 1, panX = vb.x, panY = vb.y;
  var origW = vb.width, origH = vb.height;

  function applyTransform() {{
    var w = origW / scale, h = origH / scale;
    svg.setAttribute('viewBox', panX + ' ' + panY + ' ' + w + ' ' + h);
  }}

  // Wheel → zoom centred on cursor
  wrap.addEventListener('wheel', function(e) {{
    e.preventDefault();
    var rect   = svg.getBoundingClientRect();
    var cx     = (e.clientX - rect.left) / rect.width;
    var cy     = (e.clientY - rect.top)  / rect.height;
    var factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    var newScale = Math.max(0.05, Math.min(80, scale * factor));

    var curW = origW / scale, curH = origH / scale;
    var mouseVbX = panX + cx * curW;
    var mouseVbY = panY + cy * curH;

    scale = newScale;
    var newW = origW / scale, newH = origH / scale;
    panX = mouseVbX - cx * newW;
    panY = mouseVbY - cy * newH;
    applyTransform();
  }}, {{ passive: false }});

  // Drag → pan
  var dragging = false, lastX, lastY;
  wrap.addEventListener('mousedown', function(e) {{
    if (e.button !== 0) return;
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    wrap.classList.add('dragging');
  }});
  window.addEventListener('mousemove', function(e) {{
    if (!dragging) return;
    var rect = svg.getBoundingClientRect();
    var dx = -(e.clientX - lastX) / rect.width  * (origW / scale);
    var dy = -(e.clientY - lastY) / rect.height * (origH / scale);
    panX += dx; panY += dy;
    lastX = e.clientX; lastY = e.clientY;
    applyTransform();
  }});
  window.addEventListener('mouseup', function() {{
    dragging = false; wrap.classList.remove('dragging');
  }});

  // Double-click → reset
  wrap.addEventListener('dblclick', function() {{
    scale = 1; panX = vb.x; panY = vb.y; applyTransform();
  }});
}})();
</script>
</body>
</html>"""


# ── entry point ────────────────────────────────────────────────────────────────

def plot_tree(trace_path: str, out_path: str):
    with open(trace_path) as f:
        data = json.load(f)

    dot_src = build_dot(data)
    svg     = dot_to_svg(dot_src)
    html    = build_html(data, svg)

    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    trace = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/robot/shrdlu/SHRDLUBlocksVLA/agent_traces/trace_20260621T134834939959Z.json"
    out = sys.argv[2] if len(sys.argv) > 2 else trace.replace(".json", "_plan_tree.html")
    plot_tree(trace, out)
