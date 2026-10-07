"""Graph exports: Graphviz DOT (nested clusters), Mermaid flowchart and a plain-English description."""

from __future__ import annotations

from ..schema import DiagramGraph, RelationType, TextPlacement


def _q(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def to_dot(g: DiagramGraph) -> str:
    lines = ["digraph drawing {", "  compound=true; rankdir=LR; node [fontname=Helvetica];"]
    texts_by_shape: dict[str | None, list] = {}
    for t in g.texts:
        texts_by_shape.setdefault(t.inside_shape_id, []).append(t)

    def emit(sid: str | None, indent: str):
        for t in texts_by_shape.get(sid, []):
            label = t.text or t.id
            style = ' fontcolor="gray40" style="dashed"' if t.crossed_out else ""
            lines.append(f'{indent}{t.id} [label="{_q(label)}" shape=plaintext{style}];')
        for s in g.children(sid):
            lines.append(f"{indent}subgraph cluster_{s.id} {{")
            lines.append(f'{indent}  label="{s.id} ({s.type.value})"; style=rounded;')
            lines.append(f'{indent}  {s.id} [label="" shape=point width=0.05];')
            emit(s.id, indent + "  ")
            lines.append(f"{indent}}}")

    emit(None, "  ")
    for c in g.connections:
        ids = c.attached_ids()
        attr = ' penwidth=3' if c.heavy else ""
        arrow = "" if c.type.value == "arrow" else " dir=none"
        for a, b in zip(ids, ids[1:]):
            lines.append(f'  {a} -> {b} [label="{c.id}"{attr}{arrow}];')
    for r in g.relations:
        if r.type == RelationType.OVERLAPS:
            lines.append(f'  {r.subject} -> {r.object} [style=dotted dir=none label="overlaps"];')
    lines.append("}")
    return "\n".join(lines)


def to_mermaid(g: DiagramGraph) -> str:
    lines = ["flowchart TB"]
    texts_by_shape: dict[str | None, list] = {}
    for t in g.texts:
        texts_by_shape.setdefault(t.inside_shape_id, []).append(t)

    def txt(t):
        return (t.text or t.id).replace('"', "'")

    def emit(sid: str | None, indent: str):
        for t in texts_by_shape.get(sid, []):
            lines.append(f'{indent}{t.id}["{txt(t)}"]')
        for s in g.children(sid):
            lines.append(f'{indent}subgraph {s.id}["{s.id}: {s.type.value}"]')
            if not texts_by_shape.get(s.id) and not g.children(s.id):
                lines.append(f"{indent}  {s.id}_anchor(( ))")
            emit(s.id, indent + "  ")
            lines.append(f"{indent}end")

    emit(None, "  ")
    for c in g.connections:
        ids = c.attached_ids()
        link = "==>" if c.type.value == "arrow" else ("===" if c.heavy else "---")
        for a, b in zip(ids, ids[1:]):
            lines.append(f"  {a} {link} {b}")
    return "\n".join(lines)


def describe(g: DiagramGraph, max_items: int = 60) -> str:
    """Deterministic plain-English description of the structure."""
    if not g.shapes and not g.texts:
        return "No shapes or text were found."
    out: list[str] = []
    st = g.stats()
    kinds = ", ".join(f"{n} {k.replace('_', ' ')}{'s' if n > 1 else ''}" for k, n in sorted(st["shape_types"].items()))
    out.append(
        f"Found {st['shapes']} shape(s) ({kinds or 'none'}), {st['connections']} connection(s) and {st['texts']} text item(s);"
        f" nesting depth {st['max_depth']}."
    )

    def tx(ids):
        return ", ".join(g.label(i) for i in ids)

    def shape_line(s, indent):
        bits = [f"{indent}- {s.id}: {s.type.value}"]
        if s.crossed_out:
            bits.append(" (crossed out)")
        if s.text_inside:
            bits.append(f" containing text {tx(s.text_inside)}")
        if s.text_near:
            bits.append(f"; text nearby {tx(s.text_near)}")
        out.append("".join(bits))
        for ch in g.children(s.id):
            shape_line(ch, indent + "  ")

    for s in g.roots()[:max_items]:
        shape_line(s, "")
    for c in g.connections[:max_items]:
        ends = [g.label(i) for i in c.attached_ids()]
        kind = "heavy line" if c.heavy else c.type.value
        what = " and ".join(ends) if ends else "nothing identified"
        extra = f", crossing {', '.join(c.crosses)}" if c.crosses else ""
        lab = f", labelled {tx(c.label_ids)}" if c.label_ids else ""
        out.append(f"- {c.id}: {kind} connecting {what}{extra}{lab}")
    for r in g.relations:
        if r.type == RelationType.OVERLAPS:
            out.append(f"- {r.subject} overlaps {r.object}")
    free = [t for t in g.texts if t.placement == TextPlacement.FREE and t.text]
    if free:
        out.append(f"Free text: {tx([t.id for t in free[:20]])}")
    return "\n".join(out)
