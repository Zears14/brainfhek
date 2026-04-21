"""LIRO: Loop Metadata — attach ``!llvm.loop`` hints to back-edge branches.

This pass identifies loop back-edges (``br label %loop.*head`` or
``br label %loopc.*head``) and attaches metadata nodes that hint the
LLVM vectorizer and loop unroller.
"""

from __future__ import annotations

import re

from llvmlite import binding

from bf2.liro.base import LIROPass, register_liro

_LOOP_BACKEDGE_RE = re.compile(
    r"^(\s+br\s+label\s+%(?P<label>\S*loop\S*(?:head|start)\S*))\s*(?:;.*)?$", re.IGNORECASE
)


@register_liro
class LoopMetadata(LIROPass):
    name = "loop_metadata"
    description = "Attach !llvm.loop vectorize/unroll hints to loop back-edges"

    def run(self, mod: binding.ModuleRef) -> binding.ModuleRef:
        ir_text = str(mod)
        lines = ir_text.split("\n")

        backedge_indices: list[int] = []
        for i, line in enumerate(lines):
            if _LOOP_BACKEDGE_RE.match(line):
                backedge_indices.append(i)

        if not backedge_indices:
            return mod

        max_existing_id = -1
        for line in lines:
            m = re.match(r"^!(\d+)\s*=", line)
            if m:
                max_existing_id = max(max_existing_id, int(m.group(1)))

        next_id = max_existing_id + 1
        out = list(lines)
        metadata_lines: list[str] = []

        for idx in backedge_indices:
            line = lines[idx]
            match = _LOOP_BACKEDGE_RE.match(line)
            if not match:
                continue
            label = match.group("label")

            is_clean = True
            label_def_idx = -1
            clean_label = label.strip('"')
            label_def_pattern = f"{clean_label}:"
            for i in range(idx - 1, -1, -1):
                curr_line = lines[i].strip()
                if curr_line.split(';')[0].strip() == label_def_pattern:
                    label_def_idx = i
                    break

            if label_def_idx == -1:
                continue

            for i in range(label_def_idx + 1, idx):
                lbody = lines[i]
                if (
                    ("call " in lbody or "invoke " in lbody or "ret " in lbody)
                    and "@llvm." not in lbody
                ):
                    is_clean = False
                    break

            loop_id = next_id
            unroll_id = next_id + 1
            next_id += 2

            nodes = [f"!{loop_id}", f"!{unroll_id}"]
            vec_id = -1
            if is_clean:
                vec_id = next_id
                next_id += 1
                nodes.append(f"!{vec_id}")

            out[idx] = out[idx].rstrip() + f", !llvm.loop !{loop_id}"

            node_list = ", ".join(nodes)
            metadata_lines.append(f"!{loop_id} = distinct !{{{node_list}}}")
            metadata_lines.append(f'!{unroll_id} = !{{!"llvm.loop.unroll.enable"}}')
            if is_clean:
                metadata_lines.append(f'!{vec_id} = !{{!"llvm.loop.vectorize.enable", i1 true}}')

        if metadata_lines:
            out.append("")
            out.extend(metadata_lines)

        return binding.parse_assembly("\n".join(out))