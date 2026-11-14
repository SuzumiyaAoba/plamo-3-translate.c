#!/usr/bin/env python3
"""Format this project's GNU-style AArch64 assembly without changing tokens.

Quoted data and comments are opaque. Keep directives and labels at column zero,
indent instructions by four spaces, and align their operands at column twelve.
The model generators use the same function so regeneration preserves formatting.
"""

import argparse
from pathlib import Path
import re

TOKENS = re.compile(r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|//.*|[ \t]+|,|[^\s,"'/]+|.""")
LABEL = re.compile(r"([\w.$]+:)\s*(.*)")


def format_line(line):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return stripped
    if stripped.startswith("//"):
        return ("    " if line[0].isspace() else "") + stripped
    code, comment = "", ""
    for token in TOKENS.findall(stripped):
        if token.startswith("//"):
            comment = token
            break
        if token.isspace():
            if code and not code.endswith(" "):
                code += " "
        elif token == ",":
            code = code.rstrip() + ", "
        else:
            code += token
    code = code.strip()
    label = LABEL.fullmatch(code)
    if label:
        code = label[1] + (" " + label[2] if label[2] else "")
    elif not code.startswith("."):
        instruction, *operands = code.split(maxsplit=1)
        code = "    " + (f"{instruction:<7} {operands[0]}" if operands else instruction)
    return code + ("  " + comment if comment else "")


def format_source(source):
    return "\n".join(format_line(line) for line in source.splitlines()).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        before = path.read_text()
        after = format_source(before)
        if after != before:
            path.write_text(after)


if __name__ == "__main__":
    main()
