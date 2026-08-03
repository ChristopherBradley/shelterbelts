"""Emit a file's tail as a GitHub annotation, which is readable without auth.

Temporary CI diagnostic helper. Delete once the Overpass issue is resolved.
"""
import sys

path, title = sys.argv[1], sys.argv[2]
with open(path, errors="replace") as f:
    text = f.read()[-4000:]

encoded = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
print(f"::error title={title}::{encoded}")
