"""Lossless storage deduplication AFTER each actual API call has finished.

No request, response, prompt or decision is changed before/during execution.
Reconstruction verifies the original request.json SHA256 before archival.
PNG paths remain available through NTFS hardlinks to content-addressed blobs.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .io import digest, load, save

DATA_URL = re.compile(rb'data:image/png;base64,([A-Za-z0-9+/=]+)')


def put_blob(root, data):
    value = hashlib.sha256(data).hexdigest()
    path = root / "image_blobs" / value[:2] / (value + ".png")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".png.tmp")
        tmp.write_bytes(data); os.replace(tmp, path)
    if digest(path) != value:
        raise ValueError("Changed content-addressed image blob")
    return value, path


def reconstruct(folder, root):
    record = load(folder / "request.archive.json")
    template = (folder / "request.template.json").read_bytes()
    if hashlib.sha256(template).hexdigest() != record["template_sha256"]:
        raise ValueError("Changed request template")
    for value in record["image_sha256"]:
        blob = root / "image_blobs" / value[:2] / (value + ".png")
        if digest(blob) != value:
            raise ValueError("Changed archived evidence")
        template = template.replace(("__IMAGE_DATA_URL_SHA256_" + value + "__").encode("ascii"),
            b"data:image/png;base64," + base64.b64encode(blob.read_bytes()))
    if hashlib.sha256(template).hexdigest() != record["original_request_sha256"]:
        raise ValueError("Original request reconstruction failed")
    return template


def archive_call(folder, root):
    request = folder / "request.json"
    # Metadata/error is emitted last by request_packet. Never touch an in-flight request.
    if not request.exists() or not ((folder / "metadata.json").exists() or (folder / "error.json").exists()):
        return False
    original = request.read_bytes(); values = set()

    def replace(match):
        value, _ = put_blob(root, base64.b64decode(match.group(1), validate=True))
        values.add(value)
        return ("__IMAGE_DATA_URL_SHA256_" + value + "__").encode("ascii")
    template = DATA_URL.sub(replace, original)
    temporary = folder / "request.template.json.tmp"
    temporary.write_bytes(template); os.replace(temporary, folder / "request.template.json")
    save(folder / "request.archive.json", {"original_request_sha256": hashlib.sha256(original).hexdigest(),
        "original_request_bytes": len(original), "template_sha256": hashlib.sha256(template).hexdigest(),
        "image_sha256": sorted(values), "method": "exact byte substitution of PNG data URLs; lossless",
        "reconstruct": "archive_call_artifacts.py --root <run> --reconstruct <call-folder> --output <request.json>"})
    # Verify byte-for-byte before replacing the redundant embedded-image file.
    if reconstruct(folder, root) != original:
        raise ValueError("Lossless replay mismatch")
    for path in folder.glob("*.png"):
        value, blob = put_blob(root, path.read_bytes())
        if os.stat(path).st_ino == os.stat(blob).st_ino:
            continue
        temporary = path.with_suffix(".png.dedup.tmp")
        try:
            os.link(blob, temporary)
            os.replace(temporary, path)
        except OSError:
            # If hardlinks are unsupported, preserve the original PNG path.
            if temporary.exists():
                temporary.unlink()
    request.unlink()  # Fully reversible from the verified template and blobs.
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--reconstruct", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.reconstruct:
        if args.output is None:
            parser.error("--output required for reconstruction")
        args.output.write_bytes(reconstruct(args.reconstruct, root)); return
    manifest = {"archiver_sha256": digest(__file__), "storage_only": True,
        "method_or_API_changes": False, "original_request_reconstructable_byte_for_byte": True}
    save(root / "storage_manifest.json", manifest)
    total = 0
    while True:
        for request in (root / "scenes").glob("*/gpt_autonomous/run/gpt_calls/*/request.json"):
            if archive_call(request.parent, root):
                total += 1
        save(root / "storage_progress.json", {"calls_archived_in_this_process": total,
            "all_original_request_hashes_preserved": True})
        stopped = any((root / name).exists() for name in ("completed.json", "external_failure.json", "runtime_failure.json"))
        if not args.watch or stopped:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
