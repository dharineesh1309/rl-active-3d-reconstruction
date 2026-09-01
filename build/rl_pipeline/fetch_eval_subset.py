"""
fetch_eval_subset.py
────────────────────
Pull a small, real ShapeNet subset for benchmarking and axis calibration,
without downloading the 12.3 GB ShapeNetRendering archive.

Why this exists
───────────────
The canonical renderings live in a single .tgz. Gzip cannot be seeked, so
reaching a given synset means streaming a large fraction of 12.3 GB even though
the images we actually need total a few tens of megabytes. On a slow link that
is hours for nothing.

Hugging Face mirrors publish the same R2N2 data as ZIPs, and ZIP keeps its
index at the end of the file. Given HTTP range requests, stdlib zipfile can
therefore read the index and then pull individual members — genuine random
access over the network. Cost drops from ~4 GB to an index read plus a few MB
of images.

    python fetch_eval_subset.py --models 150                  # chair only
    python fetch_eval_subset.py --synsets 02691156 02958343   # airplane + car

Two mirrors, picked automatically by which synsets are asked for:

    MIRRORS["chair"]  2.71 GB, 25 MB index, chair only
    MIRRORS["3c"]    13.69 GB, 71 MB index, airplane + car + chair

**There is no seekable mirror of all 13 categories.** The 3-class archive is
the widest random-access source found; a full 13-category sweep has to run
where the canonical data already sits (see KAGGLE_STEPS.md). Three categories
is still enough to break the chair-only monoculture, and they are usefully
different: thin open structure (airplane), solid convex mass (car), thin
cluttered structure (chair).

Caveat: these mirrors store 10 views per model, not Choy's 24. That is fine for
what it is for — backbone benchmarking and axis calibration care about real
images and matching ground truth, not about view count. It is NOT sufficient
for RL training, which needs the full 24-view set the policy's action space is
built around; fetch that on the GPU box where bandwidth is not the constraint.
"""

import argparse
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

# name -> (url, prefix holding the per-synset rendering dirs, synsets available)
MIRRORS = {
    "chair": (
        "https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset"
        "/resolve/main/r2n2_shapenet_dataset.zip",
        "r2n2_shapenet_dataset/r2n2/ShapeNetRendering/",
        {"03001627"},
    ),
    "3c": (
        "https://huggingface.co/datasets/learning3dvision/r2n2_shapenet_dataset_full"
        "/resolve/main/r2n2_shapenet_dataset_full.zip",
        "r2n2_shapenet_dataset_full/r2n2/ShapeNetRendering/",
        {"02691156", "02958343", "03001627"},
    ),
}

CHAIR = "03001627"


class HTTPRangeFile(io.RawIOBase):
    """
    Seekable read-only file backed by HTTP range requests.

    This is the whole trick: zipfile only needs seek/read, so handing it one of
    these lets it parse a remote archive's central directory and then fetch
    single members, instead of downloading the archive.
    """

    def __init__(self, url, retries=3):
        self.url = url
        self.retries = retries
        self._pos = 0
        self.bytes_fetched = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        if n <= 0 or self._pos >= self.size:
            return b""
        end = min(self._pos + n, self.size) - 1
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self._pos}-{end}", "User-Agent": "curl/8"},
        )
        last = None
        for _ in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                break
            except Exception as exc:            # transient CDN hiccups are common
                last = exc
        else:
            raise RuntimeError(f"range request failed after {self.retries} tries: {last}")
        self._pos += len(data)
        self.bytes_fetched += len(data)
        return data


def _fetch_span(remote, zf, infos):
    """
    Pull every member in `infos` with ONE range request and return {name: bytes}.

    A model's ~11 files sit contiguously in the archive, so fetching them one at
    a time costs 11 ranged GETs, each paying HF's 302 redirect and a fresh TLS
    round trip. Measured, that was about one model per minute -- hours for a
    two-synset sample. Fetching the whole span at once and splitting it locally
    turns that into one request, because the bytes in between are a handful of
    KB we would rather download than round-trip for.

    ZIP local headers are parsed by hand: `zipfile` cannot read from a bare
    slice of an archive, but the format is fixed-width and documented, so this
    is a dozen lines rather than a dependency.
    """
    import struct
    import zlib

    start = min(i.header_offset for i in infos)
    # header_offset + 30-byte fixed header + name + extra + data. The central
    # directory's `extra` can differ in length from the local one, so pad
    # generously rather than trusting it; the tail costs a few KB.
    end = max(i.header_offset + 30 + len(i.filename) + len(i.extra) + i.compress_size
              for i in infos) + 4096
    end = min(end, remote.size)

    # This is only a win because a model's files sit next to each other. If an
    # archive ever interleaves them, the span could cover gigabytes of other
    # people's data to collect a few hundred KB, so fall back to one request per
    # member rather than silently downloading the difference.
    wanted = sum(i.compress_size for i in infos)
    if end - start > max(4 * wanted, 8 << 20):
        return {i.filename: zf.read(i.filename) for i in infos}

    remote.seek(start)
    buf = b""
    while len(buf) < end - start:
        chunk = remote.read(end - start - len(buf))
        if not chunk:
            break
        buf += chunk

    out = {}
    for info in infos:
        off = info.header_offset - start
        if buf[off:off + 4] != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {info.filename}")
        name_len, extra_len = struct.unpack("<HH", buf[off + 26:off + 30])
        data_at = off + 30 + name_len + extra_len
        raw = buf[data_at:data_at + info.compress_size]
        if len(raw) != info.compress_size:
            raise RuntimeError(f"short read for {info.filename}")
        if info.compress_type == zipfile.ZIP_STORED:
            data = raw
        else:
            data = zlib.decompressobj(-15).decompress(raw)
        # The archive stores a CRC per member, so hand-parsing is self-checking:
        # a misread offset or a wrong length fails here rather than writing a
        # corrupt PNG that only surfaces as a mystery IoU later.
        if zlib.crc32(data) != info.CRC:
            raise RuntimeError(f"CRC mismatch for {info.filename}")
        out[info.filename] = data
    return out


def _open_mirror(name):
    """
    Open a mirror and return (remote, zipfile, prefix, namelist).

    Reading the central directory costs 25-71 MB and dominates a small fetch, so
    every synset asked for in one run shares a single open archive rather than
    paying the index per synset.
    """
    url, prefix, _ = MIRRORS[name]
    remote = HTTPRangeFile(url)
    print(f"  archive {remote.size / 1e9:.2f} GB; reading index ...")
    zf = zipfile.ZipFile(remote)
    names = zf.namelist()
    print(f"  index: {len(names)} entries, {remote.bytes_fetched / 1e6:.1f} MB read")
    return remote, zf, prefix, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=int, default=150,
                   help="How many models to fetch PER SYNSET (default: 150)")
    p.add_argument("--synsets", nargs="+", default=[CHAIR],
                   help=f"Synset ids. Available: {sorted(MIRRORS['3c'][2])} "
                        "(airplane, car, chair). Default: chair.")
    p.add_argument("--out", default="../ShapeNetRendering",
                   help="Destination rendering root")
    p.add_argument("--voxel-root", default="../ShapeNetVox32",
                   help="Only fetch models that already have ground-truth voxels here")
    p.add_argument("--split", default="test", choices=["train", "val", "test", "any"],
                   help="Official split to draw from (default: test). Benchmarking on "
                        "train-split models measures memorisation, not reconstruction.")
    p.add_argument("--taxonomy", default="datasets/ShapeNet.json")
    args = p.parse_args()

    wanted = list(dict.fromkeys(args.synsets))
    unavailable = [s for s in wanted if s not in MIRRORS["3c"][2]]
    if unavailable:
        p.error(f"no seekable mirror has {unavailable}. Only "
                f"{sorted(MIRRORS['3c'][2])} are reachable this way; run the "
                f"13-category sweep on Kaggle instead (see KAGGLE_STEPS.md).")

    # The chair-only mirror is 5x smaller to index, so prefer it when it suffices.
    mirror = "chair" if set(wanted) <= MIRRORS["chair"][2] else "3c"
    print(f"opening '{mirror}' mirror for {len(wanted)} synset(s) ...")
    remote, zf, prefix, names = _open_mirror(mirror)

    tax = json.load(open(args.taxonomy)) if args.split != "any" else None
    grand_total = 0

    for synset in wanted:
        out_root = Path(args.out) / synset
        vox_root = Path(args.voxel_root) / synset
        sub = prefix + synset + "/"

        by_model = {}
        for n in names:
            if n.startswith(sub) and not n.endswith("/"):
                by_model.setdefault(n[len(sub):].split("/")[0], []).append(n)
        print(f"\n[{synset}] {len(by_model)} models in mirror")

        # Restrict to the requested official split. These backbones were trained
        # on the train split, so scoring them there inflates IoU badly --
        # Pix2Vox-A jumped from a plausible number to 0.84 on train models.
        if tax is not None:
            entry = next((c for c in tax if c["taxonomy_id"] == synset), None)
            if entry is None:
                print(f"  WARNING: {synset} absent from taxonomy; skipping split filter")
            else:
                allowed = set(entry[args.split])
                by_model = {m: v for m, v in by_model.items() if m in allowed}
                print(f"  '{args.split}' split: {len(by_model)} candidates")

        # Only take models whose ground truth we already hold, so every fetched
        # object is immediately scorable.
        have_voxels = {d.name for d in vox_root.iterdir()} if vox_root.is_dir() else set()
        if have_voxels:
            usable = [m for m in sorted(by_model) if m in have_voxels]
            print(f"  {len(usable)} with local voxels")
        else:
            usable = sorted(by_model)
            print(f"  WARNING: no voxels under {vox_root}; fetching without matching")

        selected = usable[:args.models]
        before = remote.bytes_fetched
        written = 0
        for i, model in enumerate(selected, 1):
            dest = out_root / model / "rendering"
            dest.mkdir(parents=True, exist_ok=True)
            todo = []
            for name in by_model[model]:
                leaf = name.split("/")[-1]
                if not (leaf.endswith(".png") or leaf == "rendering_metadata.txt"):
                    continue
                if not (dest / leaf).exists():
                    todo.append(name)
            if todo:
                blobs = _fetch_span(remote, zf, [zf.getinfo(n) for n in todo])
                for name, data in blobs.items():
                    (dest / name.split("/")[-1]).write_bytes(data)
                    written += 1
            if i % 25 == 0 or i == len(selected):
                print(f"  {i}/{len(selected)} models, {written} files, "
                      f"{(remote.bytes_fetched - before) / 1e6:.1f} MB")
        grand_total += len(selected)
        print(f"  done: {len(selected)} models -> {out_root}")

    print(f"\ntotal: {grand_total} models across {len(wanted)} synset(s)")
    print(f"downloaded this run: {remote.bytes_fetched / 1e6:.1f} MB "
          f"(vs 12.3 GB for the canonical .tgz)")


if __name__ == "__main__":
    main()
