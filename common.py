"""Read-only source access; all derived output is confined to this directory."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
BASE = WORK.parent
YEARS = list(range(2021, 2026))
ROOTS = {y: BASE / ("MCFire_Dataset_v2_MultiYear" if y == 2022 else f"MCFire_Dataset_v2_{y}") for y in YEARS}
RELEASES = {y: ROOTS[y] / "data/release" / ("MCFire_Dataset_v2.1_2022_RELEASE" if y == 2022 else f"MCFire_Dataset_v2_{y}_RELEASE") for y in YEARS}
AUX = WORK / "data/release/MCFire_Dataset_v2.1_2022_TERRAIN_AUXILIARY"
SEED = 20260829
POOLS = ["FIRE", "NEG_LOCAL_MATCHED", "NEG_GLOBAL_STRATIFIED"]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def relative(path):
    return Path(path).resolve().relative_to(BASE).as_posix()

def read(path, **kwargs):
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)

def dump(obj, name):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def save(df, name):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, encoding="utf-8-sig")

def one(parent, pattern):
    found = sorted(Path(parent).glob(pattern))
    if len(found) != 1:
        raise ValueError(f"Expected one {parent}/{pattern}: {found}")
    return found[0]

def intermediate(y, stage, pattern):
    return one(ROOTS[y] / "data/intermediate" / stage, pattern.format(year=y))

def labels(y):
    return read(one(RELEASES[y] / "labels", f"D2_S05_FINAL_BINARY_MASTER_{y}.csv"))

def splits(y):
    return read(one(RELEASES[y] / "splits", f"D2_S10_GROUP_AWARE_SPLIT_MASTER_{y}.csv"))

def protected_inventory():
    rows=[]
    for name, root in [(str(y), RELEASES[y]) for y in YEARS]+[("2022_terrain", AUX)]:
        paths=sorted(p for p in root.rglob("*") if p.is_file())
        for p in paths:
            rows.append(dict(package=name,path=relative(p),bytes=p.stat().st_size,sha256=sha(p)))
        print(f"Hashed protected {name}: {len(paths)} files", flush=True)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        target=OUT/"protected_files_before.csv"
        if target.exists():
            raise RuntimeError("Refusing to replace baseline inventory")
        save(protected_inventory(),target.name)
    elif sys.argv[1] == "verify":
        before=read(OUT/"protected_files_before.csv",dtype={"package":str})
        after=protected_inventory()
        save(after,"protected_files_after.csv")
        merged=before.merge(after,on=["package","path"],how="outer",suffixes=("_before","_after"),indicator=True)
        bad=merged[(merged["_merge"]!="both") | (merged.sha256_before!=merged.sha256_after) | (merged.bytes_before!=merged.bytes_after)]
        save(bad,"protected_files_differences.csv")
        dump(dict(status="PASS" if bad.empty else "FAIL",files_before=len(before),files_after=len(after),differences=len(bad)),"protected_integrity_result.json")
        assert bad.empty, "Protected source files changed"
