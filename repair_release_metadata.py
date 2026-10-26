"""Repair release metadata only; never re-run extraction, labels, splits or models.

Default is preflight only. --apply requires the saved successful preflight and
creates full byte-exact backups before changing any release.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
OUT = ROOT / "release_metadata_correction_20261016"
PROJECTS = {y: (ROOT if y == 2022 else BASE / f"MCFire_Dataset_v2_{y}") for y in range(2021, 2026)}
NAMES = {y: f"MCFire_Dataset_v2_{y}_RELEASE" for y in PROJECTS}
NAMES[2022] = "MCFire_Dataset_v2.1_2022_RELEASE"
OLD_NAMES = {**NAMES, 2023: "MCFire_Dataset_v2_RELEASE"}
EXPECTED = {2021: 1077, 2022: 1083, 2023: 2023, 2024: 1316, 2025: 1163}
TIME_KEYS = {"reference_time_epoch_s.npy", "slot_time_epoch_s.npy"}
REVISION = "2026-10-16_epoch_metadata_r1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def epoch(value):
    value = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    assert dt.tzinfo is not None, value
    return int(dt.timestamp())


def relative(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def release_path(year, final=False):
    return PROJECTS[year] / "data/release" / (NAMES[year] if final else OLD_NAMES[year])


def slot_data(year, release):
    path = release / f"tensors/D2_S07_MASK_FIRST_SLOT_INDEX_{year}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig")
    assert not frame.duplicated(["sample_id", "tensor_time_index"]).any()
    manifest_path = PROJECTS[year] / f"data/intermediate/S06_HIMAWARI_CAUSAL_PLAN/D2_S06_CAUSAL_SLOT_MANIFEST_{year}.csv"
    causal = pd.read_csv(manifest_path, encoding="utf-8-sig")
    assert not causal.duplicated(["sample_id", "slot_index"]).any()
    joined = frame.merge(causal[["sample_id", "slot_index", "slot_time_utc"]],
                         on=["sample_id", "slot_index"], how="left", validate="one_to_one", suffixes=("", "_causal"))
    assert joined.slot_time_utc_causal.notna().all()
    assert all(epoch(a) == epoch(b) for a, b in zip(joined.slot_time_utc, joined.slot_time_utc_causal))
    result = {}
    for sid, group in frame.groupby("sample_id", sort=False):
        group = group.sort_values("tensor_time_index")
        assert group.tensor_time_index.tolist() == list(range(19)), sid
        offsets = group.slot_offset_from_anchor_min.astype(int).to_numpy()
        assert offsets.tolist() == list(range(-180, 1, 10)), sid
        ref_values = {epoch(s) for s in group.reference_time_utc}
        assert len(ref_values) == 1, sid
        ref = ref_values.pop()
        slots = np.array([epoch(s) for s in group.slot_time_utc], dtype=np.int64)
        assert np.array_equal(slots, ref // 600 * 600 + offsets * 60), sid
        assert slots.max() <= ref
        result[sid] = (ref, slots)
    return result, path, manifest_path


def inspect_year(year):
    print(f"PREFLIGHT {year}: loading UTC slot indexes", flush=True)
    release = release_path(year)
    assert release.exists(), release
    slots, slot_index, causal = slot_data(year, release)
    _, index = read_csv(release / f"tensors/D2_S07_MASK_FIRST_SAMPLE_INDEX_{year}.csv")
    ix = {r["sample_id"]: r for r in index}
    assert len(ix) == len(index) == EXPECTED[year]
    npzs = sorted((release / f"tensors/samples/{year}").glob("*.npz"))
    assert len(npzs) == len(ix) == len(slots)
    def inspect_tensor(p):
        raw = p.read_bytes()
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            sid = str(z["sample_id"].item())
            expected_ref, expected_slots = slots[sid]
            assert epoch(str(z["reference_time_utc"].item())) == expected_ref, sid
            assert np.array_equal(z["slot_offset_min"], np.arange(-180, 1, 10)), sid
            old_ref = int(z["reference_time_epoch_s"].item())
            ref_bad = old_ref != expected_ref or z["reference_time_epoch_s"].dtype != np.dtype("int64")
            slot_bad = not np.array_equal(z["slot_time_epoch_s"], expected_slots) or z["slot_time_epoch_s"].dtype != np.dtype("int64")
        digest = hashlib.sha256(raw).hexdigest()
        assert ix[sid]["tensor_sha256"] == digest, (sid, "initial tensor checksum mismatch")
        return dict(year=year, sample_id=sid, tensor_relative_path=p.relative_to(release).as_posix(),
                         old_sha256=digest, old_bytes=p.stat().st_size, old_reference_epoch_s=old_ref,
                         expected_reference_epoch_s=expected_ref, reference_needs_repair=bool(ref_bad),
                         slots_need_repair=bool(slot_bad))
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for number, row in enumerate(pool.map(inspect_tensor, npzs), 1):
            rows.append(row)
            if number % 200 == 0:
                print(f"PREFLIGHT {year}: tensor {number}/{len(npzs)}", flush=True)
    return rows, {"year": year, "samples": len(rows), "reference_repairs": sum(r["reference_needs_repair"] for r in rows),
                  "slot_array_repairs": sum(r["slots_need_repair"] for r in rows),
                  "slot_index": relative(slot_index), "slot_index_sha256": sha(slot_index),
                  "causal_manifest": relative(causal), "causal_manifest_sha256": sha(causal)}


def export_rules_and_counts(final=False):
    counts, audits, evidence = [], [], []
    for year, project in PROJECTS.items():
        release = release_path(year, final=final)
        for scope, path in [("S05_binary_labels", release / f"labels/D2_S05_FINAL_BINARY_MASTER_{year}.csv"),
                            ("ready_tensor_samples", release / f"splits/D2_S10_GROUP_AWARE_SPLIT_MASTER_{year}.csv")]:
            f = pd.read_csv(path, encoding="utf-8-sig")
            assert not f.sample_id.duplicated().any()
            counts.append(dict(year=year, scope=scope, FIRE=int(f.final_label.eq("FIRE").sum()),
                               NEG_LOCAL_MATCHED=int(f.source_pool.eq("NEG_LOCAL_MATCHED").sum()),
                               NEG_GLOBAL_STRATIFIED=int(f.source_pool.eq("NEG_GLOBAL_STRATIFIED").sum()),
                               total=len(f), source=relative(path), source_sha256=sha(path)))
        cfg_path = project / "config.yaml"
        config_text = cfg_path.read_text(encoding="utf-8-sig")
        # Export only this non-secret section; do not include credentials elsewhere.
        cfg = config_text.split("\ns015:\n", 1)[1].split("\ns02:", 1)[0]
        (OUT / f"candidate_rules_config_{year}.yaml").write_text("s015:\n" + cfg, encoding="utf-8")
        a = project / f"data/intermediate/S01_5_CANDIDATE_AUDIT/D2_S015_EVENT_AUDIT_MASTER_{year}.csv"
        if not a.exists() and year == 2023:
            a = ROOT / f"data/intermediate/S01_5_CANDIDATE_AUDIT/D2_S015_EVENT_AUDIT_MASTER_{year}.csv"
        f = pd.read_csv(a, encoding="utf-8-sig")
        persistent = (f.recurrence_active_months >= 6) & (f.recurrence_event_count >= 12)
        chaining = (f.duration_hours > 48) | (f.point_count > 100) | (f.bbox_diagonal_km > 5)
        hold = persistent | chaining
        p1 = ~hold & (f.reference_frp_max <= 10) & (f.reference_point_count <= 2) & (f.duration_hours <= 12) & (f.max_single_overpass_point_count <= 3)
        p2 = ~hold & ~p1 & (f.reference_frp_max <= 30) & (f.reference_point_count <= 5) & (f.duration_hours <= 24) & (f.max_single_overpass_point_count <= 8)
        pred = np.select([hold, p1, p2], ["HOLD", "P1", "P2"], default="P3")
        mismatches = int((pred != f.candidate_priority_tier).sum())
        assert mismatches == 0, (year, mismatches)
        assert not (persistent != f.persistent_source_risk.eq("HIGH")).any()
        assert not (chaining != f.chaining_risk.eq("HIGH")).any()
        audits.append(dict(year=year, candidate_events=len(f), P1=int(p1.sum()), P2=int(p2.sum()),
                           P3=int((~hold & ~p1 & ~p2).sum()), HOLD=int(hold.sum()), rule_mismatches=mismatches,
                           source=relative(a), source_sha256=sha(a)))
        code_path = project / "src/candidate_audit.py"
        code = code_path.read_text(encoding="utf-8-sig")
        lines = code.splitlines()
        with (OUT / f"candidate_rule_code_{year}.txt").open("w", encoding="utf-8") as out:
            for i, line in enumerate(lines[:222], 1):
                out.write(f"{i}: {line}\n")
        evidence.append(dict(year=year, config_path=relative(cfg_path), config_sha256=sha(cfg_path),
                             code_path=relative(code_path), code_sha256=sha(code_path)))
    write_csv(OUT / "annual_sample_counts.csv", counts)
    write_csv(OUT / "candidate_rule_historical_output_check.csv", audits)
    write_csv(OUT / "candidate_rule_sources.csv", evidence)
    return counts, audits


def preflight():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "COMPLETE.json").exists(), "Already applied; use --verify"
    all_rows, summaries = [], []
    for year in PROJECTS:
        rows, summary = inspect_year(year)
        all_rows.extend(rows)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    export_rules_and_counts()
    write_csv(OUT / "timestamp_preflight_all_samples.csv", all_rows)
    write_json(OUT / "preflight.json", {"status": "PASS", "revision": REVISION, "years": summaries, "total_samples": len(all_rows)})


def replace_time_members(path, ref, slots):
    replacements = {"reference_time_epoch_s.npy": np.array(ref, dtype=np.int64), "slot_time_epoch_s.npy": slots.astype(np.int64)}
    temporary = path.with_suffix(".npz.metadata_repair_tmp")
    original_bytes = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(original_bytes)) as src, zipfile.ZipFile(temporary, "w") as dst:
        assert TIME_KEYS <= set(src.namelist())
        for info in src.infolist():
            content = src.read(info.filename)
            if info.filename in replacements:
                b = io.BytesIO()
                np.lib.format.write_array(b, replacements[info.filename], allow_pickle=False)
                content = b.getvalue()
            dst.writestr(info, content)
    with zipfile.ZipFile(io.BytesIO(original_bytes)) as before, zipfile.ZipFile(io.BytesIO(temporary.read_bytes())) as after:
        assert before.namelist() == after.namelist()
        assert all(before.read(n) == after.read(n) for n in before.namelist() if n not in TIME_KEYS)
    os.replace(temporary, path)


def inventory(root):
    paths = [p for p in sorted(root.rglob("*")) if p.is_file() and p.name != "CURRENT_FILE_MANIFEST.json"]
    def item(p):
        return {"relative_path": p.relative_to(root).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha(p)}
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(item, paths))


def copy_release(source, destination):
    # Only used before any original package is changed. Existing partial backup
    # copies can safely be completed, but this is not a post-mutation resume.
    paths = [p for p in source.rglob("*") if p.is_file()]
    for p in paths:
        (destination / p.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
    def copy(p):
        shutil.copy2(p, destination / p.relative_to(source))
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(copy, paths))


def fix_csv(path, tensor_map):
    fields, rows = read_csv(path)
    if not fields or "sample_id" not in fields:
        return False
    targets = {"tensor_sha256", "tensor_bytes", "tensor_file", "tensor_relpath"} & set(fields)
    if not targets:
        return False
    changed = False
    for r in rows:
        item = tensor_map.get(r["sample_id"])
        if not item:
            continue
        for col in targets:
            value = {"tensor_sha256": item["new_sha256"], "tensor_bytes": str(item["new_bytes"]),
                     "tensor_file": item["tensor_relative_path"], "tensor_relpath": item["tensor_relative_path"]}[col]
            if r[col] != value:
                r[col] = value
                changed = True
    if changed:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return changed


def apply(resume=False):
    plan = json.loads((OUT / "preflight.json").read_text(encoding="utf-8"))
    assert plan["status"] == "PASS" and plan["total_samples"] == 6662
    assert not (OUT / "COMPLETE.json").exists(), "Already applied"
    auxiliary = ROOT / "data/release/MCFire_Dataset_v2.1_2022_TERRAIN_AUXILIARY"
    if resume:
        assert not release_path(2023).exists() and release_path(2023, final=True).exists()
        assert (auxiliary / "terrain").is_dir()
        for year in PROJECTS:
            original_inventory = json.loads((OUT / f"original_inventory_{year}.json").read_text(encoding="utf-8"))
            backup = OUT / f"backups/{year}/{OLD_NAMES[year]}"
            assert inventory(backup) == original_inventory, (year, "backup verification before resume")
        print("ALL ORIGINAL BACKUPS REVERIFIED FOR RESUME", flush=True)
    else:
        assert release_path(2023).exists() and not release_path(2023, final=True).exists(), "Not a pre-mutation backup resume"
        assert not auxiliary.exists()
    _, planned = read_csv(OUT / "timestamp_preflight_all_samples.csv")
    # All complete releases are copied before the first original file is changed.
    for year in ([] if resume else PROJECTS):
        original = release_path(year)
        backup = OUT / f"backups/{year}/{OLD_NAMES[year]}"
        print(f"BACKING UP {year}", flush=True)
        copy_release(original, backup)
        src_inventory, backup_inventory = inventory(original), inventory(backup)
        assert src_inventory == backup_inventory, year
        write_json(OUT / f"original_inventory_{year}.json", src_inventory)
        external = PROJECTS[year] / "data/release"
        for p in external.glob("*.json"):
            shutil.copy2(p, backup.parent / p.name)
        for p in [PROJECTS[year] / "src/mask_first_tensor_builder.py"]:
            if p.exists():
                shutil.copy2(p, backup.parent / p.name)
        print(f"BACKUP VERIFIED {year}", flush=True)
    if not resume:
        release_path(2023).rename(release_path(2023, final=True))
        auxiliary.mkdir()
        shutil.move(str(release_path(2022) / "terrain"), auxiliary / "terrain")
        shutil.copy2(release_path(2022) / "metadata/dem_data_dictionary.json", auxiliary / "dem_data_dictionary.json")
    report_rows, csv_changes = [], []
    for year in PROJECTS:
        release = release_path(year, final=True)
        slots, _, _ = slot_data(year, release)
        def repair_tensor(row):
            path = release / row["tensor_relative_path"]
            current_sha = sha(path)
            ref, times = slots[row["sample_id"]]
            needs = row["reference_needs_repair"] == "True" or row["slots_need_repair"] == "True"
            if current_sha != row["old_sha256"]:
                assert resume and needs, (year, row["sample_id"], "unexpected changed tensor")
                # An interrupted run may already have atomically installed this
                # tensor. Accept it only after full time and non-time validation.
                before = OUT / f"backups/{year}/{OLD_NAMES[year]}" / row["tensor_relative_path"]
                raw = path.read_bytes()
                with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                    assert z["reference_time_epoch_s"].dtype == np.dtype("int64")
                    assert z["slot_time_epoch_s"].dtype == np.dtype("int64")
                    assert int(z["reference_time_epoch_s"].item()) == ref
                    assert np.array_equal(z["slot_time_epoch_s"], times)
                with zipfile.ZipFile(before) as original, zipfile.ZipFile(io.BytesIO(raw)) as current:
                    assert original.namelist() == current.namelist()
                    assert all(original.read(n) == current.read(n) for n in original.namelist() if n not in TIME_KEYS)
                assert not path.with_suffix(".npz.metadata_repair_tmp").exists()
            elif needs:
                replace_time_members(path, ref, times)
            new = {**row, "new_sha256": sha(path), "new_bytes": path.stat().st_size, "modified": needs}
            return new
        year_rows = [r for r in planned if int(r["year"]) == year]
        tensor_map = {}
        with ThreadPoolExecutor(max_workers=6) as pool:
            for number, new in enumerate(pool.map(repair_tensor, year_rows), 1):
                tensor_map[new["sample_id"]] = new
                report_rows.append(new)
                if number % 200 == 0:
                    print(f"REPAIR {year}: {number}/{len(year_rows)}", flush=True)
        roots = [release] + ([auxiliary] if year == 2022 else [])
        for root in roots:
            for path in sorted(root.rglob("*.csv")):
                old_sha = sha(path)
                if fix_csv(path, tensor_map):
                    csv_changes.append(dict(year=year, file=relative(path), old_sha256=old_sha, new_sha256=sha(path)))
        # Identical mechanical bug fix in each builder; source copies were backed up.
        # Do not execute the pipeline or modify any historical intermediate data.
        builder = PROJECTS[year] / "src/mask_first_tensor_builder.py"
        code = builder.read_text(encoding="utf-8-sig")
        old = 'return (dt.astype("int64") // 10**9).to_numpy(dtype=np.int64)'
        if old in code:
            builder.write_text(code.replace(old, 'return np.asarray([int(value.timestamp()) for value in dt], dtype=np.int64)'), encoding="utf-8")
        (release / "README.md").write_text(release_readme(year), encoding="utf-8")
        if year == 2022:
            for name in ["dataset_description.json", "dem_data_dictionary.json", "v2_1_release_notes.json"]:
                p = release / "metadata" / name
                if p.exists():
                    doc = json.loads(p.read_text(encoding="utf-8-sig"))
                    doc["current_release_correction"] = {
                        "revision": REVISION,
                        "terrain_location": "../MCFire_Dataset_v2.1_2022_TERRAIN_AUXILIARY/terrain/",
                        "core_sample_metadata": "splits/D2_S10_GROUP_AWARE_SPLIT_MASTER_2022.csv",
                        "note": "Historical terrain-relative paths in original fields refer to the separate auxiliary package. No terrain was removed from the archive; it was moved out of the five-year core release."
                    }
                    write_json(p, doc)
        manifest = {"revision": REVISION, "year": year, "release_name": NAMES[year],
                    "sample_count": EXPECTED[year], "scope": "Metadata correction; all other NPZ members and scientific assignments unchanged",
                    "historical_stage_manifests": "Stage S05-S17 JSON/ZIP files are unchanged historical execution provenance; their old tensor/index hashes are not current release checksums.",
                    "files": inventory(release)}
        write_json(release / "CURRENT_FILE_MANIFEST.json", manifest)
        external_dir = PROJECTS[year] / "data/release"
        active_name = "MCFire_Dataset_v2.1_2022_FINAL_RELEASE_MANIFEST.json" if year == 2022 else f"MCFire_Dataset_v2_{year}_FINAL_RELEASE_MANIFEST.json"
        write_json(external_dir / active_name, {**manifest, "release_path": NAMES[year]})
        # Update the old-name external manifest too, retaining a copy in backups.
        if year == 2023:
            old_external = external_dir / "MCFire_Dataset_v2_FINAL_RELEASE_MANIFEST.json"
            if old_external.exists():
                write_json(old_external, {"status": "SUPERSEDED", "current_release": NAMES[year], "current_manifest": active_name, "revision": REVISION})
        current = external_dir / f"CURRENT_RELEASE_{year}.json"
        old_current = json.loads(current.read_text(encoding="utf-8-sig")) if current.exists() else {}
        write_json(current, {**old_current, "year": year, "status": "CURRENT", "metadata_revision": REVISION,
                             "recommended_release": NAMES[year], "recommended_manifest": active_name,
                             "recommended_manifest_sha256": sha(external_dir / active_name)})
        print(f"REPAIRED {year}: {sum(r['modified'] for r in tensor_map.values())} tensors", flush=True)
    (auxiliary / "README.md").write_text(terrain_readme(), encoding="utf-8")
    write_json(auxiliary / "CURRENT_FILE_MANIFEST.json", {"revision": REVISION, "files": inventory(auxiliary)})
    write_csv(OUT / "tensor_hash_changes.csv", report_rows)
    write_csv(OUT / "csv_metadata_changes.csv", csv_changes)
    export_rules_and_counts(final=True)
    verify()


def release_readme(year):
    terrain = ("\n2022 v2.1 terrain is preserved separately in `../MCFire_Dataset_v2.1_2022_TERRAIN_AUXILIARY/`. It is a post-freeze Copernicus DEM GLO-90 DSM audit and was not a predictor in the inherited frozen benchmark. The core sample table is `splits/D2_S10_GROUP_AWARE_SPLIT_MASTER_2022.csv`; historical S09 columns in it predate the DEM supplement.\n" if year == 2022 else "")
    return f"""# {NAMES[year]}

Current metadata revision: `{REVISION}`. Samples: {EXPECTED[year]}.

The two epoch arrays are int64 Unix seconds reconstructed from the recorded UTC reference and the S07 slot index, cross-checked against the S06 causal slot manifest. Nineteen nominal scan slots run from -180 to 0 min relative to floor(reference UTC / 10 min). They are not offsets from an unrounded first VIIRS detection time.

Labels, sample membership, BT arrays, masks, TRAIN/VAL/TEST assignments, models, predictions, thresholds, metrics and confidence intervals are unchanged. No interpolation or replacement imagery was used.

Tensor CSV `tensor_file` and `tensor_relpath` now resolve relative to this release root, including from split and benchmark CSVs; `tensor_sha256` and `tensor_bytes` describe the current NPZ bytes. Other source paths are historical provenance and may refer to non-distributed intermediates.

`CURRENT_FILE_MANIFEST.json` is the current file inventory/checksum authority. Original S05-S17 freezes, QC documents and evidence ZIPs record historical execution, retain their original bytes, and can contain superseded tensor/index checksums or old absolute paths. They must not be used as current package inventories. The correction report and backups preserve the old-to-new tensor mapping. This metadata repair does not establish label accuracy or absence of unrelated defects.
{terrain}
Core directories: labels (binary labels and unresolved registry), tensors (NPZ and indexes), splits (frozen assignment), benchmark (frozen results), evidence (figures/tables and historical evidence), metadata (descriptions/protocols).
"""


def terrain_readme():
    return """# 2022 v2.1 terrain auxiliary package

Moved intact out of the 2022 core package on 2026-10-16 for a consistent five-year core release. It is not a new dataset run. Join tables to core 2022 samples by sample_id. Historical paths beginning terrain/ are relative to this auxiliary root.

Source: Copernicus DEM GLO-90, AWS public bucket copernicus-dem-90m; DSM, not bare-earth DTM. Nominal horizontal spacing is approximately 90 m. A directly inspected cached tile Copernicus_DSM_COG_30_S29_00_E149_00_DEM.tif is EPSG:4326, 1200 x 1200 pixels, with 0.0008333333333333334-degree (3-arcsec) pixel spacing in both axes. The tile filename COG_30 is not a statement of 30-arcsec spacing. The historical resolution_arcsec=30 metadata value is inconsistent with the actual raster transform and is retained as historical provenance, not physical resolution.

Fields include point elevation (m), 2-km neighbourhood mean/min/max/relief (m), slope (degrees), aspect (clockwise downslope azimuth, degrees), valid fraction and tile provenance. See dem_data_dictionary.json. These were added in a post-freeze 2022 v2.1 DEM/geolocation audit, not used in the inherited S12-S16 benchmark, and do not define new labels, splits or sample exclusions. Other years did not receive this particular supplement.

Original terrain CSV checksum/path columns have been refreshed for the corrected core NPZ. Historical JSON freeze files remain execution provenance. CURRENT_FILE_MANIFEST.json records current bytes.
"""


def verify():
    checks = []
    _, changes = read_csv(OUT / "tensor_hash_changes.csv")
    by_year = {y: {r["sample_id"]: r for r in changes if int(r["year"]) == y} for y in PROJECTS}
    for year in PROJECTS:
        release = release_path(year, final=True)
        backup = OUT / f"backups/{year}/{OLD_NAMES[year]}"
        slots, _, _ = slot_data(year, release)
        tensors = sorted((release / f"tensors/samples/{year}").glob("*.npz"))
        assert len(tensors) == EXPECTED[year]
        def verify_tensor(path):
            current_bytes = path.read_bytes()
            with np.load(io.BytesIO(current_bytes), allow_pickle=False) as z:
                sid = str(z["sample_id"].item())
                ref, times = slots[sid]
                assert z["reference_time_epoch_s"].dtype == np.dtype("int64")
                assert z["slot_time_epoch_s"].dtype == np.dtype("int64")
                assert int(z["reference_time_epoch_s"].item()) == ref
                assert epoch(str(z["reference_time_utc"].item())) == ref
                assert np.array_equal(z["slot_time_epoch_s"], times)
            assert sha(path) == by_year[year][sid]["new_sha256"]
            with zipfile.ZipFile(io.BytesIO(current_bytes)) as current, zipfile.ZipFile(io.BytesIO((backup / path.relative_to(release)).read_bytes())) as original:
                assert current.namelist() == original.namelist()
                for n in current.namelist():
                    if n not in TIME_KEYS:
                        assert current.read(n) == original.read(n), (year, sid, n)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(verify_tensor, tensors))
        allowed = {"tensor_sha256", "tensor_bytes", "tensor_file", "tensor_relpath"}
        for path in release.rglob("*.csv"):
            old = backup / path.relative_to(release)
            fields, new_rows = read_csv(path)
            old_fields, old_rows = read_csv(old)
            assert fields == old_fields and len(new_rows) == len(old_rows)
            for a, b in zip(new_rows, old_rows):
                assert all(a[k] == b[k] for k in fields if k not in allowed), (year, path)
                if "tensor_sha256" in fields and a.get("sample_id") in by_year[year]:
                    assert a["tensor_sha256"] == by_year[year][a["sample_id"]]["new_sha256"]
                if "tensor_file" in fields and a.get("sample_id") in by_year[year]:
                    assert (release / a["tensor_file"]).is_file()
        manifest = json.loads((release / "CURRENT_FILE_MANIFEST.json").read_text(encoding="utf-8"))
        assert manifest["files"] == inventory(release)
        # All historical stage JSON and evidence ZIP/figure files remain byte identical.
        for p in release.rglob("*"):
            if not p.is_file():
                continue
            rp = p.relative_to(release)
            old = backup / rp
            if old.exists() and (p.suffix in {".png", ".zip"} or (p.suffix == ".json" and rp.parts[0] != "metadata")):
                assert sha(p) == sha(old), p
        checks.append(dict(year=year, samples=len(tensors), reference_epoch_pass=len(tensors),
                           slot_epoch_pass=len(tensors) * 19, all_other_npz_members_unchanged=True,
                           labels_splits_and_results_unchanged=True, csv_hashes_and_paths_pass=True,
                           current_manifest_pass=True, release_path=str(release)))
        print(f"FULL VERIFICATION PASS {year}", flush=True)
    write_csv(OUT / "verification_summary.csv", checks)
    write_json(OUT / "COMPLETE.json", {"status": "PASS", "revision": REVISION, "total_samples": sum(r["samples"] for r in checks),
                                      "total_slots": sum(r["slot_epoch_pass"] for r in checks), "years": checks})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted repair after revalidating all original backups")
    args = parser.parse_args()
    if args.verify:
        verify()
    elif args.resume:
        apply(resume=True)
    elif args.apply:
        apply()
    else:
        preflight()
