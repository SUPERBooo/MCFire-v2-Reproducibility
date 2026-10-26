"""Assemble small result archive and traceable manifests, never changing sources."""
from common import *
import ast, shutil, zipfile, platform

def run():
    dump(dict(python=sys.version,executable=sys.executable,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()),"analysis_runtime.json")
    extra=[]
    for y in YEARS:
        src=WORK/f"release_metadata_correction_20261016/candidate_rules_config_{y}.yaml"
        target=OUT/f"code_evidence/candidate_rules_config_{y}.yaml"
        shutil.copy2(src,target);extra.append(dict(path=relative(src),bytes=src.stat().st_size,sha256=sha(src)))
        for p in [ROOTS[y]/f"data/intermediate/S12_TRAIN_VAL_BASELINES/D2_S12_BASELINE_PROTOCOL_{y}.json",ROOTS[y]/f"src/train_val_baselines.py",ROOTS[y]/f"src/compact_spatial_physics_fusion.py",ROOTS[y]/f"data/intermediate/S14_COMPACT_SPATIAL_PHYSICS_FUSION/D2_S14_TRAIN_OOF_PREDICTIONS_{y}.csv"]:
            extra.append(dict(path=relative(p),bytes=p.stat().st_size,sha256=sha(p)))
    terrain=WORK/"release_metadata_correction_20261016/terrain_geotransform_evidence.json"
    shutil.copy2(terrain,OUT/"code_evidence/terrain_geotransform_prior_verification.json")
    extra.append(dict(path=relative(terrain),bytes=terrain.stat().st_size,sha256=sha(terrain)))
    source=pd.concat([read(OUT/"statistics_source_sha256.csv"),read(OUT/"benchmark_source_sha256.csv"),pd.DataFrame(extra)],ignore_index=True).drop_duplicates("path")
    assert all(sha(BASE/r.path)==r.sha256 for r in source.itertuples(index=False)),"A source changed after read"
    save(source,"source_manifest.csv")
    # Source code syntax check without generating bytecode.
    for p in OUT.glob("*.py"):ast.parse(p.read_text(encoding="utf-8"),filename=str(p))
    integrity=json.loads((OUT/"protected_integrity_result.json").read_text())
    validation=json.loads((OUT/"supplement_validation_result.json").read_text())
    assert integrity["status"]==validation["status"]=="PASS"
    report=OUT/"MCFire_Paper_Supplement_Report.md"
    text=report.read_text(encoding="utf-8")
    assert all(f"## {i}." in text for i in range(1,7))
    assert text.count("| ")>400
    metrics=read(OUT/"task5_ablation_benchmark.csv",dtype={"year":str})
    assert len(metrics)==24 and len(metrics[metrics.year.eq("MACRO_5_YEARS")])==4
    assert all(metrics[c].notna().all() for c in metrics if c.endswith("_ci_lower") or c.endswith("_ci_upper"))
    dump(dict(status="PASS",date="2026-10-16",six_tasks_completed=True,source_files_unchanged=integrity["differences"]==0,protected_file_count=integrity["files_before"],supplement_checks=validation["checks"],annual_final_models=20,initial_2023_diagnostic_models=4,ready_samples=6662,slots=126578,source_files_manifest_count=len(source),report_sha256=sha(report),figure_png_sha256=sha(OUT/"Figure5_Thermal_Showcase.png"),figure_pdf_sha256=sha(OUT/"Figure5_Thermal_Showcase.pdf")),"COMPLETE.json")
    archive=OUT/"MCFire_Paper_Supplement_20261016.zip"
    exclude={"Figure5_preview.png","Figure5_pdf_render.png","deliverables_manifest.csv","archive_verification.json",archive.name}
    paths=sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in exclude and ".matplotlib" not in p.parts and "__pycache__" not in p.parts and not p.name.endswith("_media_contact_sheet.png"))
    manifest=pd.DataFrame([dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in paths])
    save(manifest,"deliverables_manifest.csv")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[OUT/"deliverables_manifest.csv"]:z.write(p,p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert z.read(report.name)==report.read_bytes()
        for row in manifest.itertuples(index=False):
            assert hashlib.sha256(z.read(row.path)).hexdigest()==row.sha256,row.path
    dump(dict(status="PASS",archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),members=len(paths)+1,all_member_hashes_match=True),"archive_verification.json")
    print(json.dumps(dict(status="PASS",files=len(paths)+1,archive_bytes=archive.stat().st_size,report_bytes=report.stat().st_size)),flush=True)

if __name__=="__main__":run()
