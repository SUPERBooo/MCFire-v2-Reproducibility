"""Independent cross-checks and compact provenance exports; no original writes."""
from common import *
import ast, zipfile, warnings
from PIL import Image
from pypdf import PdfReader

def run():
    findings=[]
    def check(name,condition,detail):
        findings.append(dict(check=name,status="PASS" if condition else "FAIL",detail=str(detail)))
        assert condition,(name,detail)
    meta=read(OUT/"ready_sample_metadata_all_years.csv")
    funnel=read(OUT/"task1_candidate_funnel.csv")
    check("cohort_size",len(meta)==6662,len(meta))
    check("sample_id_global_uniqueness",meta.sample_id.nunique()==6662,meta.sample_id.nunique())
    check("class_counts",meta.pool.value_counts().to_dict()=={"FIRE":1824,"NEG_LOCAL_MATCHED":2099,"NEG_GLOBAL_STRATIFIED":2739},meta.pool.value_counts().to_dict())
    optical=read(OUT/"task2_fire_optical_records.csv")
    check("S05_FIRE_optical_count",len(optical)==1977,len(optical))
    check("ready_FIRE_optical_count",int(optical.ready.sum())==1824,optical.ready.sum())
    linkage=read(OUT/"linkage_checks.csv")
    check("linkage_checks",linkage[["unmatched_ready","duplicate_sample_ids","label_conflicts","unassigned_split","groups_in_multiple_splits"]].to_numpy().sum()==0,linkage.to_dict("records"))
    pred=read(OUT/"task5_new_test_predictions.csv")
    check("ablation_predictions",len(pred)==4*988,len(pred))
    check("prediction_keys_unique",not pred.duplicated(["year","representation","sample_id"]).any(),len(pred))
    repro=read(OUT/"task5_fusion74_reproduction_check.csv")
    check("FUSION74_thresholds",np.allclose(repro.new_threshold,repro.frozen_threshold),repro[["year","new_threshold","frozen_threshold"]].to_dict("records"))
    check("FUSION74_test_classes",int(repro.prediction_disagreements.sum())==0,int(repro.prediction_disagreements.sum()))
    check("FUSION74_test_metrics",float(repro.max_abs_metric_difference.max())<1e-12,float(repro.max_abs_metric_difference.max()))
    ci=read(OUT/"task5_ablation_ci.csv")
    check("annual_ci_rows",len(ci)==120,len(ci))
    check("bootstrap_validity",ci.valid_resamples.ge(1600).all(),ci.valid_resamples.min())
    check("interval_order",(ci.ci_lower<=ci.ci_upper).all(),len(ci))
    # Direct manuscript Table 7 vs source metrics, rounded exactly to published precision.
    rows=[]
    for fn in ["MCFire_Scientific_Data_Revised_reference.json","MCFire_Scientific_Data_Revised_v2_reference.json"]:
        doc=json.loads((OUT/"reference"/fn).read_text(encoding="utf-8"))
        for row in doc["benchmark_tables"][0][1:]:
            y=int(row[0]);m=read(RELEASES[y]/f"benchmark/D2_S16_FINAL_TEST_METRICS_{y}.csv").iloc[0]
            cols=[("n",1,0),("threshold",2,3),("accuracy",3,4),("balanced_accuracy",4,4),("roc_auc",5,4),("average_precision",6,4),("mcc",7,4)]
            for field,col,decimals in cols:
                actual=f"{float(m[field]):.{decimals}f}";reported=row[col]
                rows.append(dict(manuscript=doc["source"],year=y,metric=field,manuscript_value=reported,frozen_value=actual,match=actual==reported))
    save(pd.DataFrame(rows),"table7_frozen_results_comparison.csv")
    check("Table7_consistency",all(r["match"] for r in rows),f"{len(rows)} values across two manuscript files")
    # Distinguish frozen S10 diagnostic values from 2022 post-freeze DEM audit.
    aux=read(AUX/"terrain/D2_S09_DEM_PARALLAX_GEOLOCATION_SAMPLE_AUDIT_2022.csv")
    q=splits(2022).merge(aux,on="sample_id",suffixes=("_split","_terrain"),validate="one_to_one")
    diags=[]
    for field in ["candidate_onset_detected","persistent_onset_detected","possible_spatial_mismatch","possible_parallax_risk"]:
        for pool in POOLS+["ALL"]:
            g=q if pool=="ALL" else q[(q.binary_label_split.eq(1)) if pool=="FIRE" else q.source_pool_split.eq(pool)]
            a=g[field+"_split"].astype(str).str.lower().eq("true");b=g[field+"_terrain"].astype(str).str.lower().eq("true")
            diags.append(dict(year=2022,pool=pool,field=field,n=len(g),split_true=int(a.sum()),terrain_true=int(b.sum()),different_samples=int((a!=b).sum())))
    save(pd.DataFrame(diags),"task3_2022_frozen_vs_terrain_flags.csv")
    # Schema normalization audit: no inferred mappings across annual files.
    schema=[]
    for year in YEARS:
        for logical,p in [("S05_labels",one(RELEASES[year]/"labels",f"D2_S05_FINAL_BINARY_MASTER_{year}.csv")),("S10_splits",one(RELEASES[year]/"splits",f"D2_S10_GROUP_AWARE_SPLIT_MASTER_{year}.csv")),("S02_optics",intermediate(year,"S02_INDEPENDENT_EVIDENCE","D2_S02_INDEPENDENT_EVIDENCE_MASTER_{year}.csv"))]:
            d=read(p,nrows=3)
            schema.append(dict(year=year,role=logical,source=relative(p),columns=list(d.columns),pool_mapping="final_label FIRE / binary_label 1 maps source_pool INDEPENDENT_STRICT to displayed FIRE; control names unchanged"))
    dump(schema,"schema_alignment.json")
    # Compact evidence excerpts are copied as data, with line ranges and SHA256.
    snippets=[("src/train_val_baselines.py",341,449),("src/compact_spatial_physics_fusion.py",208,349),("src/sentinel2_evidence.py",1255,1349),("src/dem_parallax_geolocation_qc.py",510,635),("src/candidate_audit.py",53,215),("src/one_time_final_test_evaluation.py",230,335)]
    evidence=[]
    for rel,start,end in snippets:
        p=WORK/rel;lines=p.read_text(encoding="utf-8").splitlines()
        out=OUT/"code_evidence"/(Path(rel).stem+"_excerpt.txt")
        out.write_text(f"Source: {relative(p)}\nSHA256: {sha(p)}\nLines {start}-{end}\n\n"+"\n".join(f"{i}: {lines[i-1]}" for i in range(start,min(end,len(lines))+1)),encoding="utf-8")
        evidence.append(dict(source=relative(p),sha256=sha(p),first_line=start,last_line=end,export=out.relative_to(OUT).as_posix(),status="CURRENT_CODE_READ_NOT_AUTOMATIC_PROOF_OF_HISTORICAL_DEFAULTS"))
    dump(evidence,"code_evidence_index.json")
    trac=read(OUT/"task6_all_sample_trajectories.csv")
    check("trajectory_rows",len(trac)==6662*19,len(trac))
    check("trajectory_keys_unique",not trac.duplicated(["sample_id","offset_min"]).any(),len(trac))
    cross=read(OUT/"task6_cached_trajectory_crosscheck.csv")
    check("cached_trajectory_crosscheck",cross.mismatches.sum()==0,f"{int(cross.n.sum())} slot metrics; maximum absolute difference {cross.max_abs_difference.max()}")
    im=Image.open(OUT/"Figure5_Thermal_Showcase.png")
    check("PNG_300dpi",abs(im.info["dpi"][0]-300)<.1 and im.size==(2160,2670),dict(dpi=im.info["dpi"],size=im.size))
    pdf=PdfReader(OUT/"Figure5_Thermal_Showcase.pdf")
    check("PDF_single_page",len(pdf.pages)==1,len(pdf.pages))
    images=[]
    for name,obj in pdf.pages[0].get("/Resources",{}).get("/XObject",{}).items():
        if obj.get_object().get("/Subtype")=="/Image":images.append(str(name))
    check("PDF_vector_no_embedded_raster",not images,images)
    # Export current figures' sample choices and actual optical evidence for the exemplar.
    example=optical[optical.sample_id.eq("V22_000145")]
    save(example,"task2_showcase_optical_example.csv")
    save(pd.DataFrame(findings),"supplement_validation_checks.csv")
    dump(dict(status="PASS",checks=len(findings),all_pass=True),"supplement_validation_result.json")
    print(f"{len(findings)} supplement cross-checks PASS",flush=True)

if __name__=="__main__":run()
