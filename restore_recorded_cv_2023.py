"""Resolve a software-version fold discrepancy using recorded TRAIN fold membership.

No TEST information selects the correction: membership is copied from the original
S14 TRAIN OOF table and validated for complete group isolation. Initial modern-
splitter results are retained in full as a reproduction diagnostic.
"""
from tasks45_benchmarks import *
import shutil

def run():
    y=2023;d=ROOTS[y]/"data/intermediate/S14_COMPACT_SPATIAL_PHYSICS_FUSION"
    oldp=d/f"D2_S14_TRAIN_OOF_PREDICTIONS_{y}.csv"
    old=read(oldp);dev=read(d/f"D2_S14_TRAIN_VAL_SPATIAL_FEATURE_MASTER_{y}.csv")
    train=dev[dev.split.eq("TRAIN")].reset_index(drop=True);val=dev[dev.split.eq("VAL")].reset_index(drop=True)
    orig=old[old.representation.eq("FUSION74")].set_index("sample_id")
    fold=orig.loc[train.sample_id,"cv_fold"].to_numpy(int)
    assert set(fold)==set(range(1,6))
    for rep in old.representation.unique():
        assert np.array_equal(old[old.representation.eq(rep)].set_index("sample_id").loc[train.sample_id,"cv_fold"].to_numpy(int),fold)
    class RecordedStratifiedGroupKFold:
        def __init__(self,n_splits,shuffle,random_state):
            assert n_splits==5 and shuffle and random_state==SEED
        def split(self,X,y,groups):
            assert len(X)==len(fold)
            for f in range(1,6):
                tr=np.flatnonzero(fold!=f);va=np.flatnonzero(fold==f)
                assert not set(groups[tr])&set(groups[va])
                yield tr,va
    helpers=training_helpers(y);helpers["StratifiedGroupKFold"]=RecordedStratifiedGroupKFold
    diag=OUT/"initial_modern_cv_diagnostic"
    if diag.exists():raise RuntimeError("Initial run already archived; do not overwrite")
    diag.mkdir()
    for pattern in ["task5_*.csv","model_details/2023_*"]:
        for p in OUT.glob(pattern):
            target=diag/p.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    spec=json.loads((OUT/"model_details/2023_protocol.json").read_text(encoding="utf-8"))
    spec["cv_execution"]=dict(source=relative(oldp),source_sha256=sha(oldp),method="Replay recorded S14 TRAIN OOF cv_fold assignments; original sklearn 1.3.2 vs current 1.9.1",group_isolation_verified=True)
    dump(spec,"model_details/2023_protocol.json")
    save(pd.DataFrame({"sample_id":train.sample_id,"split_anchor_group_id":train.split_anchor_group_id,"cv_fold":fold}),"model_details/2023_recorded_train_folds.csv")
    # Diagnose fold identity for all years; no retraining outside 2023.
    comparisons=[]
    for yy in YEARS:
        oldyy=read(ROOTS[yy]/f"data/intermediate/S14_COMPACT_SPATIAL_PHYSICS_FUSION/D2_S14_TRAIN_OOF_PREDICTIONS_{yy}.csv")
        newyy=read(OUT/f"model_details/{yy}_FUSION74_train_oof_predictions.csv")
        q=oldyy[oldyy.representation.eq("FUSION74")].merge(newyy,on="sample_id",suffixes=("_old","_new"))
        comparisons.append(dict(year=yy,n=len(q),initial_fold_disagreements=int((q.cv_fold_old!=q.cv_fold_new).sum()),initial_max_oof_probability_difference=float(abs(q.prob_fire_old-q.prob_fire_new).max())))
    save(pd.DataFrame(comparisons),"task5_initial_fold_diagnosis.csv")
    bench=RELEASES[y]/"benchmark";test=read(bench/f"D2_S16_FINAL_TEST_FEATURE_MASTER_{y}.csv")
    frozen=read(bench/f"D2_S16_FINAL_TEST_PREDICTIONS_{y}.csv");fm=read(bench/f"D2_S16_FINAL_TEST_METRICS_{y}.csv").iloc[0]
    res=read(OUT/"task5_ablation_benchmark.csv");res=res[res.year.ne(y)]
    cis=read(OUT/"task5_ablation_ci.csv");cis=cis[cis.year.ne(y)]
    preds=read(OUT/"task5_new_test_predictions.csv");preds=preds[preds.year.ne(y)]
    repro=read(OUT/"task5_fusion74_reproduction_check.csv");repro=repro[repro.year.ne(y)]
    oofcheck=[]
    for rep,cols in spec["features"].items():
        print(f"2023 recorded-CV replay: {rep}",flush=True)
        with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
            result=helpers["_fit_one_representation"](rep,train[cols].to_numpy(float),train.binary_label.to_numpy(int),train.split_anchor_group_id.to_numpy(str),val[cols].to_numpy(float),val.binary_label.to_numpy(int),train,val,spec["source_protocol"])
        th=result["selected_threshold"]
        # Persist the training-only decision before scoring the corrected model.
        dump(dict(C=result["best_c"],class_weight=result["best_class_weight"],threshold=th,selection_partition="TRAIN",fold_source=relative(oldp)),f"model_details/{y}_{rep}_decision_before_test.json")
        prob=result["model"].predict_proba(test[cols].to_numpy(float))[:,1]
        pred=test[["sample_id","split_anchor_group_id","final_label","binary_label","split"]].copy()
        pred["year"]=y;pred["representation"]=rep;pred["prob_fire"]=prob;pred["selected_threshold"]=th;pred["predicted_binary_label"]=(prob>=th).astype(int)
        point,ci,boot,valid=bootstrap(pred,th)
        np.savez_compressed(OUT/f"model_details/{y}_{rep}_bootstrap_draws.npz",metrics=np.array(METRICS),values=boot)
        met=helpers["_probability_metrics"](pred.binary_label.to_numpy(int),prob,th)
        res=pd.concat([res,pd.DataFrame([dict(year=y,representation=rep,n_features=len(cols),C=result["best_c"],class_weight=result["best_class_weight"],**met)])],ignore_index=True)
        cis=pd.concat([cis,pd.DataFrame(ci_rows(y,rep,point,ci,valid,th))],ignore_index=True)
        preds=pd.concat([preds,pred],ignore_index=True)
        for name in ["cv_candidates","cv_fold_rows","threshold_curve","train_oof_predictions","val_predictions"]:save(result[name],f"model_details/{y}_{rep}_{name}.csv")
        dump(dict(C=result["best_c"],class_weight=result["best_class_weight"],threshold=th,train_oof=result["train_oof_metrics"],val=result["val_metrics"],test=met,feature_names=cols,cv="RECORDED_S14_FOLDS"),f"model_details/{y}_{rep}_selected_model.json")
        dump(sorted(set(str(w.message) for w in caught)),f"model_details/{y}_{rep}_warnings.json")
        joblib.dump(result["model"],OUT/f"model_details/{y}_{rep}_model.joblib",compress=3)
        if rep in set(old.representation):
            q=old[old.representation.eq(rep)].merge(result["train_oof_predictions"],on="sample_id",suffixes=("_old","_new"))
            oofcheck.append(dict(representation=rep,fold_mismatches=int((q.cv_fold_old!=q.cv_fold_new).sum()),max_oof_probability_difference=float(abs(q.prob_fire_old-q.prob_fire_new).max()),old_threshold=float(q.selected_threshold_old.iloc[0]),new_threshold=th))
        if rep=="FUSION74":
            q=pred.merge(frozen[["sample_id","prob_fire","predicted_binary_label"]],on="sample_id",suffixes=("","_frozen"))
            repro=pd.concat([repro,pd.DataFrame([dict(year=y,new_C=result["best_c"],frozen_C=float(fm.C),new_class_weight=result["best_class_weight"],frozen_class_weight=fm.class_weight,new_threshold=th,frozen_threshold=float(fm.threshold),max_abs_probability_difference=float(abs(q.prob_fire-q.prob_fire_frozen).max()),prediction_disagreements=int((q.predicted_binary_label!=q.predicted_binary_label_frozen).sum()),max_abs_metric_difference=max(abs(float(met[m])-float(fm[m])) for m in METRICS))])],ignore_index=True)
        print(f"Completed recorded {rep}: C={result['best_c']}, threshold={th}, AUC={point[0]:.6f}",flush=True)
    save(res.sort_values(["year","representation"]),"task5_ablation_benchmark.csv")
    save(cis.sort_values(["year","representation","metric"]),"task5_ablation_ci.csv")
    save(preds.sort_values(["year","representation","sample_id"]),"task5_new_test_predictions.csv")
    save(repro.sort_values("year"),"task5_fusion74_reproduction_check.csv")
    save(pd.DataFrame(oofcheck),"task5_recorded_cv_2023_oof_reproduction.csv")
    # Recompute derived macro summaries only, using saved bootstrap draws.
    draws={(yy,rep):np.load(OUT/f"model_details/{yy}_{rep}_bootstrap_draws.npz")["values"] for yy in YEARS for rep in spec["features"]}
    rng=np.random.default_rng(SEED);perms={yy:rng.permutation(2000) for yy in YEARS};macro=[];gains=[]
    for rep in spec["features"]:
        arr=np.stack([draws[yy,rep][perms[yy]] for yy in YEARS]).mean(axis=0);valid=np.isfinite(arr).all(axis=1)
        mu=res[res.representation.eq(rep)][METRICS].mean().to_numpy(float);ci=np.quantile(arr[valid],[.025,.975],axis=0)
        macro.extend(ci_rows("MACRO_5_YEARS",rep,mu,ci,int(valid.sum()),np.nan))
        if rep!="FUSION74":
            for i,m in enumerate(METRICS):
                diff=res[res.representation.eq("FUSION74")].set_index("year")[m]-res[res.representation.eq(rep)].set_index("year")[m]
                arr=np.stack([(draws[yy,"FUSION74"]-draws[yy,rep])[perms[yy]] for yy in YEARS]).mean(axis=0)[:,i]
                lo,hi=np.nanquantile(arr,[.025,.975]);gains.append(dict(comparator=rep,metric=m,mean_fusion_minus_comparator=diff.mean(),ci_lower=lo,ci_upper=hi,positive_years=int(diff.gt(0).sum()),negative_years=int(diff.lt(0).sum())))
    save(pd.DataFrame(macro),"task5_macro_average_ci.csv");save(pd.DataFrame(gains),"task5_fusion_mean_improvements.csv")
    dump(dict(status="PASS",correction="Replay historical TRAIN folds",source_path=relative(oldp),source_sha256=sha(oldp),test_results_used_to_select_correction=False,initial_run_retained=str(diag.relative_to(OUT)),note="Initial 2023 modern-splitter models were evaluated before detecting fold-version discrepancy; final four 2023 fits replay original folds, with both runs disclosed."),"task5_recorded_cv_correction.json")

if __name__=="__main__":run()
