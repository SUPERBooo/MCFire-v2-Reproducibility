"""Retrospective supplementary analyses. Never calls original stage runners."""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
from common import *
import ast, platform, warnings
import joblib, sklearn, scipy
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, confusion_matrix
from threadpoolctl import threadpool_limits

METRICS=["roc_auc","average_precision","balanced_accuracy","mcc","recall_sensitivity","specificity"]
SOURCES=[]

def tracked(p):
    SOURCES.append(dict(path=relative(p),sha256=sha(p),bytes=p.stat().st_size))
    return p

def training_helpers(y):
    """Extract ONLY pure training helpers, avoiding source writes and import caches."""
    p=tracked(ROOTS[y]/"src/train_val_baselines.py")
    tree=ast.parse(p.read_text(encoding="utf-8"))
    names={"_build_estimator","_probability_metrics","_select_threshold","_fit_one_representation","S12Error"}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    code='from __future__ import annotations\nfrom typing import *\nimport numpy as np\nimport pandas as pd\nfrom sklearn.impute import SimpleImputer\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import *\nfrom sklearn.model_selection import StratifiedGroupKFold, GroupKFold\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import StandardScaler\n\n'+ '\n\n'.join(ast.get_source_segment(p.read_text(encoding="utf-8"),n) for n in nodes)
    (OUT/"code_evidence").mkdir(exist_ok=True)
    (OUT/f"code_evidence/training_helpers_{y}.py").write_text(code,encoding="utf-8")
    env={};exec(compile(code,str(p),"exec"),env)
    return env

def fast_metrics(y,p,threshold):
    """Six standard binary metrics; tie-group AP matches sklearn, including bootstrap ties."""
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pos=y==1;pred=p>=threshold
    npos=int(pos.sum());nneg=len(y)-npos
    if not npos or not nneg:raise ValueError("single class")
    tp=int((pred&pos).sum());tn=int((~pred&~pos).sum());fp=nneg-tn;fn=npos-tp
    order=np.argsort(-p,kind="mergesort");ys=y[order];ps=p[order]
    ends=np.r_[np.where(np.diff(ps)!=0)[0],len(y)-1]
    tps=np.cumsum(ys)[ends];fps=(ends+1)-tps
    recalls=tps/npos;precision=tps/(ends+1)
    ap=np.sum(np.diff(np.r_[0.,recalls])*precision)
    auc=np.trapezoid(np.r_[0.,recalls],np.r_[0.,fps/nneg])
    denom=float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return np.array([auc,ap,.5*(tp/npos+tn/nneg),(tp*tn-fp*fn)/np.sqrt(denom) if denom else 0.,tp/npos,tn/nneg])

def bootstrap(df,threshold,seed=SEED):
    df=df.reset_index(drop=True)
    groups=df.split_anchor_group_id.astype(str).unique()
    index={g:np.flatnonzero(df.split_anchor_group_id.astype(str).to_numpy()==g) for g in groups}
    rng=np.random.default_rng(seed);y=df.binary_label.to_numpy(int);p=df.prob_fire.to_numpy(float)
    out=np.full((2000,len(METRICS)),np.nan)
    for b in range(2000):
        inds=np.concatenate([index[g] for g in rng.choice(groups,size=len(groups),replace=True)])
        if len(np.unique(y[inds]))<2:continue
        out[b]=fast_metrics(y[inds],p[inds],threshold)
    valid=np.isfinite(out).all(axis=1)
    assert valid.sum()>=1600
    point=fast_metrics(y,p,threshold)
    # Independent sklearn cross-check for originals and tied bootstrap samples.
    checks=[np.arange(len(y)),np.r_[np.arange(len(y)),np.arange(min(8,len(y)))]]
    for idx in checks:
        v=fast_metrics(y[idx],p[idx],threshold)
        assert np.allclose(v[:4],[roc_auc_score(y[idx],p[idx]),average_precision_score(y[idx],p[idx]),balanced_accuracy_score(y[idx],p[idx]>=threshold),matthews_corrcoef(y[idx],p[idx]>=threshold)],atol=1e-12)
    ci=np.quantile(out[valid],[.025,.975],axis=0)
    return point,ci,out,int(valid.sum())

def ci_rows(year,rep,point,ci,valid,threshold,**meta):
    return [dict(year=year,representation=rep,metric=m,point_estimate=point[i],ci_lower=ci[0,i],ci_upper=ci[1,i],
                 valid_resamples=valid,requested_resamples=2000,seed=SEED,threshold=threshold,**meta) for i,m in enumerate(METRICS)]

def run():
    dump(dict(python=sys.version,executable=sys.executable,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,platform=platform.platform()),"benchmark_runtime.json")
    control_rows=[];control_ci=[];result_rows=[];all_ci=[];all_predictions=[];reproduction=[];alignment=[];draws={}
    for y in YEARS:
        bench=RELEASES[y]/"benchmark"
        frozen=read(tracked(bench/f"D2_S16_FINAL_TEST_PREDICTIONS_{y}.csv"))
        fm=read(tracked(bench/f"D2_S16_FINAL_TEST_METRICS_{y}.csv")).iloc[0]
        spl=splits(y);assert not spl.sample_id.duplicated().any()
        pool=spl[["sample_id","source_pool"]].copy();pool["pool"]=np.where(pool.source_pool.eq("INDEPENDENT_STRICT"),"FIRE",pool.source_pool)
        frozen=frozen.merge(pool,on="sample_id",validate="one_to_one")
        assert set(frozen.sample_id)==set(spl[spl.split.eq("TEST")].sample_id)
        threshold=float(fm.threshold)
        assert np.allclose(frozen.frozen_threshold,threshold)
        assert np.array_equal(frozen.predicted_binary_label,(frozen.prob_fire>=threshold).astype(int))
        counts=frozen.pool.value_counts()
        for j,neg in enumerate(POOLS[1:]):
            sub=frozen[frozen.pool.isin(["FIRE",neg])].reset_index(drop=True)
            nneg=int(sub.binary_label.eq(0).sum());negpred=sub[sub.binary_label.eq(0)].predicted_binary_label
            fp=int(negpred.sum());tn=nneg-fp
            point,ci,_,valid=bootstrap(sub,threshold)
            control_rows.append(dict(year=y,control_pool=neg,threshold=threshold,test_fire=int(counts.get("FIRE",0)),test_local=int(counts.get(POOLS[1],0)),test_regional=int(counts.get(POOLS[2],0)),tn=tn,fp=fp,specificity=tn/nneg,fpr=fp/nneg,roc_auc=point[0],roc_auc_ci_lower=ci[0,0],roc_auc_ci_upper=ci[1,0],average_precision=point[1],average_precision_ci_lower=ci[0,1],average_precision_ci_upper=ci[1,1],valid_resamples=valid,n_groups=sub.split_anchor_group_id.nunique()))
            control_ci.extend(ci_rows(y,neg,point,ci,valid,threshold))
        # Use recorded per-year feature order and protocol, never column-name guesses alone.
        d=ROOTS[y]/"data/intermediate/S14_COMPACT_SPATIAL_PHYSICS_FUSION"
        schema=json.loads(tracked(d/f"D2_S14_FEATURE_SCHEMA_{y}.json").read_text(encoding="utf-8"))
        protocol=json.loads(tracked(d/f"D2_S14_FUSION_PROTOCOL_{y}.json").read_text(encoding="utf-8"))["protocol"]
        dev=read(tracked(d/f"D2_S14_TRAIN_VAL_SPATIAL_FEATURE_MASTER_{y}.csv"))
        test=read(tracked(bench/f"D2_S16_FINAL_TEST_FEATURE_MASTER_{y}.csv"))
        fused=schema["representations"]["FUSION74"]["feature_names"]
        spatial=schema["representations"]["SPATIAL48"]["feature_names"]
        compact=[c for c in fused if c not in spatial]
        t0=[c for c in spatial if c.endswith("__t000")]
        reps={"T0_SPATIAL16":t0,"COMPACT26":compact,"SPATIAL48":spatial,"FUSION74":fused}
        assert [len(v) for v in reps.values()]==[16,26,48,74]
        dump(dict(year=y,features=reps,source_protocol=protocol,analysis_role="RETROSPECTIVE_SUPPLEMENT_NOT_ORIGINAL_FROZEN_BENCHMARK",selection_uses_test=False,bootstrap=dict(seed=SEED,draws=2000,unit="split_anchor_group_id",method="percentile",single_class="skip",average_precision="tie-group precision-recall integral, sklearn-compatible")),f"model_details/{y}_protocol.json")
        joined=pd.concat([dev,test],ignore_index=True)
        assert joined.sample_id.nunique()==len(joined)==len(spl)
        check=joined.merge(spl,on="sample_id",suffixes=("","_split"),validate="one_to_one")
        for col in ["split_anchor_group_id","binary_label","split"]:assert check[col].astype(str).equals(check[col+"_split"].astype(str))
        group_counts=spl.groupby("split_anchor_group_id").split.nunique();assert group_counts.max()==1
        train=dev[dev.split.eq("TRAIN")].reset_index(drop=True);val=dev[dev.split.eq("VAL")].reset_index(drop=True)
        assert set(dev.split)=={"TRAIN","VAL"} and test.split.eq("TEST").all()
        alignment.append(dict(year=y,train=len(train),val=len(val),test=len(test),feature_rows=len(joined),id_label_group_split_mismatches=0))
        helpers=training_helpers(y)
        for rep,cols in reps.items():
            print(f"TASK5 {y} {rep}: TRAIN-only selection and one TEST scoring",flush=True)
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter("always")
                result=helpers["_fit_one_representation"](rep,train[cols].to_numpy(float),train.binary_label.to_numpy(int),train.split_anchor_group_id.to_numpy(str),val[cols].to_numpy(float),val.binary_label.to_numpy(int),train,val,protocol)
            dump(sorted(set(str(w.message) for w in caught)),f"model_details/{y}_{rep}_warnings.json")
            th=result["selected_threshold"]
            prob=result["model"].predict_proba(test[cols].to_numpy(float))[:,1]
            pred=test[["sample_id","split_anchor_group_id","final_label","binary_label","split"]].copy()
            pred["year"]=y;pred["representation"]=rep;pred["prob_fire"]=prob;pred["selected_threshold"]=th;pred["predicted_binary_label"]=(prob>=th).astype(int)
            all_predictions.append(pred)
            point,ci,boot,valid=bootstrap(pred,th);draws[y,rep]=boot
            np.savez_compressed(OUT/f"model_details/{y}_{rep}_bootstrap_draws.npz",metrics=np.array(METRICS),values=boot)
            met=helpers["_probability_metrics"](pred.binary_label.to_numpy(int),prob,th)
            result_rows.append(dict(year=y,representation=rep,n_features=len(cols),C=result["best_c"],class_weight=result["best_class_weight"],**met))
            all_ci.extend(ci_rows(y,rep,point,ci,valid,th))
            for name in ["cv_candidates","cv_fold_rows","threshold_curve","train_oof_predictions","val_predictions"]:
                save(result[name],f"model_details/{y}_{rep}_{name}.csv")
            dump(dict(C=result["best_c"],class_weight=result["best_class_weight"],threshold=th,train_oof=result["train_oof_metrics"],val=result["val_metrics"],test=met,feature_names=cols),f"model_details/{y}_{rep}_selected_model.json")
            joblib.dump(result["model"],OUT/f"model_details/{y}_{rep}_model.joblib",compress=3)
            if rep=="FUSION74":
                pp=pred.merge(frozen[["sample_id","prob_fire","predicted_binary_label"]],on="sample_id",suffixes=("","_frozen"),validate="one_to_one")
                reproduction.append(dict(year=y,new_C=result["best_c"],frozen_C=float(fm.C),new_class_weight=result["best_class_weight"],frozen_class_weight=fm.class_weight,new_threshold=th,frozen_threshold=threshold,max_abs_probability_difference=float(np.abs(pp.prob_fire-pp.prob_fire_frozen).max()),prediction_disagreements=int((pp.predicted_binary_label!=pp.predicted_binary_label_frozen).sum()),max_abs_metric_difference=max(abs(float(met[m])-float(fm[m])) for m in METRICS)))
            print(f"Completed {y}/{rep}: C={result['best_c']}, weight={result['best_class_weight']}, threshold={th}, AUC={point[0]:.6f}",flush=True)
        # Checkpoints remain inside output; original frozen results never touched.
        save(pd.DataFrame(result_rows),"task5_ablation_benchmark.csv")
        save(pd.DataFrame(all_ci),"task5_ablation_ci.csv")
        save(pd.DataFrame(control_rows),"task4_control_pool_specificity.csv")
        save(pd.DataFrame(reproduction),"task5_fusion74_reproduction_check.csv")
    save(pd.DataFrame(control_ci),"task4_control_pool_all_metric_ci.csv")
    save(pd.concat(all_predictions,ignore_index=True),"task5_new_test_predictions.csv")
    save(pd.DataFrame(alignment),"task5_feature_alignment_checks.csv")
    save(pd.DataFrame(SOURCES).drop_duplicates("path"),"benchmark_source_sha256.csv")
    # Equal-year macro mean: independent annual bootstrap streams, no pooled generalization claim.
    results=pd.DataFrame(result_rows);macro=[];gains=[]
    rng=np.random.default_rng(SEED)
    permutations={y:rng.permutation(2000) for y in YEARS}
    for rep in reps:
        arr=np.stack([draws[y,rep][permutations[y]] for y in YEARS]).mean(axis=0)
        valid=np.isfinite(arr).all(axis=1)
        mu=results[results.representation.eq(rep)][METRICS].mean().to_numpy(float)
        ci=np.quantile(arr[valid],[.025,.975],axis=0)
        macro.extend(ci_rows("MACRO_5_YEARS",rep,mu,ci,int(valid.sum()),np.nan))
        if rep!="FUSION74":
            for i,m in enumerate(METRICS):
                diff=results[results.representation.eq("FUSION74")].set_index("year")[m]-results[results.representation.eq(rep)].set_index("year")[m]
                gainboot=np.stack([(draws[y,"FUSION74"]-draws[y,rep])[permutations[y]] for y in YEARS]).mean(axis=0)[:,i]
                lo,hi=np.nanquantile(gainboot,[.025,.975])
                gains.append(dict(comparator=rep,metric=m,mean_fusion_minus_comparator=diff.mean(),ci_lower=lo,ci_upper=hi,positive_years=int(diff.gt(0).sum()),negative_years=int(diff.lt(0).sum())))
    save(pd.DataFrame(macro),"task5_macro_average_ci.csv")
    save(pd.DataFrame(gains),"task5_fusion_mean_improvements.csv")
    dump(dict(status="PASS",models=20,seed=SEED,resamples=2000,ci_method="percentile",new_results="retrospective supplementary analyses"),"tasks45_complete.json")

if __name__=="__main__":run()
