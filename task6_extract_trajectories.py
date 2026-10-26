"""Compute mask-aware per-slot physical summaries from all released tensors."""
from common import *
import time

def run():
    records=[];checks=[];started=time.time()
    for y in YEARS:
        s=splits(y)
        prot=json.loads(intermediate(y,"S12_TRAIN_VAL_BASELINES","D2_S12_BASELINE_PROTOCOL_{year}.json").read_text(encoding="utf-8"))["protocol"]
        mincore=int(prot["min_core_all_band_valid_pixels"]);minbg=int(prot["min_background_all_band_valid_pixels"])
        for i,row in enumerate(s.itertuples(index=False),1):
            path=RELEASES[y]/row.tensor_relpath
            assert sha(path)==row.tensor_sha256
            with np.load(path,allow_pickle=False) as z:
                bt=z["bt_k"];valid=z["all_band_valid_mask"];slot=z["slot_valid_mask"]
                core=z["core_5km_mask"];bg=z["background_10_30km_mask"];ext=z["extraction_35km_mask"];dist=z["distance_km"]
                offs=z["slot_offset_min"];epochs=z["slot_time_epoch_s"];ref=int(z["reference_time_epoch_s"])
                assert list(z["band_names"])==["B07","B14","B15"]
                assert bt.shape==(19,3,81,81) and np.array_equal(offs,np.arange(-180,1,10))
                assert np.array_equal(valid,np.isfinite(bt).all(axis=1))
                assert np.array_equal(epochs,ref//600*600+offs.astype(np.int64)*60)
                assert not np.isfinite(bt.transpose(0,2,3,1)[~ext]).any()
                pool="FIRE" if row.binary_label==1 else row.source_pool
                for t,off in enumerate(offs):
                    cm=valid[t]&core[t]&ext[t];bm=valid[t]&bg[t]&ext[t]
                    ncore=int(cm.sum());nbg=int(bm.sum());usable=bool(slot[t]) and ncore>=mincore and nbg>=minbg
                    vals=dict(core_b07_p95_k=np.nan,background_b07_median_k=np.nan,core_btd0714_p95_k=np.nan,background_btd0714_median_k=np.nan,contrast_btd0714_k=np.nan)
                    if usable:
                        # Preserve float32 per-pixel BTD subtraction used by frozen extraction.
                        b07=bt[t,0];btd=bt[t,0]-bt[t,1]
                        vals.update(core_b07_p95_k=float(np.percentile(b07[cm].astype(np.float64),95)),background_b07_median_k=float(np.median(b07[bm].astype(np.float64))),core_btd0714_p95_k=float(np.percentile(btd[cm].astype(np.float64),95)),background_btd0714_median_k=float(np.median(btd[bm].astype(np.float64))))
                        vals["contrast_btd0714_k"]=vals["core_btd0714_p95_k"]-vals["background_btd0714_median_k"]
                    records.append(dict(year=y,sample_id=row.sample_id,split_anchor_group_id=row.split_anchor_group_id,pool=pool,split=row.split,offset_min=int(off),nominal_scan_epoch_s=int(epochs[t]),slot_valid=bool(slot[t]),metric_usable=usable,core_common_pixels=ncore,background_common_pixels=nbg,**vals))
            if i%250==0 or i==len(s):print(f"Task6 {y}: {i}/{len(s)} tensors; elapsed {time.time()-started:.1f}s",flush=True)
        checks.append(dict(year=y,tensor_count=len(s),hash_mismatches=0,epoch_mismatches=0,mask_mismatches=0,min_core_pixels=mincore,min_background_pixels=minbg))
    df=pd.DataFrame(records);save(df,"task6_all_sample_trajectories.csv")
    summaries=[]
    for (pool,off),g in df.groupby(["pool","offset_min"]):
        a=g.contrast_btd0714_k.dropna()
        summaries.append(dict(pool=pool,offset_min=off,n_total=len(g),n_valid=len(a),n_missing=len(g)-len(a),median=a.median(),p25=a.quantile(.25),p75=a.quantile(.75),mean=a.mean(),std=a.std(ddof=1)))
    agg=pd.DataFrame(summaries);save(agg,"task6_aggregate_trajectories.csv");save(agg[agg.offset_min.isin([-180,-60,-20,-10,0])],"task6_key_timepoint_values.csv")
    save(pd.DataFrame(checks),"task6_tensor_checks.csv")
    # Independent comparison with cached TRAIN/VAL trajectory features.
    cached_checks=[]
    for y in YEARS:
        cache=read(intermediate(y,"S12_TRAIN_VAL_BASELINES","D2_S12_TRAIN_VAL_FEATURE_MASTER_{year}.csv"))
        g=df[df.year.eq(y)].pivot(index="sample_id",columns="offset_min",values="contrast_btd0714_k")
        for off in range(-180,1,10):
            suffix=f"m{abs(off):03d}" if off<0 else "t000"
            field="t133__delta_btd0714_p95_bgmed__"+suffix
            old=cache.set_index("sample_id")[field].astype(float);new=g.loc[old.index,off]
            a=old.to_numpy();b=new.to_numpy();ok=np.isclose(a,b,rtol=1e-6,atol=1e-6,equal_nan=True)
            cached_checks.append(dict(year=y,offset_min=off,n=len(a),mismatches=int((~ok).sum()),max_abs_difference=float(np.nanmax(np.abs(a-b)))))
    save(pd.DataFrame(cached_checks),"task6_cached_trajectory_crosscheck.csv")
    assert not any(c["mismatches"] for c in cached_checks)
    dump(dict(status="PASS",samples=df.sample_id.nunique(),slots=len(df),seconds=time.time()-started,method="common three-band valid pixels; core P95 minus background median; no interpolation"),"task6_extraction_complete.json")

if __name__=="__main__":run()
