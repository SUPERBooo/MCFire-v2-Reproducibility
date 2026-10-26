"""Source-preserving candidate, optical evidence and domain summaries."""
from common import *

SOURCES=[]
def load(p):
    SOURCES.append(dict(path=relative(p),bytes=p.stat().st_size,sha256=sha(p)))
    return read(p)

def stats(df, field, **meta):
    a=pd.to_numeric(df[field],errors="coerce") if field in df else pd.Series(np.nan,index=df.index)
    a=a[np.isfinite(a)]
    return dict(**meta,field=field,n_total=len(df),n_valid=len(a),n_missing=len(df)-len(a),
                mean=a.mean(),std=a.std(ddof=1),median=a.median(),p25=a.quantile(.25),p75=a.quantile(.75),
                iqr=a.quantile(.75)-a.quantile(.25),minimum=a.min(),maximum=a.max())

def boolean(s):
    return s.astype(str).str.lower().map({"true":True,"false":False,"1":True,"0":False})

def run():
    funnels=[]; tiers=[]; statuses=[]; optical=[]; cohorts=[]; joins=[]; controls=[]
    for y in YEARS:
        p=lambda stage,name: intermediate(y,stage,name+"_{year}.csv")
        pix=load(p("S01_VIIRS","D2_S01_VIIRS_PIXELS"))
        evt=load(p("S01_VIIRS","D2_S01_VIIRS_CANDIDATE_EVENTS"))
        aud=load(p("S01_5_CANDIDATE_AUDIT","D2_S015_EVENT_AUDIT_MASTER"))
        queue=load(p("S02_INDEPENDENT_EVIDENCE","D2_S02_INPUT_QUEUE"))
        ev=load(p("S02_INDEPENDENT_EVIDENCE","D2_S02_INDEPENDENT_EVIDENCE_MASTER"))
        dec=load(p("S03_EVIDENCE_FUSION","D2_S03_INDEPENDENT_LABEL_MASTER"))
        lab=load(one(RELEASES[y]/"labels",f"D2_S05_FINAL_BINARY_MASTER_{y}.csv"))
        spl=load(one(RELEASES[y]/"splits",f"D2_S10_GROUP_AWARE_SPLIT_MASTER_{y}.csv"))
        for frame,key in [(pix,"detection_id"),(evt,"event_id"),(aud,"event_id"),(queue,"event_id"),(ev,"event_id"),(dec,"event_id"),(lab,"sample_id"),(spl,"sample_id")]:
            assert not frame[key].duplicated().any(), (y,key)
        assert set(evt.event_id)==set(aud.event_id)
        assert set(queue.event_id)==set(ev.event_id)==set(dec.event_id)
        assert set(spl.sample_id).issubset(set(lab.sample_id))
        joined=spl.merge(lab,on="sample_id",suffixes=("","_label"),validate="one_to_one")
        for col in ["split_anchor_group_id","binary_label","final_label","source_pool","reference_time_utc"]:
            assert joined[col].astype(str).equals(joined[col+"_label"].astype(str)), (y,col)
        duplicated_fields=[c+"_label" for c in lab.columns if c in spl.columns and c!="sample_id"]
        joined=joined.drop(columns=duplicated_fields).copy()
        joined["pool"]=np.where(joined.binary_label.eq(1),"FIRE",joined.source_pool)
        joined["year"]=y
        assert joined.pool.isin(POOLS).all()
        cohorts.append(joined)
        joins.append(dict(year=y,s05_n=len(lab),ready_n=len(spl),unmatched_ready=0,duplicate_sample_ids=0,label_conflicts=0,
                          unassigned_split=int((~spl.split.isin(["TRAIN","VAL","TEST"])).sum()),
                          groups_in_multiple_splits=int((spl.groupby("split_anchor_group_id").split.nunique()>1).sum())))
        held=aud.candidate_disposition.eq("HOLD_SCIENTIFIC_RISK")
        assert held.equals(aud.candidate_priority_tier.eq("HOLD"))
        ph=aud.persistent_source_risk.eq("HIGH"); ch=aud.chaining_risk.eq("HIGH")
        assert held.equals(ph|ch)
        row=dict(year=y,qc_viirs_detections=len(pix),candidate_events=len(evt),hold_total=int(held.sum()),
                 hold_persistent_only=int((ph&~ch).sum()),hold_chaining_only=int((~ph&ch).sum()),hold_both=int((ph&ch).sum()),
                 moderate_nonheld=int((~held&(aud.persistent_source_risk.eq("MODERATE")|aud.chaining_risk.eq("MODERATE"))).sum()),
                 eligible_nonheld=int((~held).sum()),optical_queue=len(queue),
                 s03_strict=int(dec.s03_independent_label_tier.eq("INDEPENDENT_STRICT").sum()),
                 s05_fire=int(lab.binary_label.eq(1).sum()),ready_fire=int(spl.binary_label.eq(1).sum()))
        for t in ["P1","P2","P3"]: row[t]=int(aud.candidate_priority_tier.eq(t).sum())
        assert row["P1"]+row["P2"]+row["P3"]==row["eligible_nonheld"]
        funnels.append(row)
        for field,out in [("s03_independent_label_tier",tiers),("s03_evidence_status",statuses)]:
            for k,n in dec[field].fillna("MISSING_FIELD").value_counts().items():
                out.append(dict(year=y,level=k,n=int(n),denominator=len(dec),percent=100*n/len(dec)))
        fire=lab[lab.binary_label.eq(1)].copy()
        fire=fire.merge(ev,left_on="source_event_id",right_on="event_id",how="left",validate="one_to_one",indicator=True,suffixes=("","_s02"))
        assert fire._merge.eq("both").all()
        fire=fire.drop(columns="_merge")
        fire["year"]=y
        fire["ready"]=fire.sample_id.isin(spl.sample_id)
        fire["any_selected_post_after_primary_end"]=fire.apply(lambda r: any(pd.Timestamp(d)>=pd.Timestamp(r.post_primary_end) for d in str(r.post_dates).split("|") if d and d!="nan"),axis=1)
        optical.append(fire)
        ctl=load(one(RELEASES[y]/"labels",f"D2_S05_FINAL_NON_FIRE_CONTROLS_{y}.csv"))
        ctl=ctl[ctl.s05_sample_id.isin(joined[joined.pool.eq("NEG_LOCAL_MATCHED")].sample_id)].copy()
        ctl["year"]=y
        assert len(ctl)==int(joined.pool.eq("NEG_LOCAL_MATCHED").sum())
        controls.append(ctl)
        print(f"Tasks 1-3 {y}: {len(pix)} detections, {len(evt)} events, {len(joined)} ready",flush=True)
    f=pd.DataFrame(funnels)
    f=pd.concat([f,pd.DataFrame([dict(year="ALL",**f.drop(columns="year").sum().to_dict())])],ignore_index=True)
    save(f,"task1_candidate_funnel.csv")
    for rows,name in [(tiers,"task1_optical_tiers.csv"),(statuses,"task1_optical_statuses.csv")]:
        df=pd.DataFrame(rows)
        a=df.groupby("level",as_index=False).n.sum(); a["year"]="ALL";a["denominator"]=int(df.groupby("year").denominator.first().sum());a["percent"]=100*a.n/a.denominator
        save(pd.concat([df,a],ignore_index=True),name)
    optics=pd.concat(optical,ignore_index=True)
    save(optics,"task2_fire_optical_records.csv")
    fields=["pre_usable_count","post_usable_count","pre_clear_fraction_mean","post_clear_fraction_mean",
            "adaptive_threshold_dNBR","adaptive_threshold_dNDVI","adaptive_threshold_dMIRBI",
            "viirs_prior_dNBR_median","viirs_prior_dNBR_max","viirs_prior_dNDVI_median","viirs_prior_dNDVI_max",
            "viirs_prior_dMIRBI_median","viirs_prior_dMIRBI_max","core500_dNBR_median","core500_dNDVI_median","core500_dMIRBI_median",
            "max_persistence_count_viirs_prior","median_persistence_count_viirs_prior",
            "viirs_supported_connected_dnbr_anchor_pixel_count","connected_dnbr_anchor_pixel_count_1500m"]
    statrows=[];windows=[]
    for pop,base in [("S05_FIRE",optics),("READY_FIRE",optics[optics.ready])]:
        for year,g in [(y,base[base.year.eq(y)]) for y in YEARS]+[("ALL",base)]:
            for field in fields:statrows.append(stats(g,field,year=year,cohort=pop))
            for field in ["post_selection_mode","any_selected_post_after_primary_end"]:
                for k,n in g[field].fillna("UNKNOWN").value_counts().items():
                    windows.append(dict(year=year,cohort=pop,field=field,level=k,n=int(n),denominator=len(g),percent=100*n/len(g)))
    save(pd.DataFrame(statrows),"task2_optical_evidence_stats.csv")
    save(pd.DataFrame(windows),"task2_post_window_use.csv")
    full=pd.concat(cohorts,ignore_index=True)
    assert len(full)==6662 and full.sample_id.nunique()==6662
    full["month_utc"]=pd.to_datetime(full.reference_time_utc,utc=True).dt.month
    full["season"]=np.where(full.month_utc.isin([10,11,12,1,2,3]),"WARM_OCT_MAR","COOL_APR_SEP")
    save(full,"ready_sample_metadata_all_years.csv")
    rows=[]
    for year,g in [(y,full[full.year.eq(y)]) for y in YEARS]+[("ALL",full)]:
        for pool,h in g.groupby("pool",sort=False):
            for field in ["landcover_group","domain_class","month_utc","season"]:
                for k,n in h[field].fillna("UNKNOWN").value_counts().items():
                    rows.append(dict(year=year,pool=pool,dimension=field,level=k,n=int(n),denominator=len(h),n_known=int(h[field].notna().sum()),percent=100*n/len(h)))
            for field in ["candidate_onset_detected","persistent_onset_detected","possible_spatial_mismatch","possible_parallax_risk"]:
                b=boolean(h[field]);n=int(b.eq(True).sum())
                rows.append(dict(year=year,pool=pool,dimension=field,level="TRUE",n=n,denominator=len(h),n_known=int(b.notna().sum()),percent=100*n/len(h)))
    save(pd.DataFrame(rows),"task3_domain_and_diagnostics.csv")
    c=pd.concat(controls,ignore_index=True)
    save(c[["year","s05_sample_id","source_anchor_event_id","s05_split_anchor_group_id","s04_landcover_match_level","s04_landcover_exact_match","s04_landcover_broad_match"]],"task3_local_match_records.csv")
    matches=[]
    for year,g in [(y,c[c.year.eq(y)]) for y in YEARS]+[("ALL",c)]:
        for k,n in g.s04_landcover_match_level.fillna("UNKNOWN").value_counts().items():
            matches.append(dict(year=year,match_level=k,n=int(n),denominator=len(g),percent=100*n/len(g)))
    save(pd.DataFrame(matches),"task3_local_match_summary.csv")
    ter=load(AUX/"terrain/D2_V21_SAMPLE_TERRAIN_WITH_SPLIT_2022.csv")
    assert set(ter.sample_id)==set(full[full.year.eq(2022)].sample_id)
    ter["pool"]=np.where(ter.binary_label.eq(1),"FIRE",ter.source_pool)
    tr=[]
    for pool,g in [(p,ter[ter.pool.eq(p)]) for p in POOLS]+[("ALL",ter)]:
        for field in ["elevation_m","slope_deg","local_relief_m"]:tr.append(stats(g,field,year=2022,pool=pool))
    save(pd.DataFrame(tr),"task3_terrain_statistics_2022.csv")
    save(pd.DataFrame(joins),"linkage_checks.csv")
    save(pd.DataFrame(SOURCES).drop_duplicates("path"),"statistics_source_sha256.csv")
    dump(dict(status="PASS",ready=len(full),ready_fire=int(full.pool.eq("FIRE").sum()),s05_fire=len(optics)),"tasks123_complete.json")

if __name__=="__main__":run()
