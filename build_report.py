"""Build self-contained scientific handoff from actual computed CSV/JSON outputs."""
from common import *
import shutil

ORDER=["T0_SPATIAL16","COMPACT26","SPATIAL48","FUSION74"]
METRICS=["roc_auc","average_precision","balanced_accuracy","mcc","recall_sensitivity","specificity"]
NAMES={"roc_auc":"ROC-AUC","average_precision":"Average precision","balanced_accuracy":"Balanced accuracy","mcc":"MCC","recall_sensitivity":"Sensitivity","specificity":"Specificity"}
def f(v,d=4):
    return "UNKNOWN" if pd.isna(v) else f"{float(v):.{d}f}"
def table(df,fmt=None):
    fmt=fmt or {}
    def cell(v,k):
        if pd.isna(v):return "UNKNOWN"
        if k in fmt:return fmt[k](v)
        if isinstance(v,(float,np.floating)):return f(v)
        return str(v).replace("|","\\|").replace("\n"," ")
    headers=list(df.columns)
    return "| "+" | ".join(headers)+" |\n|"+"|".join(["---"]*len(headers))+"|\n"+"\n".join("| "+" | ".join(cell(v,k) for k,v in zip(headers,row))+" |" for row in df.itertuples(index=False,name=None))+"\n"
def numstat(df):
    d=df.copy()
    d["N 有效/总数"]=d.n_valid.astype(str)+"/"+d.n_total.astype(str)
    d["中位数 [P25, P75]"]=d.apply(lambda r:f"{f(r['median'])} [{f(r.p25)}, {f(r.p75)}]",axis=1)
    return table(d[["field","N 有效/总数","mean","std","中位数 [P25, P75]","minimum","maximum"]].rename(columns={"field":"原始字段","mean":"均值","std":"SD","minimum":"最小值","maximum":"最大值"}))
def count_percent(df,index):
    d=df.copy();d["n (%)"]=d.apply(lambda r:f"{int(r.n):,} ({r.percent:.2f}%)",axis=1)
    out=d.pivot(index=index,columns="pool",values="n (%)").reindex(columns=POOLS).fillna("0 (0.00%)").reset_index()
    return table(out)
def metric_ci_table(annual,macro,metrics):
    rows=[]
    for year,source in [(y,annual[annual.year.eq(y)]) for y in YEARS]+[("五年等权均值",macro)]:
        for rep in ORDER:
            row={"年度":year,"表示":rep}
            for metric in metrics:
                q=source[source.representation.eq(rep)&source.metric.eq(metric)].iloc[0]
                row[NAMES[metric]]=f"{f(q.point_estimate)} [{f(q.ci_lower)}, {f(q.ci_upper)}]"
            rows.append(row)
    return table(pd.DataFrame(rows))

def run():
    funnel=read(OUT/"task1_candidate_funnel.csv")
    optics=read(OUT/"task2_optical_evidence_stats.csv")
    dom=read(OUT/"task3_domain_and_diagnostics.csv",dtype={"year":str,"level":str})
    ready=read(OUT/"ready_sample_metadata_all_years.csv")
    bench=read(OUT/"task5_annual_point_metrics.csv") if (OUT/"task5_annual_point_metrics.csv").exists() else read(OUT/"task5_ablation_benchmark.csv")
    save(bench,"task5_annual_point_metrics.csv")
    ci=read(OUT/"task5_ablation_ci.csv");macro=read(OUT/"task5_macro_average_ci.csv")
    gains=read(OUT/"task5_fusion_mean_improvements.csv")
    repro=read(OUT/"task5_fusion74_reproduction_check.csv")
    pair=read(OUT/"task4_control_pool_specificity.csv")
    integ=json.loads((OUT/"protected_integrity_result.json").read_text())
    assert integ["status"]=="PASS"
    checks=json.loads((OUT/"supplement_validation_result.json").read_text())
    assert checks["status"]=="PASS"
    fig=json.loads((OUT/"Figure5_parameters.json").read_text(encoding="utf-8"))
    runtime=json.loads((OUT/"benchmark_runtime.json").read_text())
    parts=[]
    def add(text=""):parts.append(text.strip()+"\n")
    add("# MCFire 论文补充统计、消融实验与 Figure 5\n\n生成日期：2026-10-16。适用论文：Scientific Data Data Descriptor。\n\n本报告基于五套已完成 `2026-10-16_epoch_metadata_r1` 修正的年度核心包、2022 v2.1 地形辅助包及实际保存的中间审计表。所有数字来自本轮读取与计算，数据集标签、张量、划分及原冻结结果均未修改。新增实验属于在既有 TEST 已完成评估之后开展的回顾性补充分析，不应写成原冻结前预注册的首次测试。")
    add(f"执行前后对六个受保护目录的 **{integ['files_before']:,} 个文件**逐文件检查大小及 SHA256，新增、删除和内容变化均为 **0**。另有 {checks['checks']} 项补充结果交叉检查通过。报告、脚本、CSV、模型副本及图件全部位于 `{OUT.as_posix()}`。")
    add("## 阅读入口与核心结论")
    add("本报告正文包含全部任务的汇总表和精确统计值；CSV 提供未四舍五入的数值、逐样本记录和更多审计字段。Figure 5 的 PNG 为 2160 × 2670 px、300 DPI；PDF 的图格、曲线、轮廓和文字均为矢量对象，未嵌入栅格图片。")
    add("- 488,964 条质控后 VIIRS 检测记录形成 90,092 个候选事件；6,732 个因 HIGH 风险 HOLD。五年各取 600 个进入光学评估，共 3,000 个，产生 1,977 个 S05 FIRE，其中 1,824 个具有最终可用张量。\n- 最终 6,662 个张量样本包括 FIRE 1,824、本地匹配对照 2,099、区域分层对照 2,739。样本并非全部森林火灾；FIRE 中 `landcover_group=CROPLAND` 占 32.46%，`domain_class=CULTIVATED_DOMINANT` 占 36.35%，二者来自不同地类归并口径。\n- 可用 FIRE 的 89.31% 使用 PRIMARY_25D 光学窗口，10.69% 使用 FALLBACK_35D。光学统计描述参与标签生成的证据，不能充当独立标签准确率估计。\n- 最终 FUSION74 在五年均恢复了冻结阈值、分类和指标。五年平均 ROC-AUC 为 0.8095，高于 T0_SPATIAL16 的 0.7476、COMPACT26 的 0.7689 和 SPATIAL48 的 0.7897。相对 SPATIAL48，balanced accuracy 的均值差为 −0.0001，不能宣称所有指标均改进或证明时空联合表征在一切任务中“必不可少”。\n- 2023 首轮使用当前软件重新生成 CV 折次时与历史折次不一致；已按保存的 TRAIN OOF 折次重放，首轮结果完整保留并在第 5 节披露。2022 冻结 split 中的视差标志与事后地形辅助审计存在 158 条差异，第 3 节分别列出。")
    add("## 0. 年度版本、源文件与统计口径")
    roots=[]
    for y in YEARS:roots.append(dict(年度=y,工作区=ROOTS[y].name,核心包=relative(RELEASES[y])))
    add(table(pd.DataFrame(roots)))
    add(f"全部源相对路径基准为 `{BASE.as_posix()}`。2022 仅采用 v2.1 核心包；2023 使用已规范的含年份包名。独立地形辅助包为 `{relative(AUX)}`。未使用弃用的 2022 v2 核心包或旧 2023 包。")
    source_rows=[
        ("任务 1","data/intermediate/S01_VIIRS/D2_S01_VIIRS_PIXELS_YEAR.csv","一行是一个质控后检测记录，按 detection_id 检查唯一性"),
        ("任务 1","data/intermediate/S01_VIIRS/D2_S01_VIIRS_CANDIDATE_EVENTS_YEAR.csv","一行是一个时空连通候选事件"),
        ("任务 1","data/intermediate/S01_5_CANDIDATE_AUDIT/D2_S015_EVENT_AUDIT_MASTER_YEAR.csv","读取风险、HOLD 和 P1/P2/P3，不以当前默认配置重写历史"),
        ("任务 1/2","data/intermediate/S02_INDEPENDENT_EVIDENCE/D2_S02_INPUT_QUEUE_YEAR.csv；D2_S02_INDEPENDENT_EVIDENCE_MASTER_YEAR.csv","实际光学队列和已保存的光学观测/阈值/变化指标"),
        ("任务 1","data/intermediate/S03_EVIDENCE_FUSION/D2_S03_INDEPENDENT_LABEL_MASTER_YEAR.csv","实际判定 tier 与 evidence_status"),
        ("任务 2/3","核心包/labels/D2_S05_FINAL_BINARY_MASTER_YEAR.csv；D2_S05_FINAL_NON_FIRE_CONTROLS_YEAR.csv","最终标签库、source_event_id 和真实负样本匹配方式"),
        ("任务 2/3/5/6","核心包/splits/D2_S10_GROUP_AWARE_SPLIT_MASTER_YEAR.csv","可用张量样本、split、锚点组、冻结诊断字段和相对张量路径"),
        ("任务 4","核心包/benchmark/D2_S16_FINAL_TEST_PREDICTIONS_YEAR.csv；D2_S16_FINAL_TEST_METRICS_YEAR.csv","冻结 TEST 分数、阈值、类别及指标"),
        ("任务 5","data/intermediate/S14_COMPACT_SPATIAL_PHYSICS_FUSION/D2_S14_TRAIN_VAL_SPATIAL_FEATURE_MASTER_YEAR.csv；D2_S14_FEATURE_SCHEMA_YEAR.json；D2_S14_FUSION_PROTOCOL_YEAR.json","TRAIN/VAL 原始特征列顺序、确切特征定义和记录的模型协议"),
        ("任务 5","核心包/benchmark/D2_S16_FINAL_TEST_FEATURE_MASTER_YEAR.csv","原保存的 TEST 74 列特征，不重新提取或修改"),
        ("任务 6","核心包/tensors/samples/YEAR/*.npz；data/intermediate/S12_TRAIN_VAL_BASELINES/D2_S12_BASELINE_PROTOCOL_YEAR.json","全量实际亮温与掩码；核心区/背景区有效像元下限"),
        ("地形","辅助包/terrain/D2_V21_SAMPLE_TERRAIN_WITH_SPLIT_2022.csv；D2_S09_DEM_PARALLAX_GEOLOCATION_SAMPLE_AUDIT_2022.csv","1083 个样本的地形统计和事后诊断"),
    ]
    add(table(pd.DataFrame(source_rows,columns=["用途","相对年度工作区或包根目录的文件","解释"])))
    add("`statistics_source_sha256.csv`、`benchmark_source_sha256.csv`、`schema_alignment.json` 和 `source_inventory_schema.json` 记录实际来源、字节数、列名及哈希。引用代码片段见 `code_evidence/`；当前代码的说明作用与历史运行配置的证据作用已区分。\n\n显示池 `FIRE` 对应 `binary_label=1 / final_label=FIRE`，其原始 `source_pool` 实际值为 `INDEPENDENT_STRICT`；两个负池保持原名。关联统一按 sample_id 完成，光学证据通过 source_event_id → event_id 一对一关联。所有年度均检查重复键、缺失关联、标签/锚点组/划分一致性，结果为 0 冲突。\n\n检测点、候选事件、S05 标签样本、张量样本及时间槽是不同计数单位。每个最终样本对应一份 NPZ、19 个时间槽和 3 个热红外波段；6,662 × 19 = 126,578 个槽位置，不等于全部位置都有有效观测，也不是 126,578 份互相独立的原始卫星影像。五年 ALL 行按样本或事件汇总；只有模型“五年等权均值”按年度等权。")
    add("## 1. 候选事件、风险与光学判定漏斗")
    add("### 1.1 从质控检测记录到优先级队列")
    add(table(funnel[["year","qc_viirs_detections","candidate_events","hold_total","eligible_nonheld","P1","P2","P3"]]))
    add("S01 检测数来自已经质控并关联 event_id 的 PIXELS 文件，不等于下载前的原始记录数。候选事件总数与 EVENT_AUDIT_MASTER 完全一致。非 HOLD 不代表已确认 FIRE，P1/P2/P3 只是光学判定优先级。")
    add(table(funnel[["year","hold_persistent_only","hold_chaining_only","hold_both","moderate_nonheld","optical_queue","s03_strict","s05_fire","ready_fire"]]))
    add("三个 HIGH 子类互斥，其和为 hold_total；MODERATE 统计任一风险为 MODERATE 且未 HOLD 的事件。未进入每年 600 个评估队列的合格候选属于未抽入队列，不应统称为光学失败。队列内未成为严格 FIRE 的事件不改标为 NON_FIRE，最终负样本另由 S04 生成。")
    add("### 1.2 光学判定的完整结果类别")
    status=read(OUT/"task1_optical_statuses.csv",dtype={"year":str})
    pivot=status.pivot(index="level",columns="year",values="n").reindex(columns=[str(y) for y in YEARS]+["ALL"]).fillna(0).astype(int).reset_index()
    add(table(pivot))
    add("上表六类互斥，每年合计 600、五年合计 3,000。实际字段中没有单独名为 `no-change` 的类别，最接近的是 `NO_DECISIVE_SENTINEL2_BURNSCAR_SUPPORT`，其含义是没有决定性支持，不能解释为确认无变化。3,000 条队列均有 S02/S03 对应记录，缺失关联为 0；`S2_INSUFFICIENT` 是覆盖不足而非记录丢失。")
    add(table(read(OUT/"task1_optical_tiers.csv")))
    add("### 1.3 HIGH 与优先级阈值（沿用已有实现）")
    add("持续热源风险：事件中心在 EPSG:3577 下按欧氏距离 DBSCAN（eps=1.5 km，min_samples=1）组成重复出现组；组内活跃月份 ≥6 **且**事件数 ≥12 为 HIGH；MODERATE 为未达 HIGH、活跃月份 ≥3 且事件数 ≥5。DBSCAN 允许链式连接，因此组员不必全部处于同一中心的 1.5 km 内。\n\n链式风险 HIGH：持续时间 >48 h **或**检测点总数 >100 **或**投影边界框对角线 >5 km。对应 MODERATE 门槛为 >24 h、>20 点、>2.5 km，并在 HIGH 之后分级。任一风险 HIGH 即 HOLD_SCIENTIFIC_RISK。")
    add(table(pd.DataFrame([dict(层级="P1",参考最大FRP_MW="≤10",参考检测数="≤2",持续时间_h="≤12",单次过境最大点数="≤3"),dict(层级="P2（排除P1）",参考最大FRP_MW="≤30",参考检测数="≤5",持续时间_h="≤24",单次过境最大点数="≤8"),dict(层级="P3",参考最大FRP_MW="其余非 HOLD",参考检测数="—",持续时间_h="—",单次过境最大点数="—")])) )
    add("P1/P2 每行四个条件为 AND。参考 FRP 为最早 acquisition_time_utc 检测行中的最大 FRP，而非总 FRP；过境按卫星与 acquisition_time_utc 组合。依据为年度 candidate_audit.py 的 `_build_recurrence_audit`、`_assign_chaining_risk`、`_assign_priority_and_disposition` 函数，以及导出的 `code_evidence/candidate_rules_config_YEAR.yaml`（已移除无关配置）。这些门槛不是实际烧毁面积或真实起火阶段的测量。")
    add("可用于 Methods 的表述：After quality control, 488,964 VIIRS detection records were grouped into 90,092 candidate events. The risk audit held 6,732 candidates outside optical assessment. A stratified queue of 600 candidates per year was assessed using optical evidence, yielding 1,977 strict positive events, of which 1,824 subsequently met the input-availability and tensor-readiness requirements.")
    add("## 2. FIRE 的光学证据物理分布")
    add("### 2.1 统计定义及其限制")
    add("光学 dNBR、dNDVI、dMIRBI 及其阈值是无量纲指数差值；clear fraction 为 0–1 比例。它们与后文单位为 Kelvin 的 Himawari 亮温和热对比度不同。")
    add("分别报告全部 S05 FIRE（n=1,977）和最终 READY_FIRE（n=1,824）。SD 为样本标准差（ddof=1）；P25/P75 使用 pandas 的线性分位数，IQR 宽度为 P75−P25，CSV 同时保留宽度。统计对有限原始值进行，没有以 0 填补缺测。\n\n`pre_usable_count`/`post_usable_count` 是选中的可用日镶嵌观测日期数，不是全部 Sentinel-2 瓦片数；`*_clear_fraction_mean` 是每个事件内选中观测晴空比例的均值，再在事件之间统计，未按瓦片或面积重新加权。阈值和 dNBR/dNDVI/dMIRBI 均直接取已保存字段；没有重新访问 Earth Engine。\n\n变化指标分别提供 VIIRS 支持区域内的中位数和最大值，以及 core500 中位数。正标签要求局部支持像元满足条件，因此区域中位数低于阈值或略为负值并不等价于标签判定失败。`max_persistence_count_viirs_prior` 是该支持区域内像元持久性计数的最大值，不能写成“整个区域在所有这些日期均燃烧”。`median_persistence_count_viirs_prior` 的中位数可以为 0。\n\n两个 connected pixel count 字段保留实际浮点值。已有值并非总为整数；原实现通过掩码影像及 Earth Engine reduceRegion(sum) 统计，几何边界权重可产生小数。它们不是单个连通分量的无权整数大小，本轮不四舍五入为像元个数。")
    add("### 2.2 主窗口与扩展窗口")
    win=read(OUT/"task2_post_window_use.csv")
    add(table(win[win.field.eq("post_selection_mode")][["year","cohort","level","n","denominator","percent"]]))
    add("PRIMARY_25D 和 FALLBACK_35D 为源记录原名。本轮还逐个读取 post_dates 与 post_primary_end，确认 READY_FIRE 中实际使用主窗边界之后日期的样本为 195/1,824（10.69%），与 fallback 计数一致。fallback 模式从扩展的完整候选窗口选择日期，并不意味着选中日期全部位于第 26–35 天；GEE 日期过滤右端不包含、日镶嵌以日历日期表示，精确日期以逐事件表为准。")
    for cohort,label in [("READY_FIRE","最终 1,824 个可用 FIRE"),("S05_FIRE","全部 1,977 个 S05 FIRE")]:
        add(f"### 2.3 {label}：五年合并统计" if cohort=="READY_FIRE" else f"### 2.4 {label}：五年合并统计")
        add(numstat(optics[optics.year.eq("ALL")&optics.cohort.eq(cohort)]))
    add("### 2.5 各年度完整统计")
    for y in YEARS:
        for cohort in ["READY_FIRE","S05_FIRE"]:
            add(f"#### {y} / {cohort}")
            add(numstat(optics[optics.year.eq(str(y))&optics.cohort.eq(cohort)]))
    example=read(OUT/"task2_showcase_optical_example.csv").iloc[0]
    fields=["sample_id","reference_time_utc","pre_dates","post_dates","post_selection_mode","pre_usable_count","post_usable_count","pre_clear_fraction_mean","post_clear_fraction_mean","adaptive_threshold_dNBR","adaptive_threshold_dNDVI","adaptive_threshold_dMIRBI","viirs_prior_dNBR_max","viirs_prior_dNDVI_max","viirs_prior_dMIRBI_max","max_persistence_count_viirs_prior","viirs_supported_connected_dnbr_anchor_pixel_count"]
    add("### 2.6 Figure 5 事件的实际光学记录")
    add(table(pd.DataFrame([dict(字段=k,值=example[k]) for k in fields])))
    add("该事件的 FIRE 来源是独立于热输入的光学标签流程；这些标签生成证据不是独立盲法验证。当前文件未提供额外专家准确率、独立第三方事件核验或新的人工复核，本次也未开展这类工作。")
    add("## 3. 可用张量样本的地类、季节与诊断标志")
    counts=ready.groupby(["year","pool"]).size().unstack(fill_value=0).reindex(columns=POOLS);counts["TOTAL"]=counts.sum(axis=1);counts.loc["ALL"]=counts.sum();add(table(counts.reset_index()))
    add("以下比例的分母是同年度、同样本池的全部 READY 样本；缺失值另列 UNKNOWN，不并入未触发。ALL 为按样本合并，2023 的样本数较多，因此权重更大。")
    for dim,title in [("landcover_group","3.1 土地覆盖组"),("domain_class","3.2 生态域分类"),("season","3.3 暖季/冷季"),("month_utc","3.4 UTC 月份"),("candidate_onset_detected","3.5 物理及几何诊断")]:
        add("### "+title)
        if dim=="candidate_onset_detected":
            q=dom[dom.dimension.isin(["candidate_onset_detected","persistent_onset_detected","possible_spatial_mismatch","possible_parallax_risk"])]
            add(count_percent(q,["year","dimension"]))
        else:
            add(count_percent(dom[dom.dimension.eq(dim)],["year","level"]))
    add("暖季定义为 10–3 月，冷季为 4–9 月；月份来自实际 reference_time_utc 的 UTC 月份，不重定义火季年份。上表的 onset 是 S08 规则诊断，不能读作真实点火时间或独立火灾标签。风险字段也不是经定位真值验证的错误率。")
    add("### 3.6 本地负样本的实际匹配等级")
    add(table(read(OUT/"task3_local_match_summary.csv")))
    add("EXACT_GROUP 是目标 landcover group 的精确组匹配；BROAD_DOMAIN_FALLBACK 是未满足组内配额时的宽泛生态域回退，不表示经纬度、燃料或所有背景条件完全一致。统计使用负样本生成记录中的 s04_landcover_match_level，避免事后仅比较 landcover 字符串反推抽样方式。")
    add("### 3.7 2022 v2.1 独立地形辅助包")
    terrain=read(OUT/"task3_terrain_statistics_2022.csv")
    for pool in POOLS+["ALL"]:
        add(f"**2022 / {pool}**\n\n"+numstat(terrain[terrain.pool.eq(pool)]))
    add("数据源为 Copernicus DEM GLO-90 DSM（名义约 90 m）。高程单位 m，坡度单位 degree，relief 为邻域最大高程减最小高程（m）。代码将 ±2 km 转为经纬度包围窗口，因此应写“2 km 扩展范围的局部窗口”或约 4 km × 4 km 邻域，不能擅称精确圆形缓冲区。1083 个样本高程和 relief 全部可用，坡度 1080 个有效、3 个缺失。历史 protocol 中 resolution_arcsec=30 与此前已保存的实际栅格变换核验（3 arcsec）存在元数据冲突，本次未修改原文件；不应把 30 arcsec 写成约 90 m。")
    add("### 3.8 2022 split 冻结标志与事后地形审计差异")
    auxdiff=read(OUT/"task3_2022_frozen_vs_terrain_flags.csv")
    add(table(auxdiff))
    add("任务 3 主表严格采用所指定的 S05 × S10 关联，因此 `possible_parallax_risk` 使用冻结 split 字段。2022 split 中为 0 个 TRUE，但事后 v2.1 地形辅助审计为 158 个 TRUE（FIRE 37、本地 80、区域 41），是不同审计版本，不能混成一个未标注来源的百分比。其它三个标志没有差异。若论文选择报告最新地形审计，应引用本表或单列辅助结果；这里的 0 不能解释为已经证明不存在视差风险。")
    add("## 4. 冻结 TEST 的两类负样本拆分评估")
    add("直接使用年度 S16 冻结分数和阈值：2021=0.360、2022 v2.1=0.555、2023=0.220、2024=0.470、2025=0.545。判定为 score ≥ threshold。TN/FP 只在各自负样本池计算；特异度=TN/(TN+FP)，FPR=FP/(TN+FP)。")
    add(table(pair[["year","control_pool","threshold","test_fire","test_local","test_regional","tn","fp","specificity","fpr"]]))
    p=pair.copy();p["ROC-AUC [95% CI]"]=p.apply(lambda r:f"{f(r.roc_auc)} [{f(r.roc_auc_ci_lower)}, {f(r.roc_auc_ci_upper)}]",axis=1);p["AP [95% CI]"]=p.apply(lambda r:f"{f(r.average_precision)} [{f(r.average_precision_ci_lower)}, {f(r.average_precision_ci_upper)}]",axis=1)
    add(table(p[["year","control_pool","ROC-AUC [95% CI]","AP [95% CI]","n_groups","valid_resamples"]]))
    add("每个二分类子集包含该年全部 TEST FIRE 和所选负池，不强制一对一配对或只留下具备 ready 正样本的锚点。以 split_anchor_group_id 为单位有放回抽取原有组数，组内样本全部纳入；2,000 次、seed=20260829、2.5%/97.5% percentile CI，单类或非有限重采样跳过，所有本表有效次数为 2,000。\n\n这些是新增子集统计和新增 CI，未覆盖原冻结 S16 CI。两种子集正例率不同，Average precision 的差异同时受类比例影响，不能单独解释成负样本的物理难度差异。2024 本地池 FPR 为 36.21%，区域池为 26.25%，显示本地对照中误报更高；其它年份方向并不完全相同。")
    add("## 5. 特征子集消融实验")
    add("### 5.1 表示定义与训练协议")
    add(table(pd.DataFrame([("T0_SPATIAL16",16,"原 S14 schema 的 s48__*__t000 列，仅 offset=0"),("COMPACT26",26,"6 条19槽物理轨迹，每条 median/max/near-t0 max/slope 共24列，加整体和近末端有效槽比例2列"),("SPATIAL48",48,"offset −20、−10、0 各16个空间特征"),("FUSION74",74,"按年度 schema 原始顺序连接 COMPACT26 与 SPATIAL48")],columns=["表示","原始特征列数","实际计算含义"])))
    add("COMPACT26 的 6 条轨迹为 B07/B14/B15 各自的核心 P95 减背景中位数、B07−B14 与 B07−B15 的核心 P95 减对应背景中位数，以及核心 B07 P95。每槽共同有效像元下限为核心 3、背景 10；斜率仅在至少 3 个有限时刻存在时以 K/h 计算。SPATIAL48 每时刻的 16 个特征包括：B07 在 r≤2/5/7.5/10 km 与三个相邻环带的 P95−背景中位数（7项）；5 km 核心两种 BTD 对比（2项）；核心/10 km 热像元比例（3项）；10 km 内最热 B07/BTD 像元距中心距离（2项）；5 km B07 标准差和 P99−median（2项）。背景为 10–30 km，空间区域不足 3 个共同有效像元或背景不足 10 个时记 NaN。\n\n原始特征是物理 Kelvin/差值/距离/比例摘要，并未把 81×81 的全部格点直接送入 LogisticRegression，也没有用本轮全数据统计重新标准化。每折独立拟合 `SimpleImputer(median, add_indicator=True) → StandardScaler → L2 LogisticRegression`，solver=liblinear，max_iter=3000，random_state=20260829。74 是填补前的物理特征列数，最终五年度模型均含 107 个估计器输入维度（额外缺测指示列），见 frozen_model_runtime_evidence.csv。")
    add("仅在 TRAIN 内做 5 折分组 CV，group=split_anchor_group_id，种子 20260829；C∈{0.01,0.1,1,10,100}，class_weight∈{NONE,BALANCED}。超参数依次按 mean ROC-AUC、mean Average precision、mean balanced accuracy@0.5 降序，再按 |log10 C| 升序、NONE 优先处理平局。使用选定超参数生成 TRAIN OOF 分数，阈值网格 0.05–0.95（步长 0.005），依次按 balanced accuracy、MCC 降序，再按距 0.5、阈值本身升序。VAL 只作开发诊断，不参与本轮超参数或阈值选择；最终模型仅在 TRAIN 拟合。年度输入的标签、组和划分均与 S10 逐行核对。")
    add("全部确切列名在 model_details/YEAR_protocol.json；各候选超参数、逐折指标、OOF、阈值曲线、VAL 分数、最终模型及新增 TEST 分数均已保存。所有特征子集和评估指标来自用户预先指定的清单，没有依据此次 TEST 表现筛选子集或重新调阈值。")
    add("### 5.2 2023 软件版本与折次复现处理（必须保留的运行记录）")
    add(table(read(OUT/"task5_initial_fold_diagnosis.csv")))
    add("2023 原保存模型含 scikit-learn 1.3.2 版本信息；本轮环境为 1.9.1。同一协议种子在当前 StratifiedGroupKFold 生成的折次中，有 1,144/1,422 条 TRAIN 记录的折号与保存的 S14 OOF 不同。首轮 FUSION74 的阈值因此为 0.280，冻结阈值为 0.220，尽管全 TRAIN 拟合的预测分数几乎一致。\n\n处理只依据原始 TRAIN OOF 的 cv_fold：先确认 S14 两种表示的历史折次相同和组隔离成立，再为 2023 四个子集固定使用这些保存的折次。阈值决策由 TRAIN 数据重新生成并在最终 TEST 评分前记录；没有以 TEST 分数优劣决定修正。结果恢复 FUSION74=0.220 和 SPATIAL48=0.250，OOF 分数与历史文件的最大绝对差分别约 8.70×10⁻¹¹、2.68×10⁻¹²。\n\n因此本轮实际有 20 组最终消融结果，以及 4 组 2023 首轮软件复现诊断，共 24 次最终拟合；2023 首轮诊断已经产生过 TEST 分数，不能声称本轮在发现该差异前从未访问 TEST。首轮完整结果在 initial_modern_cv_diagnostic/，最终表使用历史折次重放版本，未择优拼接两轮结果。原先的冻结 S16 结果没有重跑或改写。")
    add(table(read(OUT/"task5_recorded_cv_2023_oof_reproduction.csv")))
    add("### 5.3 最终超参数与 TEST 混淆计数")
    add(table(bench[["year","representation","n_features","C","class_weight","threshold","n","tn","fp","fn","tp"]]))
    add("### 5.4 年度结果与五年等权均值（95% CI）")
    add(metric_ci_table(ci,macro,METRICS[:3]));add(metric_ci_table(ci,macro,METRICS[3:]))
    add("每一年度的 CI 使用该表示选定阈值下的 TEST 预测，以 split_anchor_group_id 分组有放回抽取 2,000 次（seed=20260829），单类/非有限抽样跳过；最终均为 2,000 个有效重采样。区间为 percentile，不是 BCa。它们描述给定已训练模型、给定年度 TEST 组的抽样不确定性，不包含重复训练或模型选择的不确定性。\n\n五年均值为五个年度点估计的算术平均，不是在混合年份 TEST 上训练/评估新模型。均值 CI 在固定五个年度内独立重采样锚点组，并对每轮的五个年度指标取平均；保存的年度 bootstrap 序列经过固定种子的独立年度置换组合，保持同年度不同表示的配对。该区间没有再抽样年份，不能视为对未来年份或跨年泛化的置信区间。\n\n新增 bootstrap 的 ROC-AUC 使用标准梯形积分，Average precision 按同分数组合后的 precision-recall 增量计算，并与 sklearn 在原样本和包含重复分数的样本上交叉核对。原 S16 快速 bootstrap 的 AP 函数按逐记录排名累加，遇到重采样重复分数时不完全等价；原冻结 CI 全部保留，报告中的新 CI 不应被标注为原 S16 已冻结 CI。")
    add("### 5.5 FUSION74 与原冻结结果及论文 Table 7 的核对")
    rp=repro.copy();rp["max_abs_probability_difference"]=rp.max_abs_probability_difference.map(lambda v:f"{v:.3e}");rp["max_abs_metric_difference"]=rp.max_abs_metric_difference.map(lambda v:f"{v:.3e}")
    add(table(rp))
    add("五年阈值、分类和六项指标均一致，预测分数只有数值精度级差异；最大差出现在 2025（约 6.39×10⁻⁸），未改变排序指标或阈值分类。直接读取了 MCFire_Scientific_Data_Revised.docx 与 Revised_v2.docx 的 Table 7，两份文稿共 70 个年度数值项与原冻结指标按发表精度一致，详情为 table7_frozen_results_comparison.csv。它们未在本任务中改写。")
    add("### 5.6 五年平均提升幅度")
    add(table(gains.rename(columns={"mean_fusion_minus_comparator":"FUSION74减对照的均值差"})))
    add("可用于 Technical Validation 的表述：In a retrospective feature-subset analysis using the recorded annual partitions and training-only model-selection procedure, FUSION74 achieved a mean annual ROC-AUC of 0.8095, compared with 0.7476 for terminal-frame spatial features, 0.7689 for trajectory summaries and 0.7897 for near-terminal spatial features. The corresponding differences were 0.0619, 0.0406 and 0.0198. However, the mean balanced accuracy did not improve over SPATIAL48 (difference −0.0001). These results support complementary ranking information in the combined representation, rather than universal superiority across metrics or proof of early ignition prediction.\n\n本分析的 COMPACT26 本身包含核心/背景空间汇总，SPATIAL48 包含三个时间槽；这不是完全正交的“纯时间对纯空间”对照。也未比较多种裁剪宽度，因此不能据此证明恰好 81×81 宽度的必要性。标签支持、分类能力、起火预测和真实提前发现时间是不同论断。")
    add("## 6. Figure 5：空间切片和全量热轨迹")
    add("![Figure 5](Figure5_Thermal_Showcase.png)")
    add("### 6.1 样本、物理定义与时间锚点")
    add(f"展示 FIRE `{fig['event_id']}` 及同一 split_anchor_group_id 下按 sample_id 字典序选取的第一个可用本地对照 `{fig['control_id']}`；两者属于 TRAIN。选择对照未依据热曲线差异。参考时间为 2022-01-13 14:04 UTC，offset 0 对应名义 14:00 UTC 扫描，先于参考时刻 4 分钟。−120/−60/−20/0 min 分别对应 12:00/13:00/13:40/14:00 UTC。")
    add("该指定 FIRE 样本的 candidate_onset_detected=False、persistent_onset_detected=False，thermal_trajectory_class=NO_CLEAR_THERMAL_ONSET。Figure 5 展示其真实热环境和与控制样本的差别，不能把它描写成清楚观察到起火/快速升温的典型案例。没有以更漂亮的事件替换用户指定样本。\n\n(a) B07 的四个完整 81×81 空间切片；(b) 相同时刻的逐像元 B07−B14；仅三个波段共同有效且处于 extraction_35km_mask 内的像元显示，其他留白。原 HSD 格网以行列偏移显示，没有把 nominal 2 km 分辨率误作各地固定 km/pixel 比例。轮廓使用每个格点实际 distance_km：5 km 实线，10/30 km 虚线，因此投影到格网上可以不是完美圆形。\n\n(c) 19 槽核心 B07 P95，以及 P95_core(B07−B14)−median_background(B07−B14)；core≤5 km、background=10–30 km，至少 3/10 个共同有效像元才计算。(d) 先计算每个样本每个槽的对比度，再在同样本池内计算中位数和 P25–P75。NaN 直接剔除于该槽统计，不插值、不用零填补；各槽有效 n 不同。半透明带是跨样本 IQR，不是均值的置信区间，也不能反映真实起火前后的同一阶段。")
    add("### 6.2 关键时间点的数值（K）")
    add(table(read(OUT/"task6_key_timepoint_values.csv")[["pool","offset_min","n_total","n_valid","n_missing","median","p25","p75"]]))
    add("### 6.3 全部 19 个槽的汇总")
    add(table(read(OUT/"task6_aggregate_trajectories.csv")[["pool","offset_min","n_total","n_valid","n_missing","median","p25","p75"]]))
    add("### 6.4 实际检查与制图参数")
    add(table(read(OUT/"task6_tensor_checks.csv")))
    add("全量逐样本核对 tensor_sha256、19×3×81×81 形状、波段 B07/B14/B15、共同有效掩码、35 km 外 NaN 及 UTC/epoch 槽位公式。新计算的 TRAIN/VAL 热对比轨迹还与 S12 缓存逐槽比较，5,674×19=107,806 个指标位置全部匹配（允许 1e−6 的数值容差），没有使用旧错误 epoch。")
    add(table(pd.DataFrame([dict(参数="输出",值="PNG 300 DPI，2160×2670 px；单页矢量 PDF"),dict(参数="物理尺寸",值="7.2×8.9 inch（182.88×226.06 mm）"),dict(参数="字体",值="DejaVu Sans；PDF 嵌入 TrueType 字体"),dict(参数="池颜色",值="FIRE #C05A2B；LOCAL #215A73；REGIONAL #8C9EB4"),dict(参数="B07色阶",值="inferno，286–296 K，四时刻共用"),dict(参数="BTD色阶",值="viridis，−1–9 K，四时刻共用"),dict(参数="插值",值="无时间插值；图格为原值 pcolormesh，无图像平滑"),dict(参数="版面核验",值="PNG 与 PDF 渲染已目视检查；PDF 对象检查无嵌入栅格图"),dict(参数="风格依据",值="读取两份现有文稿中的 Figure 3/4，沿用白底、无衬线字体、暖色正例和冷色对照")])) )
    add("### 6.5 可交给论文助手的英文图注")
    add("**Figure 5. Thermal context and trajectories of the tensor-ready MCFire cohorts.** (a) B07 brightness temperatures and (b) B07−B14 brightness-temperature differences for the optically supported FIRE sample V22_000145 at offsets −120, −60, −20 and 0 min. Panels retain the original 81×81 grid; pixels outside the 35 km extraction mask or lacking common validity are blank. Solid contours indicate 5 km and dashed contours indicate 10 and 30 km from the sample centre using the stored distance grid. (c) Core B07 95th-percentile temperatures and core-to-background B07−B14 contrasts for V22_000145 and the local matched control N22_L_000009 from the same anchor group. This illustrative event has no clear onset under the frozen thermal diagnostic and is not presented as an observed ignition sequence. (d) Pooled median contrasts and interquartile ranges for FIRE (n=1,824), local matched controls (n=2,099) and regional stratified controls (n=2,739). Contrasts are computed per sample as the core 95th percentile minus the background median using common valid pixels. Missing slots are omitted without interpolation, so effective sample counts vary by slot. Offsets are relative to the terminal nominal scan; for the illustrated event, scan time 14:00 UTC precedes the first VIIRS confirmation at 14:04 UTC on 13 January 2022. Neither the reference timestamp nor the aggregated trajectories establishes true ignition time.")
    add("## 7. 交付清单、复现与未修改范围")
    output_rows=[("MCFire_Paper_Supplement_Report.md","本自包含报告，可直接交给写作窗口"),("Figure5_Thermal_Showcase.png / .pdf","投稿图件，PNG 300 DPI；PDF 全矢量"),("task1_candidate_funnel.csv / task1_optical_tiers.csv / task1_optical_statuses.csv","漏斗、优先级、风险和全部判定类别"),("task2_optical_evidence_stats.csv / task2_post_window_use.csv","两个 FIRE 队列的年度与合并统计"),("task2_fire_optical_records.csv / task2_showcase_optical_example.csv","逐事件指标、选中观测日期及展示事件实际记录"),("task3_domain_and_diagnostics.csv / task3_local_match_summary.csv","地类、生态域、月、季节、诊断标志和匹配比例"),("task3_terrain_statistics_2022.csv / task3_2022_frozen_vs_terrain_flags.csv","地形统计及冻结/事后审计冲突对照"),("task4_control_pool_specificity.csv","冻结分数拆分的 TN/FP、特异度、FPR、ROC-AUC、AP 和新增 CI"),("task5_ablation_benchmark.csv / task5_ablation_ci.csv / task5_macro_average_ci.csv","最终消融数值、年度 CI 和五年等权均值 CI"),("task5_fusion_mean_improvements.csv / task5_fusion74_reproduction_check.csv","平均提升、差值 CI 以及冻结复现核对"),("task5_new_test_predictions.csv","四种最终表示的 TEST 样本 ID、group、真实标签、分数、阈值和类别，方便后续画 ROC/PR"),("model_details/","每年度确切特征列表、协议、超参数候选、OOF、阈值曲线、VAL、模型、bootstrap 抽样指标"),("initial_modern_cv_diagnostic/","2023 首轮当前软件折次诊断及原始中间结果，不能混入最终结果"),("task6_all_sample_trajectories.csv / task6_aggregate_trajectories.csv / task6_key_timepoint_values.csv","全 126,578 槽位置、19 槽池级轨迹和关键时点原值"),("ready_sample_metadata_all_years.csv","6662 个样本元数据，包含坐标、标签、划分和来源池"),("protected_files_before.csv / protected_files_after.csv / protected_files_differences.csv","六个受保护目录的全量执行前后哈希及零差异记录"),("supplement_validation_checks.csv / source_manifest.csv / code_evidence/","交叉验证、来源清单和代码依据")]
    add(table(pd.DataFrame(output_rows,columns=["文件（相对本报告目录）","用途"])))
    add(f"模型与图件使用本机已有运行环境 `{runtime['executable']}`：Python 3.13.15、numpy {runtime['numpy']}、pandas {runtime['pandas']}、scikit-learn {runtime['sklearn']}、scipy {runtime['scipy']}；matplotlib {fig['matplotlib']}。只读统计/哈希检查另使用 Codex 现有 Python 环境，版本见 analysis_runtime.json。没有安装新依赖。")
    add("脚本入口：`tasks123_statistics.py`（任务1–3）、`tasks45_benchmarks.py`（主补充实验）、`restore_recorded_cv_2023.py`（历史折次兼容处理）、`task6_extract_trajectories.py`、`plot_figure5.py`、`validate_and_export_evidence.py`、`build_report.py`。原阶段 run_sXX 没有调用。完整安全重现步骤见 README_REPRODUCE.md；不要再次执行任何原发布修复/冻结脚本。代码的写入目标均是本独立目录，原始年度包只作为输入。")
    add("本轮没有下载遥感影像、修改或重新划分样本、重构标签、插值补齐张量、改写历史指标、上传仓库或创建 DOI。新增训练使用既有特征表及原划分；TEST 用途和 2023 复现诊断的两轮访问已明确记录。文件完整性通过不能解释为标签绝对准确。")
    add("## 8. 写作时必须保留的限制或差异")
    add(table(pd.DataFrame([
        ("2023 CV 软件差异","实际模型版本信息和原 OOF 折次已读取；首轮当前软件生成折次不同，最终按记录重放","补充实验属于回顾性分析；保留首轮诊断与最终版本，不能称首次从未见 TEST"),
        ("2022 视差标志","冻结 split 与地形辅助审计有 158 个标志差异","注明诊断版本；不能把冻结 0 个 TRUE 写成无视差风险"),
        ("2022 地形分辨率字段","历史 resolution_arcsec=30 与已有栅格几何核验 3 arcsec 冲突","描述 Copernicus GLO-90 名义分辨率；不要使用错误换算"),
        ("光学证据的独立性","本轮统计来自参与标签生成的同一证据","不能据此给出独立人工/第三方标签准确率"),
        ("connected count 是浮点","已存 reduceRegion 统计含非整数值","报告原始计数定义，不能当作单分量的精确整数像元数"),
        ("参考时间与 t=0","真实 UTC/epoch 和 nominal scan 全量核对；没有真实点火时刻","不得把 VIIRS 首次确认前写成真实起火前"),
        ("Figure 5 指定事件","真实诊断为 NO_CLEAR_THERMAL_ONSET","作为示例和热环境展示，不能包装成提前发现成功案例"),
        ("新 bootstrap 与冻结 bootstrap","新 AP 对同分数统一处理；原 CI 未改","新 CI 标注本轮新增，不替换成原 S16 冻结结果"),
        ("消融并非严格正交","COMPACT26 包含空间统计；SPATIAL48 含3槽；未改变裁剪宽度","支持组合表示的部分指标改善，不证明完整19帧或81×81宽度的普遍必要性"),
        ("五年均值及全量轨迹","固定年度等权模型均值；轨迹则按样本合并，含 TRAIN/VAL/TEST","不是跨年模型 TEST，也不是可用于新模型调参的额外独立验证集"),
    ],columns=["事项","已核实情况","对论文的影响"])))
    (OUT/"MCFire_Paper_Supplement_Report.md").write_text("\n".join(parts),encoding="utf-8")
    # Enrich requested main CSV with all annual confidence intervals and macro rows.
    wide=bench.copy()
    for metric in METRICS:
        sub=ci[ci.metric.eq(metric)][["year","representation","ci_lower","ci_upper"]].rename(columns={"ci_lower":metric+"_ci_lower","ci_upper":metric+"_ci_upper"})
        wide=wide.merge(sub,on=["year","representation"],validate="one_to_one")
    macro_rows=[]
    for rep in ORDER:
        r=dict(year="MACRO_5_YEARS",representation=rep,n_features=int(bench[bench.representation.eq(rep)].n_features.iloc[0]),n=988)
        for metric in METRICS:
            q=macro[macro.representation.eq(rep)&macro.metric.eq(metric)].iloc[0]
            r.update({metric:q.point_estimate,metric+"_ci_lower":q.ci_lower,metric+"_ci_upper":q.ci_upper})
        macro_rows.append(r)
    wide["aggregation"]="ANNUAL_TEST"
    mc=pd.DataFrame(macro_rows);mc["aggregation"]="EQUAL_YEAR_MEAN_FIXED_FIVE_COHORTS"
    save(pd.concat([wide,mc],ignore_index=True),"task5_ablation_benchmark.csv")
    print(f"Report saved: {len(parts)} sections/blocks, {(OUT/'MCFire_Paper_Supplement_Report.md').stat().st_size:,} bytes",flush=True)

if __name__=="__main__":run()
