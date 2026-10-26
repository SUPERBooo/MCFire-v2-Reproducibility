"""Publication figure from measured arrays; axes use original HSD-grid pixel offsets."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent/".matplotlib"))
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, TwoSlopeNorm

COLORS={"FIRE":"#C05A2B","NEG_LOCAL_MATCHED":"#215A73","NEG_GLOBAL_STRATIFIED":"#8C9EB4"}

def run():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"axes.linewidth":.6,"pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"})
    s=splits(2022);fire=s[s.sample_id.eq("V22_000145")].iloc[0]
    controls=s[s.split_anchor_group_id.eq(fire.split_anchor_group_id)&s.source_pool.eq("NEG_LOCAL_MATCHED")].sort_values("sample_id")
    assert len(controls)>0
    control=controls.iloc[0]
    sid=str(fire.sample_id);cid=str(control.sample_id)
    trajectories=read(OUT/"task6_all_sample_trajectories.csv")
    agg=read(OUT/"task6_aggregate_trajectories.csv")
    with np.load(RELEASES[2022]/fire.tensor_relpath,allow_pickle=False) as z:
        bt=z["bt_k"];distance=z["distance_km"];ext=z["extraction_35km_mask"];valid=z["all_band_valid_mask"];offsets=z["slot_offset_min"];epoch=z["slot_time_epoch_s"]
    selected=[int(np.flatnonzero(offsets==o)[0]) for o in [-120,-60,-20,0]]
    maps_a=[np.where(valid[t]&ext[t],bt[t,0],np.nan) for t in selected]
    maps_b=[np.where(valid[t]&ext[t],bt[t,0]-bt[t,1],np.nan) for t in selected]
    a=np.concatenate([m[np.isfinite(m)] for m in maps_a]);b=np.concatenate([m[np.isfinite(m)] for m in maps_b])
    norm_a=Normalize(vmin=float(np.floor(a.min())),vmax=float(np.ceil(a.max())))
    norm_b=Normalize(vmin=float(np.floor(b.min())),vmax=float(np.ceil(b.max())))
    cmap_a=plt.get_cmap("inferno").copy();cmap_b=plt.get_cmap("viridis").copy();cmap_a.set_bad("white");cmap_b.set_bad("white")
    fig=plt.figure(figsize=(7.2,8.9),layout=None)
    outer=fig.add_gridspec(4,1,left=.09,right=.89,top=.88,bottom=.12,hspace=.66,height_ratios=[1,1,1.1,1.35])
    fig.suptitle("Thermal context and 19-slot trajectories",x=.09,y=.987,ha="left",fontsize=11,fontweight="bold",color="#182B35")
    fig.text(.09,.962,f"Illustrative event {sid}  |  VIIRS reference: 13 Jan 2022, 14:04 UTC",fontsize=7.5,color="#556B78")
    x=np.arange(-40,41);ygrid=np.arange(-40,41)
    panel_positions=[]
    for r,(maps,norm,cmap,title,clabel) in enumerate([(maps_a,norm_a,cmap_a,"B07 brightness temperature","B07 (K)"),(maps_b,norm_b,cmap_b,"B07 - B14 brightness temperature difference","B07 - B14 (K)")]):
        gs=outer[r].subgridspec(1,4,wspace=.16);axes=[]
        for j,(t,data) in enumerate(zip(selected,maps)):
            ax=fig.add_subplot(gs[0,j]);axes.append(ax)
            # Pcolormesh cells, outlines and text remain vector objects in PDF.
            im=ax.pcolormesh(x,ygrid,np.ma.masked_invalid(data),cmap=cmap,norm=norm,shading="nearest",edgecolors="face",linewidth=.05,antialiased=False,rasterized=False)
            ax.contour(x,ygrid,distance[t],levels=[5],colors=["#FFFFFF"],linewidths=[.9])
            ax.contour(x,ygrid,distance[t],levels=[5],colors=["#202F37"],linewidths=[.45])
            ax.contour(x,ygrid,distance[t],levels=[10,30],colors=["#DDE5EC"],linewidths=[.7],linestyles="dashed")
            ax.set_aspect("equal");ax.set_xlim(-40.5,40.5);ax.set_ylim(40.5,-40.5)
            ax.set_xticks([-40,0,40]);ax.set_yticks([-40,0,40]);ax.tick_params(length=2,pad=2)
            if j==0:ax.set_ylabel("Row offset (px)",labelpad=2)
            else:ax.set_yticklabels([])
            ax.set_xlabel("Column offset (px)",labelpad=2,fontsize=6.5)
            scan=pd.to_datetime(int(epoch[t]),unit="s",utc=True).strftime("%H:%M")
            ax.set_title(f"{int(offsets[t])} min\n{scan} UTC",fontsize=7.5,pad=4)
            for spine in ax.spines.values():spine.set_color("#B6C0C8")
        pos=axes[0].get_position();fig.text(.025,pos.y1+.041,"ab"[r],fontsize=11,fontweight="bold")
        fig.text(.09,pos.y1+.043,title,fontsize=9,fontweight="bold")
        cbax=fig.add_axes([.911,axes[-1].get_position().y0,.012,axes[-1].get_position().height])
        cb=fig.colorbar(im,cax=cbax);cb.set_label(clabel,fontsize=7,labelpad=5);cb.ax.tick_params(labelsize=6,length=2)
        cb.solids.set_rasterized(False)
        cb.solids.set_edgecolor("face")
    gs=outer[2].subgridspec(1,2,wspace=.35);axt=fig.add_subplot(gs[0,0]);axd=fig.add_subplot(gs[0,1])
    for id_,pool,style in [(sid,"FIRE","-"),(cid,"NEG_LOCAL_MATCHED","--")]:
        q=trajectories[trajectories.sample_id.eq(id_)].sort_values("offset_min")
        for ax,field in [(axt,"core_b07_p95_k"),(axd,"contrast_btd0714_k")]:
            ax.plot(q.offset_min,q[field],style,color=COLORS[pool],lw=1.5,marker="o",markersize=2.5,label=f"{'FIRE' if pool=='FIRE' else 'Local control'}: {id_}")
    for ax,title,ylabel in [(axt,"Core B07","Core P95 B07 (K)"),(axd,"Core-to-background contrast","BTD contrast (K)")]:
        ax.set_title(title,loc="left",fontsize=8);ax.set_ylabel(ylabel,labelpad=3)
        ax.set_xlabel("Offset from terminal scan (min)",fontsize=7);ax.set_xlim(-180,0);ax.set_xticks([-180,-120,-60,0]);ax.grid(axis="y",color="#E8ECF0",lw=.5)
        ax.spines[["top","right"]].set_visible(False)
    pos=axt.get_position();fig.text(.025,pos.y1+.026,"c",fontsize=11,fontweight="bold")
    fig.text(.09,pos.y1+.028,"Event and matched control",fontsize=9,fontweight="bold")
    axt.legend(loc="upper left",fontsize=6,frameon=False,handlelength=2,borderaxespad=.2)
    ax=fig.add_subplot(outer[3]);styles=["-","--","-."]
    for pool,style in zip(POOLS,styles):
        q=agg[agg.pool.eq(pool)].sort_values("offset_min");n=int(q.n_total.iloc[0])
        label={"FIRE":"FIRE","NEG_LOCAL_MATCHED":"Local matched","NEG_GLOBAL_STRATIFIED":"Regional stratified"}[pool]
        ax.fill_between(q.offset_min,q.p25,q.p75,color=COLORS[pool],alpha=.15,linewidth=0)
        ax.plot(q.offset_min,q["median"],style,color=COLORS[pool],lw=1.6,label=f"{label} (n = {n:,})")
    ax.axhline(0,color="#AAB4BB",lw=.6);ax.set_xlim(-180,0);ax.set_xticks(np.arange(-180,1,30));ax.set_xlabel("Offset from terminal nominal scan (min)");ax.set_ylabel("B07 - B14 contrast (K)",fontsize=8)
    ax.grid(axis="y",color="#E8ECF0",lw=.5);ax.spines[["top","right"]].set_visible(False)
    ax.legend(loc="upper left",frameon=False,fontsize=7,ncols=1)
    pos=ax.get_position();fig.text(.025,pos.y1+.023,"d",fontsize=11,fontweight="bold")
    fig.text(.09,pos.y1+.025,"All five annual cohorts: median and interquartile range",fontsize=9,fontweight="bold")
    fig.text(.09,.022,"Common valid pixels only. Blank map cells are masked. Solid contour: 5 km; dashed: 10 and 30 km.\nSlot 0 is the nominal 14:00 UTC scan, 4 min before this event's VIIRS reference; no ignition time is implied.",fontsize=6.5,color="#556B78",linespacing=1.55)
    fig.savefig(OUT/"Figure5_Thermal_Showcase.png",dpi=300)
    fig.savefig(OUT/"Figure5_Thermal_Showcase.pdf",dpi=300,metadata={"Title":"MCFire Figure 5: thermal context and 19-slot trajectories","Author":"MCFire supplementary analysis"})
    fig.savefig(OUT/"Figure5_preview.png",dpi=150)
    plt.close(fig)
    # Save every number behind panels c/d, including per-slot effective denominators.
    save(trajectories[trajectories.sample_id.isin([sid,cid])],"task6_showcase_pair_trajectories.csv")
    dump(dict(event_id=sid,control_id=cid,control_selection="lexicographically first ready NEG_LOCAL_MATCHED with same split_anchor_group_id; not selected by thermal separation",reference_time_utc=fire.reference_time_utc,slot0_time_utc=pd.to_datetime(int(epoch[-1]),unit="s",utc=True).isoformat(),sample_split=fire.split,event_candidate_onset_detected=bool(fire.candidate_onset_detected),event_persistent_onset_detected=bool(fire.persistent_onset_detected),event_thermal_trajectory_class=fire.thermal_trajectory_class,figure_size_inches=[7.2,8.9],dpi=300,font="DejaVu Sans",colors=COLORS,map_axis="original 81x81 HSD grid offsets in pixels; actual distance_km contours, not an assumed km/pixel scale",b07_color_limits=[norm_a.vmin,norm_a.vmax],btd_color_limits=[norm_b.vmin,norm_b.vmax],b07_colormap="inferno",btd_colormap="viridis",pdf="vector pcolormesh, lines, contours, axes and embedded TrueType text",contrast_formula="P95_core(B07-B14) - median_background(B07-B14)",aggregation="per-sample contrast, then pooled sample-weighted median/P25/P75 at each slot; NaN omitted without interpolation",style_reference="embedded Figures 3 and 4 in MCFire_Scientific_Data_Revised_v2.docx",matplotlib=matplotlib.__version__),"Figure5_parameters.json")
    print("Figure5 PNG and vector PDF saved",flush=True)

if __name__=="__main__":run()
