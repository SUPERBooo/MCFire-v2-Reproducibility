from common import *

stage_prefixes=("S01_", "S01_5", "S02_", "S03_", "S04_", "S05_", "S08_", "S09_", "S12_", "S14_", "S15_")
inventory=[]
for year in YEARS:
    candidates=[]
    for d in (ROOTS[year]/"data/intermediate").iterdir():
        if d.is_dir() and d.name.startswith(stage_prefixes):
            candidates.extend(p for p in d.glob(f"*{year}*") if p.is_file() and p.suffix in (".csv",".json"))
    for folder in ("labels","splits","benchmark","metadata"):
        candidates.extend(p for p in (RELEASES[year]/folder).glob("*") if p.is_file() and p.suffix in (".csv",".json"))
    if year==2022:
        candidates.extend((AUX/"terrain").glob("*.csv"))
    for p in sorted(set(candidates)):
        info=dict(year=year,path=relative(p),bytes=p.stat().st_size)
        if p.suffix==".csv":
            try:
                df=read(p)
                info.update(rows=len(df),columns=list(df.columns))
            except Exception as e:
                info["error"]=str(e)
        inventory.append(info)
dump(inventory,"source_inventory_schema.json")
for row in inventory:
    if row['year']==2022:
        print(row['path'], row.get('rows',''), ','.join(row.get('columns',[])),flush=True)
