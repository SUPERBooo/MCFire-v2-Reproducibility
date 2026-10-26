"""Read manuscript XML/media as evidence without modifying the document."""
from common import *
import zipfile
from xml.etree import ElementTree as ET
from PIL import Image, ImageOps, ImageDraw
from io import BytesIO

paths = [OUT / "manuscript/MCFire_Scientific_Data_Revised_v2.docx", OUT / "manuscript/MCFire_Scientific_Data_Revised.docx"]
ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
for p in paths:
    if not p.exists():
        continue
    with zipfile.ZipFile(p) as z:
        root = ET.fromstring(z.read("word/document.xml"))
        paras = ["".join(t.text or "" for t in e.findall(".//w:t", ns)) for e in root.findall(".//w:body/w:p", ns)]
        tables = []
        for tbl in root.findall(".//w:tbl", ns):
            rows = [["".join(t.text or "" for t in c.findall(".//w:t", ns)) for c in r.findall("w:tc", ns)] for r in tbl.findall("w:tr", ns)]
            if any("ROC" in str(r) or "0.843" in str(r) for r in rows):
                tables.append(rows)
        ref = dict(source=p.name, sha256=sha(p), captions=[t for t in paras if t.startswith("Figure") or t.startswith("Table 7")], benchmark_tables=tables)
        dump(ref, f"reference/{p.stem}_reference.json")
        print(json.dumps(ref, ensure_ascii=True), flush=True)
        thumbs = []
        for name in z.namelist():
            if name.startswith("word/media/"):
                try:
                    im = Image.open(BytesIO(z.read(name))).convert("RGB")
                    canvas = Image.new("RGB", (520, 330), "white")
                    small = ImageOps.contain(im, (510, 300))
                    canvas.paste(small, ((520 - small.width) // 2, 22))
                    ImageDraw.Draw(canvas).text((6, 4), f"{name} {im.size}", fill="black")
                    thumbs.append(canvas)
                except Exception:
                    pass
        sheet = Image.new("RGB", (1040, 330 * ((len(thumbs) + 1) // 2)), "#eeeeee")
        for i, im in enumerate(thumbs):
            sheet.paste(im, ((i % 2) * 520, (i // 2) * 330))
        sheet.save(OUT / f"reference/{p.stem}_media_contact_sheet.png")
