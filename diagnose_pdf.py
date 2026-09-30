import fitz

pdf_path = r"C:\Users\HP\Desktop\agent_ai\docs_stage\questionnaire fin de mission.pdf"  
doc = fitz.open(pdf_path)

for pn, page in enumerate(doc):
    print(f"PAGE {pn+1} ")
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0: continue
        for line in block["lines"]:
            items = [(round(sp["bbox"][0]), round(sp["bbox"][1]), sp["text"].strip())
                     for sp in line["spans"] if sp.get("text","").strip()]
            if items:
                print(f"  y={items[0][1]:3d} : " +
                      " | ".join(f"x={x} '{t[:40]}'" for x,y,t in items))
    print("  --- RECTS ---")
    for path in page.get_drawings():
        r = path.get("rect")
        if r:
            w, h = r[2]-r[0], r[3]-r[1]
            if 15 < w < 560 and h > 3:
                print(f"  RECT x={r[0]:.0f}→{r[2]:.0f} y={r[1]:.0f}→{r[3]:.0f} w={w:.0f} h={h:.0f}")
doc.close()