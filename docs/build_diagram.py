"""Generate docs/pipeline.drawio (2 pages: Pipeline, Code map) from experiments/*.json.

Re-run after new experiments so the diagram shows current scores:
  python docs/build_diagram.py
Then open docs/pipeline.drawio in draw.io (app or app.diagrams.net) to view or edit.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
OUT = ROOT / "docs" / "pipeline.drawio"

# draw.io's standard palette: (fill, stroke) per stage
BLUE, GREEN, YELLOW = ("#dae8fc", "#6c8ebf"), ("#d5e8d4", "#82b366"), ("#fff2cc", "#d6b656")
PURPLE, ORANGE, RED, GREY = ("#e1d5e7", "#9673a6"), ("#ffe6cc", "#d79b00"), ("#f8cecc", "#b85450"), ("#f5f5f5", "#666666")

BOX = "rounded=1;whiteSpace=wrap;html=1;align=left;verticalAlign=top;spacingLeft=8;spacingTop=4;arcSize=6;fontSize=11;"
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;endFill=1;strokeWidth=1.5;fontSize=10;"


class Page:
    def __init__(self, name: str):
        self.name, self.cells, self.n = name, [], 0

    def _id(self, prefix):
        self.n += 1
        return f"{prefix}{self.n}"

    def lane(self, title, x, y, w, h, color):
        cid = self._id("lane")
        style = (f"swimlane;rounded=1;html=1;startSize=34;fillColor={color[0]};strokeColor={color[1]};"
                 "fontStyle=1;fontSize=14;swimlaneFillColor=#ffffff;arcSize=4;")
        self.cells.append(("v", cid, title, style, "1", (x, y, w, h)))
        return cid

    def box(self, label, x, y, w, h, color=GREY, parent="1", extra=""):
        cid = self._id("box")
        self.cells.append(("v", cid, label, f"{BOX}fillColor={color[0]};strokeColor={color[1]};{extra}", parent, (x, y, w, h)))
        return cid

    def text(self, label, x, y, w, h, size=12, extra=""):
        cid = self._id("txt")
        style = f"text;html=1;align=left;verticalAlign=top;whiteSpace=wrap;fontSize={size};{extra}"
        self.cells.append(("v", cid, label, style, "1", (x, y, w, h)))
        return cid

    def edge(self, src, dst, label="", dashed=False, extra="", points=()):
        style = EDGE + ("dashed=1;" if dashed else "") + extra
        self.cells.append(("e", self._id("edge"), label, style, src, dst, points))

    def to_xml(self, diagram_id):
        d = ET.Element("diagram", id=diagram_id, name=self.name)
        model = ET.SubElement(d, "mxGraphModel", dx="1600", dy="1000", grid="1", gridSize="10", guides="1",
                              tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1",
                              pageWidth="1800", pageHeight="1100", math="0", shadow="0")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        for cell in self.cells:
            if cell[0] == "v":
                _, cid, label, style, parent, (x, y, w, h) = cell
                c = ET.SubElement(root, "mxCell", id=cid, value=label, style=style, vertex="1", parent=parent)
                ET.SubElement(c, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
            else:
                _, cid, label, style, src, dst, points = cell
                c = ET.SubElement(root, "mxCell", id=cid, value=label, style=style, edge="1", parent="1",
                                  source=src, target=dst)
                geo = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
                if points:
                    arr = ET.SubElement(geo, "Array", **{"as": "points"})
                    for px, py in points:
                        ET.SubElement(arr, "mxPoint", x=str(px), y=str(py))
        return d


def load_experiments():
    singles, blends = [], []
    for path in sorted(EXPERIMENTS.glob("*.json")):
        r = json.loads(path.read_text())
        (blends if "members" in r else singles).append(r)
    return singles, blends


def pipeline_page(singles, blends) -> Page:
    p = Page("Pipeline")
    p.text("<b>Predicting Airline Satisfaction · Kaggle Playground S6E10</b>", 20, 16, 1200, 30, size=22)
    p.text("Binary classification · metric ROC AUC · 699,635 train / 299,844 test rows · deadline 31 Oct 2026",
           20, 48, 1200, 22, size=13, extra="fontColor=#555555;")

    W, H, Y, GAP = 260, 560, 90, 34
    xs = [20 + i * (W + GAP) for i in range(6)]
    inner = W - 20

    # 1. Data
    data = p.lane("1 · Data", xs[0], Y, W, H, BLUE)
    dl = p.box("<b>Kaggle CLI</b><br><font face='Courier New'>make data</font><br>kaggle competitions download<br>playground-series-s6e10", 10, 46, inner, 78, BLUE, data)
    train_csv = p.box("<b>train.csv</b><br>699,635 rows × 23 cols<br>id · 21 features · <i>satisfaction</i><br>44.36% satisfied", 10, 144, inner, 80, BLUE, data)
    test_csv = p.box("<b>test.csv</b><br>299,844 rows × 22 cols<br>(no target)", 10, 244, inner, 62, BLUE, data)
    sample = p.box("<b>sample_submission.csv</b><br>id, satisfaction (probability)", 10, 326, inner, 50, BLUE, data)
    p.box("<i>data/ is git-ignored: Kaggle rules forbid redistributing competition data.</i>", 10, 396, inner, 50, GREY, data)
    for t in (train_csv, test_csv, sample):
        p.edge(dl, t, extra="exitX=0;exitY=0.5;entryX=0;entryY=0.5;")

    # 2. Load & EDA
    eda = p.lane("2 · Load & EDA", xs[1], Y, W, H, GREEN)
    loader = p.box("<b>src/data.py</b><br>• categoricals → <i>category</i> dtype<br>• target True/False → 1/0<br>• id becomes the index", 10, 46, inner, 84, GREEN, eda)
    eda_py = p.box("<b>src/eda.py</b> → reports/eda.md", 10, 150, inner, 34, GREEN, eda)
    p.box("<b>Key findings</b><br>• Online boarding alone: AUC 0.84<br>• Class: Business 72.5% satisfied vs Eco 16.8%<br>"
          "• Personal travel: only 9.7% satisfied<br>• Customer Type: loyal 49.5% vs disloyal 20.1%<br>"
          "• NaN only in Arrival Delay (292 train / 130 test)<br>• No train/test drift (all means within 2%)<br>"
          "• Delays, Gender, Gate location ≈ no signal alone", 10, 204, inner, 232, GREEN, eda)
    p.edge(train_csv, loader, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.35;")
    p.edge(test_csv, loader, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.75;")
    p.edge(loader, eda_py)

    # 3. Features
    feat = p.lane("3 · Features", xs[2], Y, W, H, YELLOW)
    raw = p.box("<b>raw · 21 features</b><br>• Age, Flight Distance<br>• Departure / Arrival delay (min)<br>"
                "• 13 service ratings (0–5)<br>• Gender, Customer Type,<br>&nbsp;&nbsp;Type of Travel, Class", 10, 46, inner, 120, YELLOW, feat)
    fe = p.box("<b>fe · 34 features</b> = raw +<br>• svc_mean / min / max / std<br>• svc_n5, svc_n0, svc_low<br>"
               "• boarding_x_wifi<br>• delay_total, delay_recovered,<br>&nbsp;&nbsp;delay_per_1000km<br>"
               "• class_x_travel, customer_x_travel", 10, 196, inner, 150, YELLOW, feat)
    p.box("<b>src/features.py</b><br>choose with <font face='Courier New'>--features raw|fe</font>", 10, 366, inner, 48, GREY, feat)
    p.edge(loader, raw)
    p.edge(raw, fe, "+13")

    # 4. Validation
    val = p.lane("4 · Validation", xs[3], Y, W, H, PURPLE)
    skf = p.box("<b>StratifiedKFold</b> (src/cv.py)<br>5 folds · shuffle · seed 42<br>same folds for every experiment<br>"
                "139,927 rows per fold<br>44.36% satisfied in every fold", 10, 46, inner, 112, PURPLE, val)
    loop = p.box("<b>For each fold k</b><br>1. train on the other 4 folds<br>2. early-stop on fold k (200 rounds)<br>"
                 "3. predict fold k → OOF<br>4. predict test → average of 5", 10, 198, inner, 104, PURPLE, val)
    p.box("<b>CV score</b> = ROC AUC of OOF<br>predictions on all 699,635 rows", 10, 322, inner, 48, PURPLE, val)
    p.edge(raw, loop, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.3;")
    p.edge(fe, loop, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.7;")
    p.edge(skf, loop)

    # 5. Models (from experiments/*.json)
    mod = p.lane("5 · Models", xs[4], Y, W, H, ORANGE)
    best_single = max(singles, key=lambda r: r["oof_auc"]) if singles else None
    exp_boxes, y = [], 46
    for r in singles:
        avg_trees = round(sum(r["best_iterations"]) / len(r["best_iterations"]))
        star = " ★" if r is best_single else ""
        label = (f"<b>{r['experiment']}</b>{star}<br>{r['model']} · {r['features']} ({r['n_features']})<br>"
                 f"lr {r['params']['learning_rate']} · ~{avg_trees:,} trees/fold<br>"
                 f"<b>OOF AUC {r['oof_auc']:.6f}</b> ± {r['fold_auc_std']:.5f}")
        exp_boxes.append(p.box(label, 10, y, inner, 72, ORANGE, mod))
        p.edge(loop, exp_boxes[-1], extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
        y += 86
    if singles:
        prm = singles[0]["params"]
        p.box(f"<b>LightGBM params</b> (src/models.py)<br>num_leaves {prm['num_leaves']} · min_child {prm['min_child_samples']}<br>"
              f"subsample {prm['subsample']} · colsample {prm['colsample_bytree']}<br>reg_lambda {prm['reg_lambda']} · "
              f"early stop 200", 10, y, inner, 74, GREY, mod)
        top = list(singles[0]["feature_importance_gain"].items())[:5]
        p.box("<b>Top gain (" + singles[0]["experiment"] + ")</b><br>" +
              "<br>".join(f"{name}: {g:.0%}" for name, g in top), 10, y + 88, inner, 96, GREY, mod)

    # 6. Blend & submit
    sub = p.lane("6 · Blend & submit", xs[5], Y, W, H, RED)
    if blends:
        b = max(blends, key=lambda r: r["oof_auc"])
        blend = p.box(f"<b>Rank-average blend</b><br>{' + '.join(m.split('_')[0] for m in b['members'])}<br>"
                      f"<b>OOF AUC {b['oof_auc']:.6f}</b><br><i>AUC only cares about order → average ranks</i>",
                      10, 46, inner, 84, RED, sub)
        for e in exp_boxes:
            p.edge(e, blend, dashed=False, extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    else:
        blend = p.box("<b>Rank-average blend</b>", 10, 46, inner, 84, RED, sub)
    checks = p.box("<b>src/submit.py checks</b><br>• same rows & id order as<br>&nbsp;&nbsp;sample_submission.csv<br>• ids unique · preds in [0, 1]", 10, 150, inner, 80, RED, sub)
    csv = p.box("<b>submissions/&lt;name&gt;.csv</b>", 10, 250, inner, 34, RED, sub)
    kag = p.box("<b>kag submit … --cv</b><br>→ Kaggle public leaderboard<br>(10 submissions / day)", 10, 304, inner, 62, RED, sub)
    sync = p.box("<b>kag sync</b><br>CV vs LB correlation:<br>trust CV if they agree", 10, 386, inner, 62, RED, sub)
    p.edge(blend, checks)
    p.edge(checks, csv)
    p.edge(csv, kag)
    p.edge(kag, sync)
    below = Y + H + 18
    p.edge(sync, feat, "next idea: new features / models", dashed=True,
           extra="exitX=0.5;exitY=1;entryX=0.5;entryY=1;",
           points=[(xs[5] + W / 2, below), (xs[2] + W / 2, below)])

    # Legend
    ly = Y + H + 50
    p.text("<b>Legend</b>", 20, ly, 100, 22, size=13)
    for i, (name, color) in enumerate([("Data", BLUE), ("Load & EDA", GREEN), ("Features", YELLOW),
                                       ("Validation", PURPLE), ("Models", ORANGE), ("Blend & submit", RED)]):
        p.box(name, 20 + i * 130, ly + 28, 118, 30, color, extra="align=center;verticalAlign=middle;spacingLeft=0;")
    p.text("solid arrow = data flow · dashed = check / feedback loop · ★ = best single model", 820, ly + 32, 600, 22, size=12)
    p.text("<i>Generated by docs/build_diagram.py from experiments/*.json. Re-run it after new experiments.</i>",
           20, ly + 72, 900, 22, size=11, extra="fontColor=#777777;")
    return p


def code_map_page() -> Page:
    p = Page("Code map")
    p.text("<b>Code map</b>: which file does what, and who imports whom", 20, 16, 1000, 30, size=22)
    p.text("solid arrow = imports / uses · dashed arrow = imports config.py, runs, or reads experiments/*.json",
           20, 48, 1000, 22, size=12, extra="fontColor=#555555;")

    config = p.box("<b>src/config.py</b><br>paths (data/, outputs/, experiments/, submissions/)<br>"
                   "ID, TARGET, SEED = 42, N_FOLDS = 5<br>column groups: NUMERIC, SERVICE_RATINGS, CATEGORICAL",
                   560, 80, 420, 84, PURPLE)

    # row 1: library modules
    data = p.box("<b>src/data.py</b><br>load_train() · load_test()<br>load_sample_submission()", 40, 240, 300, 70, GREEN)
    feats = p.box("<b>src/features.py</b><br>raw(df) · fe(df)<br>build(name, df)", 380, 240, 300, 70, YELLOW)
    cv = p.box("<b>src/cv.py</b><br>make_folds(y) → fold id per row", 720, 240, 300, 70, PURPLE)
    models = p.box("<b>src/models.py</b><br>LGBM_PARAMS<br>EARLY_STOPPING_ROUNDS", 1060, 240, 300, 70, ORANGE)
    p.edge(data, config, dashed=True, extra="exitX=0.5;exitY=0;entryX=0;entryY=0.5;")
    p.edge(feats, config, dashed=True, extra="exitX=0.5;exitY=0;entryX=0.095;entryY=1;")
    p.edge(cv, config, dashed=True, extra="exitX=0.5;exitY=0;entryX=0.738;entryY=1;")
    p.edge(models, config, dashed=True, extra="exitX=0.5;exitY=0;entryX=1;entryY=0.5;")

    # row 2: scripts
    eda = p.box("<b>src/eda.py</b><br>single-feature AUC, segments,<br>drift → <b>reports/eda.md</b>", 40, 400, 300, 74, GREEN)
    train = p.box("<b>src/train.py</b>&nbsp;&nbsp;<font face='Courier New'>--exp --features --lr --quick</font><br>"
                  "5-fold LightGBM with early stopping → OOF ROC AUC<br>"
                  "writes <b>outputs/&lt;exp&gt;/oof.npy, test.npy</b> (git-ignored)<br>"
                  "and <b>experiments/&lt;exp&gt;.json</b> (scores, params, feature importance)", 380, 400, 640, 100, ORANGE)
    submit = p.box("<b>src/submit.py</b>&nbsp;&nbsp;<font face='Courier New'>exp [exp …] --name</font><br>"
                   "rank-blend · uses data.py to check ids/order<br>writes <b>submissions/&lt;name&gt;.csv</b><br>"
                   "and experiments/blend_*.json", 1060, 400, 300, 100, RED)
    p.edge(data, eda, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    p.edge(data, train, extra="exitX=0.85;exitY=1;entryX=0.06;entryY=0;")
    p.edge(feats, train, extra="exitX=0.5;exitY=1;entryX=0.234;entryY=0;")
    p.edge(cv, train, extra="exitX=0.5;exitY=1;entryX=0.766;entryY=0;")
    p.edge(models, train, extra="exitX=0.15;exitY=1;entryX=0.95;entryY=0;")
    p.edge(train, submit, "outputs/*.npy", extra="exitX=1;exitY=0.5;entryX=0;entryY=0.5;")

    # row 3: tooling
    make = p.box("<b>Makefile</b><br>make setup · data · eda · smoke · train · submit", 40, 580, 300, 64, GREY,
                 extra="dashed=1;")
    diagram = p.box("<b>docs/build_diagram.py</b><br>reads experiments/*.json<br>→ <b>docs/pipeline.drawio</b> (this file)",
                    1060, 580, 300, 70, BLUE)
    p.edge(make, eda, dashed=True, extra="exitX=0.5;exitY=0;entryX=0.5;entryY=1;")
    p.edge(make, train, dashed=True, extra="exitX=1;exitY=0.5;entryX=0.1;entryY=1;")
    p.edge(submit, diagram, "experiments/*.json", dashed=True, extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
    p.edge(train, diagram, "experiments/*.json", dashed=True, extra="exitX=0.9;exitY=1;entryX=0;entryY=0.5;")

    p.box("<b>Git-ignored</b> (local only)<br>data/ · outputs/ · submissions/ · .venv/", 40, 700, 420, 50, GREY)
    p.box("<b>Tracked</b><br>src/ · docs/ · reports/ · experiments/ · README.md · Makefile", 500, 700, 520, 50, GREY)
    return p


def main():
    singles, blends = load_experiments()
    mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device")
    mxfile.append(pipeline_page(singles, blends).to_xml("pipeline"))
    mxfile.append(code_map_page().to_xml("code-map"))
    ET.indent(mxfile)
    OUT.write_text(ET.tostring(mxfile, encoding="unicode") + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(singles)} experiments, {len(blends)} blends)")


if __name__ == "__main__":
    main()
