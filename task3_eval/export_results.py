"""
Export Task-3 results to SQLite (results.db) + Excel (results.xlsx), plus print
example SQL queries.

Robust to synced/locked folders: SQLite is built in the system temp dir (where
file locking works), then copied into --out. Avoids the "disk I/O error" that
happens when SQLite writes directly inside a watched folder.

Run:
  python task3_eval/export_results2.py --task3 outputs/task3 --out outputs
  (needs: pip install openpyxl)
"""
import argparse, json, os, sqlite3, tempfile, shutil
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task3", default="outputs/task3")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames = {}
    for split in ("dev", "eval"):
        csv = os.path.join(args.task3, f"per_image_{split}.csv")
        if os.path.exists(csv):
            frames[split] = pd.read_csv(csv)

    tmp = tempfile.mkdtemp()
    tmp_db = os.path.join(tmp, "results.db")

    # ---- SQLite (built in temp, then copied) ----
    con = sqlite3.connect(tmp_db)
    for split, df in frames.items():
        df.to_sql(f"{split}_results", con, if_exists="replace", index=False)
        print(f"[sql] {split}_results ({len(df)} rows)")
    cur = con.cursor()
    def show(title, sql):
        print(f"\n--- {title} ---")
        try:
            for row in cur.execute(sql).fetchall():
                print("  ", row)
        except Exception as e:
            print("   (query failed:", e, ")")
    show("10 worst dev images by IoU",
         "SELECT stem, ROUND(iou,3) FROM dev_results ORDER BY iou ASC LIMIT 10;")
    show("Average metrics on dev (good GT only)",
         "SELECT ROUND(AVG(iou),3) iou, ROUND(AVG(prec),3) prec, ROUND(AVG(rec),3) rec "
         "FROM dev_results WHERE bad_gt=0;")
    show("IoU by egg size bucket (dev)",
         "SELECT CASE WHEN gt_area<0.08 THEN 'small' WHEN gt_area<0.2 THEN 'medium' "
         "ELSE 'large' END size_bucket, COUNT(*) n, ROUND(AVG(iou),3) avg_iou "
         "FROM dev_results WHERE bad_gt=0 GROUP BY size_bucket ORDER BY avg_iou;")
    con.commit(); con.close()
    shutil.copy(tmp_db, os.path.join(args.out, "results.db"))
    print(f"\n[sql] copied -> {os.path.join(args.out, 'results.db')}")

    # ---- Excel (built in temp, then copied) ----
    summary = None
    mj = os.path.join(args.task3, "metrics.json")
    if os.path.exists(mj):
        m = json.load(open(mj))
        rows = [{"split": s, "group": g, **m[s][g]}
                for s in ("dev", "eval") if s in m for g in ("all", "good_gt") if g in m[s]]
        summary = pd.DataFrame(rows)
    try:
        tmp_xlsx = os.path.join(tmp, "results.xlsx")
        with pd.ExcelWriter(tmp_xlsx, engine="openpyxl") as xw:
            if summary is not None:
                summary.to_excel(xw, sheet_name="summary", index=False)
            for split, df in frames.items():
                df.to_excel(xw, sheet_name=split, index=False)
        shutil.copy(tmp_xlsx, os.path.join(args.out, "results.xlsx"))
        print(f"[xlsx] copied -> {os.path.join(args.out, 'results.xlsx')} (sheets: summary + {list(frames)})")
    except ImportError:
        print("[xlsx] skipped — run: pip install openpyxl")

    shutil.rmtree(tmp, ignore_errors=True)
    print("[done]")

if __name__ == "__main__":
    main()
