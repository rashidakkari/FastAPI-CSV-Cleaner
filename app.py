from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
from datetime import datetime

app = FastAPI(title="CSV Cleaner API")

YES_SET = {"yes","y","true","1"}
NO_SET  = {"no","n","false","0"}

def detect_date_columns(df: pd.DataFrame):
    # Name hint
    name_hits = [c for c in df.columns if any(k in c.lower() for k in ["date","time","dob"])]
    # Parseability hint (>=50% non-null parseable)
    parse_hits = []
    sample_n = min(len(df), 200)
    sample_df = df.sample(sample_n, random_state=42) if len(df) > 0 else df
    for c in df.columns:
        try:
            parsed = pd.to_datetime(sample_df[c], errors="coerce", infer_datetime_format=True)
            non_null = parsed.notna().sum()
            if sample_n and (non_null / max(sample_n,1)) >= 0.5:
                parse_hits.append(c)
        except Exception:
            pass
    return list(dict.fromkeys(name_hits + parse_hits))

def mode_or_blank(series: pd.Series):
    vals = series.dropna()
    if vals.empty:
        return None
    return vals.mode().iloc[0]

def clean_dataframe(df: pd.DataFrame):
    logs = []
    logs.append({"step":"load","detail":f"Loaded {len(df)} rows, {len(df.columns)} columns."})

    # Trim whitespace for object columns
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})
    logs.append({"step":"trim_whitespace","detail":f"Trimmed whitespace in {len(obj_cols)} text columns."})

    # Lowercase email columns
    email_cols = [c for c in df.columns if "email" in c.lower()]
    for c in email_cols:
        df[c] = df[c].astype(str).str.strip().str.lower().replace({"nan": np.nan})
    if email_cols:
        logs.append({"step":"standardize_email","detail":f"Lowercased emails in columns: {', '.join(email_cols)}"})

    # Normalize Yes/No flags on object-like columns
    flags_normed = []
    for c in obj_cols:
        uniq = set(str(x).strip().lower() for x in df[c].dropna().unique().tolist())
        if uniq and uniq.issubset(YES_SET.union(NO_SET)):
            df[c] = df[c].apply(lambda x: "Yes" if str(x).strip().lower() in YES_SET else ("No" if str(x).strip().lower() in NO_SET else x))
            flags_normed.append(c)
    if flags_normed:
        logs.append({"step":"normalize_flags","detail":f"Normalized Yes/No flags in: {', '.join(flags_normed)}"})

    # Dates
    date_cols = detect_date_columns(df)
    changed_dates = []
    for c in date_cols:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            df[c] = parsed.dt.date.astype("string")
            changed_dates.append(c)
        except Exception:
            pass
    if changed_dates:
        logs.append({"step":"standardize_dates","detail":f"Standardized dates in: {', '.join(changed_dates)}"})

    # Missing values strategy
    nulls_before = int(df.isna().sum().sum())
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            med = df[c].median(skipna=True)
            df[c] = df[c].fillna(med)
        elif c in date_cols:
            # leave blanks (already string "NaT"/<NA>); replace to empty string for clarity
            df[c] = df[c].replace({"<NA>": None})
        else:
            fill = mode_or_blank(df[c])
            if fill is not None:
                df[c] = df[c].fillna(fill)
    nulls_after = int(df.isna().sum().sum())
    logs.append({"step":"fill_missing","detail":f"Missing values filled: before={nulls_before}, after={nulls_after}."})

    # Dedupe full rows
    before = len(df)
    df = df.drop_duplicates(keep="first")
    removed = before - len(df)
    logs.append({"step":"deduplicate","detail":f"Removed {removed} duplicate rows."})

    # Reorder columns: id first then alphabetical
    cols = df.columns.tolist()
    if "id" in cols:
        cols.remove("id")
        cols = ["id"] + sorted(cols, key=str.lower)
    else:
        cols = sorted(cols, key=str.lower)
    df = df[cols]
    logs.append({"step":"reorder_columns","detail":f"Column order set to: {', '.join(cols)}"})

    return df, logs

@app.post("/clean")
async def clean_endpoint(data: UploadFile = File(...)):
    filename = data.filename or "data.csv"
    raw = await data.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    # Try UTF-8 first, fallback latin-1
    try:
        df = pd.read_csv(io.BytesIO(raw), dtype=object)
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), dtype=object, encoding="latin-1")

    cleaned_df, logs = clean_dataframe(df)

    # Convert to JSON-rows
    cleaned_rows = cleaned_df.replace({np.nan: None}).to_dict(orient="records")
    fix_log_rows = logs

    return JSONResponse(content={
        "ok": True,
        "cleaned_rows": cleaned_rows,
        "fix_log_rows": fix_log_rows,
        "filename": filename,
        "stamp": datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    })
