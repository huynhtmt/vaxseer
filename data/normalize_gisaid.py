#!/usr/bin/env python3
"""
normalize_gisaid.py — Robust GISAID normalizer (Option C header)

Features:
 - Repairs enriched FASTA headers (handles '_|_', '|_|', duplicated blocks, 'A_/_H3N2')
 - Parses header fields: EPI_ISL, Strain, Type(A/B), Marker (A/HxNy or B lineage), Date
 - Falls back to metadata (Virus name / Country / Subtype / Lineage / Date) when missing
 - Guarantees canonical Option C header: >EPI_ISL|Strain|Country|Type|Marker|Date|HA|
 - Writes:
     * ha.mapped.fasta (repaired headers, unfiltered; preview this in UI)
     * author-style filtered FASTA in ha_processed/... for the scoring pipeline
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


# ======================================================================
# -------------------------- HEADER REPAIR ------------------------------
# ======================================================================

CUSTOM_PIPE_TOKENS = ("_|_", "|_|", "__|__", "|__|")

# Tolerant identification patterns
EPI_PAT     = re.compile(r"\bEPI[_-]ISL[_-]?\d+\b", re.IGNORECASE)
DATE_PAT    = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")

# Strain like A/Singapore/SGH0374/2018 or B/Manitoba/RV02743-25/2025
STRAIN_PAT  = re.compile(r"\b[AB][ _/]?/(?:[A-Za-z0-9._-]+/){1,3}\d{4}\b", re.IGNORECASE)

# A subtype A/H3N2 (tolerant to underscores/spaces around '/')
SUBTYPE_PAT = re.compile(r"\bA[ _/]?/?H\d+N\d+\b", re.IGNORECASE)

# B lineage/clade patterns: V1A(.x), 3C(.x), Victoria, Yamagata, C1, C2, …
LINEAGE_PAT = re.compile(
    r"\b(?:V1A(?:\.[0-9A-Za-z.]+)*|3C(?:\.[0-9A-Za-z.]+)*|Victoria|Yamagata|C1A?|C2A?)\b",
    re.IGNORECASE
)

TYPE_PAT    = re.compile(r"\b(?:A|B)\b", re.IGNORECASE)


def _normalize_slash_artifacts(s: str) -> str:
    """Fix enriched tokens like 'A_/_H3N2' → 'A/H3N2', collapse underscores."""
    s = s.replace("_/_", "/")
    s = re.sub(r"_{2,}", "_", s)
    s = re.sub(r"_+/(?=[A-Za-z0-9])", "/", s)
    s = re.sub(r"/_+", "/", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def collapse_custom_pipes(h: str) -> str:
    """Normalize GISAID enriched GUI encoding into a clean '|' separated string."""
    for tok in CUSTOM_PIPE_TOKENS:
        h = h.replace(tok, "|")
    h = _normalize_slash_artifacts(h)
    h = re.sub(r"\|\s*\|\s*", "|", h)
    return h.strip().strip("|")


def squash_repeated_blocks(tokens: list[str]) -> list[str]:
    """If the header block repeats (2–4×), keep first block; else order‑preserving unique."""
    n = len(tokens)
    for p in range(4, min(12, max(4, n // 2)) + 1):
        if n % p == 0:
            block = tokens[:p]
            if all(tokens[i*p:(i+1)*p] == block for i in range(n // p)):
                return block
    seen = set()
    out = []
    for t in tokens:
        if t not in seen and t != "_":
            seen.add(t)
            out.append(t)
    return out


def parse_header_fields(collapsed: str) -> dict:
    """Extract epi, strain, type, marker (subtype or lineage), date."""
    epi    = (EPI_PAT.search(collapsed).group(0).replace("-", "_").upper()
              if EPI_PAT.search(collapsed) else "")
    date   = (DATE_PAT.search(collapsed).group(0) if DATE_PAT.search(collapsed) else "")
    strain = (STRAIN_PAT.search(collapsed).group(0) if STRAIN_PAT.search(collapsed) else "")

    subtype= (SUBTYPE_PAT.search(collapsed).group(0).upper().replace(" ", "")
              if SUBTYPE_PAT.search(collapsed) else "")
    typ_m  = TYPE_PAT.search(collapsed)
    typ    = typ_m.group(0).upper() if typ_m else ""
    line   = (LINEAGE_PAT.search(collapsed).group(0)
              if LINEAGE_PAT.search(collapsed) else "")

    # Prefer A/HxNy for A; else B lineage; else whichever we have
    marker = (subtype if (typ == "A" and subtype)
              else (line if typ == "B" else (subtype or line)))

    return dict(epi=epi, strain=strain, typ=typ, marker=marker, date=date)


# ======================================================================
# ----------------------- METADATA UTILITIES ----------------------------
# ======================================================================

def detect_delimiter(first_line: str) -> str:
    return ',' if first_line.count(',') >= first_line.count(';') else ';'


def read_metadata(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in ('.xls', '.xlsx'):
        # pandas will use openpyxl engine if available
        return pd.read_excel(path)

    raw = path.read_bytes()
    text = None
    for enc in ('utf-8-sig', 'utf-8', 'cp1252', 'latin-1', 'iso-8859-1'):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        text = raw.decode('latin-1', errors='replace')

    lines = text.splitlines()
    delim = detect_delimiter(lines[0] if lines else "")
    reader = csv.reader(io.StringIO(text), delimiter=delim)
    rows = list(reader)
    buf = io.StringIO()
    csv.writer(buf).writerows(rows)
    buf.seek(0)
    return pd.read_csv(buf)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    for cand in ["Collection_Date", "collection_date", "Date", "date", "Collection date"]:
        if cand in df.columns:
            col = cand
            break
    else:
        raise ValueError("No date-like column found.")
    def fix(d):
        d = str(d).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d): return d
        if re.match(r"^\d{4}-\d{2}$", d):       return d + "-15"
        if re.match(r"^\d{4}$", d):             return d + "-06-30"
        return d
    df[col] = df[col].apply(fix)
    df["__date_norm__"] = pd.to_datetime(df[col], errors="coerce")
    return df


def find_first(df: pd.DataFrame, candidates) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def build_iso_to_seg_map(df: pd.DataFrame) -> dict:
    if "Isolate_Id" not in df.columns or "HA Segment_Id" not in df.columns:
        return {}
    def left(x): return str(x).split("|", 1)[0].strip()
    iso = (df["Isolate_Id"].astype(str).str.replace("-", "_", regex=False)
                                .str.upper().str.strip())
    seg = df["HA Segment_Id"].astype(str).map(left)
    return dict(zip(iso, seg))


def build_iso_meta_map(df: pd.DataFrame) -> dict:
    """
    Build a dictionary:
        iso_meta[EPI_ISL] = {
            'strain':  "A/City/ID/2019",
            'country': "Belgium",
            'type':    "A" or "B",
            'marker':  "A/H3N2" or "V1A.3a.2" etc.,
            'date':    "YYYY-MM-DD"
        }
    using tolerant column detection.
    """
    iso_col = find_first(df, ["Isolate_Id", "isolate_id", "epi_isl", "EPI_ISL"])
    if not iso_col:
        return {}

    # candidates for each attribute
    strain_col  = find_first(df, ["Virus name", "virus name", "Strain", "strain", "Isolate name"])
    country_col = find_first(df, ["Country", "country", "Location", "location"])
    type_col    = find_first(df, ["Type", "type"])
    subtype_col = find_first(df, ["Subtype", "subtype", "Segment Subtype", "segment_subtype"])
    lineage_col = find_first(df, ["Lineage", "lineage", "Clade", "clade", "Pango lineage"])  # tolerant
    date_col    = "__date_norm__"

    df_iso = (df[iso_col].astype(str).str.replace("-", "_", regex=False)
                                .str.upper().str.strip())
    iso_meta: dict[str, dict] = {}

    # Build
    for i, iso in enumerate(df_iso):
        if not iso:
            continue
        row = df.iloc[i]
        m = {}

        # Strain
        if strain_col:
            s = str(row[strain_col]).strip()
            # keep plausible A/... or B/...
            if re.search(r"^[AB]/", s, flags=re.IGNORECASE):
                m['strain'] = s

        # Country (prefer explicit column; else derive first token of Location)
        if country_col:
            c = str(row[country_col]).strip()
            # If 'Location' is like "Belgium/Brussels", keep first token as country
            if "/" in c:
                c = c.split("/", 1)[0].strip()
            m['country'] = c

        # Type
        t = None
        if type_col:
            t_raw = str(row[type_col]).strip().upper()
            if t_raw in ("A", "B"):
                t = t_raw
        if not t and 'strain' in m:
            # derive from strain
            first_char = m['strain'].strip()[:1].upper()
            if first_char in ("A", "B"):
                t = first_char
        if t:
            m['type'] = t

        # Marker (prefer lineage for B; subtype for A)
        marker = None
        if subtype_col:
            st = str(row[subtype_col]).strip()
            if re.search(r"H\d+N\d+", st, flags=re.IGNORECASE):
                marker = st.replace(" ", "").upper()
        if not marker and lineage_col:
            lg = str(row[lineage_col]).strip()
            if lg:
                marker = lg

        if marker:
            m['marker'] = marker

        # Date
        if date_col in df.columns:
            dt_val = df.iloc[i][date_col]
            if pd.notna(dt_val):
                m['date'] = pd.to_datetime(dt_val).strftime("%Y-%m-%d")

        iso_meta[iso] = m

    return iso_meta


# ======================================================================
# ------------------------ CANONICAL HEADER -----------------------------
# ======================================================================

def canonical_header(raw_header: str,
                     seg: str | None,
                     iso_meta: dict[str, dict] | None) -> str:
    """
    OPTION C canonical header (robust):
       >EPI_ISL|Strain|Country|Type|Marker|Date|HA|

    Uses header parsing first; fills missing fields from iso_meta (metadata).
    """
    collapsed = collapse_custom_pipes(raw_header)
    tokens = [t for t in collapsed.split("|") if t]
    tokens = squash_repeated_blocks(tokens)
    collapsed = "|".join(tokens)

    f = parse_header_fields(collapsed)

    # Guarantee EPI_ISL
    epi = f["epi"]
    if not epi:
        m = EPI_PAT.search(raw_header)
        if m:
            epi = m.group(0).replace("-", "_").upper()
        else:
            import hashlib
            epi = "NO_EPI_" + hashlib.md5(raw_header.encode()).hexdigest()[:12]

    # Collect payload
    meta = iso_meta.get(epi, {}) if iso_meta else {}

    strain  = f["strain"] or meta.get("strain", "")
    country = meta.get("country", "")
    typ     = f["typ"] or meta.get("type", "")
    marker  = f["marker"] or meta.get("marker", "")
    date    = f["date"] or meta.get("date", "")

    # Build Option C fields (skip empties gracefully)
    fields = [epi]
    if strain:  fields.append(strain)
    if country: fields.append(country)
    if typ:     fields.append(typ)
    if marker:  fields.append(marker)
    if date:    fields.append(date)
    fields.append("HA")

    return ">" + "|".join(fields) + "|"


# ======================================================================
# ------------------------ REMAP FASTA HEADERS --------------------------
# ======================================================================

def remap_fasta_headers(fasta_in: Path, fasta_out: Path,
                        iso2seg: dict, iso_meta: dict) -> dict:
    """
    Repair enriched headers, extract EPI_ISL, and emit Option C headers.
    Segment ID is NOT included in Option C header (not reliable in modern GISAID),
    but iso2seg is still computed for stats.
    """
    epi_re = re.compile(r"(EPI[_-]ISL[_-]\d+)", re.IGNORECASE)
    stats = {"total": 0, "with_iso": 0, "no_iso": 0, "mapped_seg": 0, "repaired": 0}

    with open(fasta_in, "r", errors="ignore") as fin, \
         open(fasta_out, "w") as fout:

        for line in fin:
            if not line.startswith(">"):
                fout.write(line)
                continue

            stats["total"] += 1
            raw = line[1:].strip()

            m = epi_re.search(raw)
            if not m:
                stats["no_iso"] += 1
                out = canonical_header(raw, None, iso_meta)
                fout.write(out + "\n")
                stats["repaired"] += 1
                continue

            iso = m.group(1).replace("-", "_").upper()
            stats["with_iso"] += 1

            seg = iso2seg.get(iso)
            if seg:
                stats["mapped_seg"] += 1

            out = canonical_header(raw, seg, iso_meta)
            fout.write(out + "\n")
            stats["repaired"] += 1

    return stats


# ======================================================================
# --------------------- AUTHOR COMPATIBLE FILTERING ---------------------
# ======================================================================

def infer_window(df: pd.DataFrame) -> tuple[str, str]:
    dt = df["__date_norm__"].dropna()
    if dt.empty:
        return "2019-01", "2019-12"
    return dt.min().strftime("%Y-%m"), dt.max().strftime("%Y-%m")


def x_ratio(s: str) -> float:
    s = str(s).upper()
    return s.count("X") / len(s) if s else 1.0


def drop_metadata_dupes(df: pd.DataFrame, acc_col: str, date_col: str = "__date_norm__") -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    df = df.sort_values([acc_col, date_col])
    return df.drop_duplicates(acc_col, keep="first")


def build_author_fasta(meta_csv: Path, fasta_path: Path, out_root: Path,
                       subtype: str | None = None,
                       year: int | None = None,
                       host_value: str = "human",
                       minBinSize: int = 1000,
                       lenQuantile: float = 0.2,
                       minCnt: int = 5,
                       max_x_ratio: float = 1.0) -> dict:

    meta = pd.read_csv(meta_csv)
    meta["__date_norm__"] = pd.to_datetime(meta["__date_norm__"], errors="coerce")

    host_col    = find_first(meta, ["host", "Host"])
    subtype_col = find_first(meta, ["subtype", "Subtype", "Segment Subtype"])
    acc_col     = find_first(meta, ["accession_id", "Accession ID", "strain", "Virus name"]) or meta.columns[0]

    if subtype is None:
        if subtype_col:
            s = str(meta[subtype_col].astype(str).mode().iloc[0]).lower()
            subtype = "a_h3n2" if "h3n2" in s else "a_h1n1"
        else:
            subtype = "a_h3n2"

    meta["__host__"] = meta[host_col].astype(str).str.lower() if host_col else "human"
    meta["__subtype__"] = subtype

    # Window
    if year:
        start = f"{year-3}-02"
        end   = f"{year}-02"
    else:
        s_inf, e_inf = infer_window(meta)
        e_y, e_m = map(int, e_inf.split("-"))
        start = f"{e_y-3:04d}-{e_m:02d}"
        end   = f"{e_y:04d}-{e_m:02d}"

    print(f"[author] Window = {start} → {end} (9999M), subtype={subtype}, host={host_value}")

    s_y, s_m = map(int, start.split("-"))
    e_y, e_m = map(int, end.split("-"))

    def in_w(d):
        if pd.isna(d):
            return False
        return (s_y, s_m) <= (d.year, d.month) <= (e_y, e_m)

    meta = meta[(meta["__host__"] == host_value.lower()) &
                (meta["__subtype__"] == subtype) &
                (meta["__date_norm__"].apply(in_w))]

    if meta.empty:
        raise ValueError("No metadata rows after filters.")

    meta = drop_metadata_dupes(meta, acc_col)

    # Load repaired FASTA
    epi_re = re.compile(r"(EPI[_-]ISL[_-]\d+)", re.IGNORECASE)
    fasta_records = list(SeqIO.parse(str(fasta_path), "fasta"))

    # Build lookup maps from repaired FASTA
    iso_map = {}
    for rec in fasta_records:
        hdr = rec.id
        m = epi_re.search(hdr)
        if m:
            iso = m.group(1).replace("-", "_").upper()
            iso_map[iso] = rec

    iso_col = find_first(meta, ["isolate_id", "epi_isl"])
    if not iso_col:
        raise ValueError("Metadata lacks Isolate_Id / EPI_ISL column for matching.")

    iso_keys = (meta[iso_col].astype(str).str.replace("-", "_", regex=False).str.upper().str.strip())

    matched = []
    for i, k in iso_keys.items():
        if k in iso_map:
            matched.append((iso_map[k], str(meta.loc[i, acc_col])))

    if not matched:
        raise ValueError("No metadata matched repaired FASTA (EPI_ISL).")

    # Unique by accession (stable)
    uniq = {}
    for rec, acc in matched:
        uniq.setdefault(acc, rec)

    accs = list(uniq.keys())
    seqs = [uniq[a] for a in accs]

    # length filter
    lengths = np.array([len(str(r.seq)) for r in seqs])
    thr = int(np.quantile(lengths, lenQuantile))
    keep = lengths >= thr
    accs = [a for a, k in zip(accs, keep) if k]
    seqs = [s for s, k in zip(seqs, keep) if k]

    # X‑ratio filter
    if max_x_ratio < 1.0:
        mask = [(x_ratio(str(r.seq)) <= max_x_ratio) for r in seqs]
        accs = [a for a, k in zip(accs, mask) if k]
        seqs = [s for s, k in zip(seqs, mask) if k]

    # exact AA variant minCnt
    seq_strs = [str(r.seq) for r in seqs]
    counts = pd.Series(seq_strs).value_counts()
    keep_set = set(counts[counts >= minCnt].index)
    mask = [s in keep_set for s in seq_strs]
    accs = [a for a, k in zip(accs, mask) if k]
    seqs = [s for s, k in zip(seqs, mask) if k]

    out_dir = out_root / f"{start}_to_{end}_9999M" / subtype
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fa  = out_dir / "human_minBinSize1000_lenQuantile0.2_minCnt5.fasta"
    out_ids = out_dir / "human_minBinSize1000_lenQuantile0.2_minCnt5.ids"

    # Rename records to accession (author behavior)
    for rec, acc in zip(seqs, accs):
        rec.id = acc
        rec.name = acc
        rec.description = acc

    SeqIO.write(seqs, str(out_fa), "fasta")
    with open(out_ids, "w", encoding="utf-8") as f:
        f.write("\n".join(accs))

    return {"fasta": str(out_fa), "ids": str(out_ids), "n": len(seqs),
            "min_len": thr, "window": f"{start}_to_{end}_9999M", "subtype": subtype}


# ======================================================================
# -------------------------------- MAIN --------------------------------
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--fasta",    required=True)
    ap.add_argument("--outdir",   required=True)
    ap.add_argument("--year",     type=int, default=None)
    ap.add_argument("--subtype",  choices=["a_h3n2", "a_h1n1"], default=None)
    ap.add_argument("--host",     default="human")
    ap.add_argument("--max-x-ratio", type=float, default=1.0)
    ap.add_argument("--min-bin-size", type=int, default=1000)
    ap.add_argument("--len-quantile", type=float, default=0.2)
    ap.add_argument("--min-cnt", type=int, default=5)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Normalize metadata
    df = read_metadata(Path(args.metadata))
    df = normalize_dates(df)
    norm_meta = outdir / "metadata.norm.csv"
    df.to_csv(norm_meta, index=False)
    print(f"[normalize] wrote {norm_meta}")

    # 2) Repair headers & build ha.mapped.fasta (unfiltered)
    iso2seg  = build_iso_to_seg_map(df)
    iso_meta = build_iso_meta_map(df)

    mapped = outdir / "ha.mapped.fasta"
    stats  = remap_fasta_headers(Path(args.fasta), mapped, iso2seg, iso_meta)
    print("[normalize] header mapping & repair:", stats)
    print(f"[normalize] repaired FASTA (unfiltered): {mapped}")

    # 3) Build author-style filtered FASTA (dominance input)
    save_dir = outdir / "ha_processed"
    res = build_author_fasta(
        meta_csv     = norm_meta,
        fasta_path   = mapped,
        out_root     = save_dir,
        subtype      = args.subtype,
        year         = args.year,
        host_value   = args.host,
        minBinSize   = args.min_bin_size,
        lenQuantile  = args.len_quantile,
        minCnt       = args.min_cnt,
        max_x_ratio  = args.max_x_ratio
    )

    print(f"[normalize] author FASTA: {res['fasta']}")
    print(f"[normalize] ids:          {res['ids']}")
    print(f"[normalize] N={res['n']} (min_len={res['min_len']}) window={res['window']} subtype={res['subtype']}")


if __name__ == "__main__":
    main()
