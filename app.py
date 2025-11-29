# app.py — Data Integrity Lab (HOLCIM Edition) — sans sidebar
# -----------------------------------------------------------------------------
from __future__ import annotations

import csv
import json
import re
import textwrap
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from string import Template

import pandas as pd
import streamlit as st

# --- compat rerun ---
def _rerun():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ----------------- Page config -----------------
st.set_page_config(page_title="Data Integrity Lab — HOLCIM", page_icon="🧪", layout="wide")

# ----------------- Style (Holcim) -----------------
HOLCIM_GREEN = "#94C12E"
HOLCIM_BLUE = "#0073AF"
HOLCIM_DARKBLUE = "#1D4370"
HOLCIM_GRAY = "#918F90"
HOLCIM_BLACK = "#231F20"

STYLE = """
<style>
:root{
  --bg:#0f1722; --panel:#0f1c33; --card:#0b162a; --muted:#9fb0c8;
  --brand-green:%(green)s; --brand-blue:%(blue)s; --brand-dark:%(dark)s;
  --brand-gray:%(gray)s; --brand-black:%(black)s;
}
html,body{background:var(--bg)} .block-container{padding-top:2rem;} section.main>div{max-width:1450px;}
.hero{border-radius:18px;padding:24px 22px;margin-bottom:18px;color:#fff;
      background:linear-gradient(90deg,var(--brand-green),var(--brand-blue));
      box-shadow:0 10px 30px rgba(0,0,0,.25);}
.hero h1{margin:0;font-weight:800;letter-spacing:.2px} .hero p{margin:.2rem 0 0;opacity:.92}
.card{background:var(--card);border:1px solid #1f2a44;border-radius:18px;padding:16px}
.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:8px 0 18px}
.metric{background:var(--panel);border:1px solid #1f2a44;border-radius:16px;padding:14px;color:#e2e8f0}
.metric .label{font-size:12px;color:var(--muted)} .metric .value{font-size:22px;font-weight:700}
.badge{display:inline-flex;gap:6px;align-items:center;font-size:12px;border:1px solid #1f2a44;color:#e2e8f0;padding:4px 8px;border-radius:999px;background:#0b1528}
.dataframe{border-radius:12px;overflow:hidden} footer{visibility:hidden}
</style>
""" % {"green": HOLCIM_GREEN, "blue": HOLCIM_BLUE, "dark": HOLCIM_DARKBLUE, "gray": HOLCIM_GRAY, "black": HOLCIM_BLACK}
st.markdown(STYLE, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <h1>🧪 Data Integrity Lab — HOLCIM</h1>
  <p>Contrôles de qualité des données — profils, correspondances, règles métier — aux couleurs du Groupe.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------- Helpers -----------------
@dataclass
class ReadResult:
    df: pd.DataFrame
    sep: Optional[str]
    meta: str

@st.cache_data(show_spinner=False)
def sniff_delimiter(sample: str, fallback: str = ",") -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        return fallback

@st.cache_data(show_spinner=False)
def read_any(uploaded_file) -> Optional[ReadResult]:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        pos = uploaded_file.tell()
        sample_bytes = uploaded_file.read(4096)
        uploaded_file.seek(pos)
        sample_text = sample_bytes.decode("utf-8", errors="ignore") if sample_bytes else ""
        sep = sniff_delimiter(sample_text)
        for enc in ("utf-8", "latin-1"):
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=sep, dtype=object, keep_default_na=False, encoding=enc)
                return ReadResult(df=df, sep=sep, meta=f"CSV • enc={enc} • sep='{sep}'")
            except UnicodeDecodeError:
                continue
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep=sep, dtype=object, keep_default_na=False)
        return ReadResult(df=df, sep=sep, meta=f"CSV • sep='{sep}'")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, dtype=object, keep_default_na=False)
        return ReadResult(df=df, sep=None, meta="Excel")
    st.warning("Format non supporté (utilisez CSV ou XLSX).")
    return None

def normalize_value(x: str, to_lower: bool, strip_ws: bool, remove_acc: bool, collapse_ws: bool) -> str:
    s = "" if x is None else str(x)
    if strip_ws: s = s.strip()
    if collapse_ws: s = " ".join(s.split())
    if remove_acc:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    if to_lower: s = s.lower()
    return s

def df_to_csv_bytes(df: pd.DataFrame, sep: str) -> bytes:
    return df.to_csv(index=False, sep=sep).encode("utf-8")

# internes (presets & logs, conservés pour modules)
HISTORY_KEY = "__history"
if HISTORY_KEY not in st.session_state: st.session_state[HISTORY_KEY] = []
def log_run(module: str, meta: dict):
    st.session_state[HISTORY_KEY].append({"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "module": module, **meta})

PRESETS_KEY = "__presets"
if PRESETS_KEY not in st.session_state: st.session_state[PRESETS_KEY] = {}
def save_preset(name: str, data: dict):
    if name: st.session_state[PRESETS_KEY][name] = data

# opérations cœur
def anti_join(left: pd.DataFrame, right: pd.DataFrame, left_keys: List[str], right_keys: List[str]) -> pd.DataFrame:
    right_keys_only = right[right_keys].drop_duplicates()
    merged = left.merge(right_keys_only, left_on=left_keys, right_on=right_keys, how="left", indicator=True)
    return left.loc[merged["_merge"].eq("left_only")].copy()

def duplicates(df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    if not keys: return pd.DataFrame()
    grp = df.groupby(keys, dropna=False, as_index=False).size().rename(columns={"size":"__count"})
    d_keys = grp[grp["__count"] > 1][keys]
    if d_keys.empty: return pd.DataFrame(columns=list(df.columns)+["__dup_count"])[:0]
    res = d_keys.merge(df, on=keys, how="left").merge(grp, on=keys, how="left").rename(columns={"__count":"__dup_count"})
    return res

def foreign_key_misses(child: pd.DataFrame, parent: pd.DataFrame, child_keys: List[str], parent_keys: List[str]) -> pd.DataFrame:
    parent_keys_only = parent[parent_keys].drop_duplicates()
    m = child.merge(parent_keys_only, left_on=child_keys, right_on=parent_keys, how="left", indicator=True)
    return child.loc[m["_merge"].eq("left_only")].copy()

# ----------------- UI (onglets) -----------------
anomaly_tabs = st.tabs([
    "📊 Profil de colonnes",
    "🔗 Anti-jointure",
    "🧭 Doublons",
    "🗂️ Références manquantes",
    "✅ Règles de validation",
])

# 0) Profil de colonnes
with anomaly_tabs[0]:
    st.subheader("📊 Profil de colonnes d'un fichier")
    f = st.file_uploader("Fichier (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="prof_f")
    if f:
        r = read_any(f)
        if r:
            st.markdown(
                '<div class="metric-grid">'
                + f"<div class='metric'><div class='label'>Lignes</div><div class='value'>{len(r.df):,}</div><div class='label'>{r.meta}</div></div>"
                + f"<div class='metric'><div class='label'>Colonnes</div><div class='value'>{len(r.df.columns)}</div></div>"
                + '<div class="metric"></div><div class="metric"></div>'
                + "</div>", unsafe_allow_html=True)

            df = r.df.copy()
            rows = []
            for c in df.columns:
                s = df[c].astype(str)
                total = len(s); empty = int((s.str.len()==0).sum()); nunique = int(s.nunique(dropna=False))
                sample_vals = ", ".join(map(str, s.drop_duplicates().head(5).tolist()))
                nums = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
                is_num = nums.notna().mean() > 0.9
                rows.append({"colonne":c, "type":"num" if is_num else "str", "lignes":total,
                             "% vides":round(100*empty/total,2), "valeurs uniques":nunique,
                             "longueur moyenne":round(s.str.len().mean(),2),
                             "min":nums.min() if is_num else None, "max":nums.max() if is_num else None,
                             "exemples":sample_vals})
            profile_df = pd.DataFrame(rows)
            st.dataframe(profile_df, use_container_width=True)

            if st.button("📝 Générer un rapport"):
                tpl = Template("""
                <html><head><meta charset='utf-8'>
                <style>body{font-family:Arial, sans-serif; padding:24px}
                h1{margin:0 0 8px}.sub{color:#666;margin-bottom:18px}
                table{border-collapse:collapse;width:100%}
                th,td{border:1px solid #ddd;padding:8px;font-size:12px}
                th{background:#f4f6f8;text-align:left}</style></head><body>
                <h1>Data Integrity Lab — Rapport de profil</h1>
                <div class='sub'>Généré: $now — Fichier: $name</div>$table</body></html>""")
                html = tpl.safe_substitute(now=datetime.now().strftime("%Y-%m-%d %H:%M"),
                                           name=f.name, table=profile_df.to_html(index=False))
                st.download_button("⬇️ Télécharger le rapport (HTML)", html.encode("utf-8"),
                                   "profil.html", mime="text/html")
                try:
                    from reportlab.lib.pagesizes import A4
                    from reportlab.pdfgen import canvas
                    from reportlab.lib.units import cm
                    from reportlab.lib.utils import simpleSplit
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    c = canvas.Canvas(tmp.name, pagesize=A4); w, h = A4; y = h - 2*cm
                    c.setFont("Helvetica-Bold", 14); c.drawString(2*cm, y, "Data Integrity Lab — Rapport de profil"); y -= .8*cm
                    c.setFont("Helvetica", 10); c.drawString(2*cm, y, f"Fichier: {f.name} — Généré: {datetime.now().strftime('%Y-%m-%d %H:%M')}"); y -= 1*cm
                    c.setFont("Helvetica-Bold", 11); c.drawString(2*cm, y, "Résumé"); y -= .6*cm
                    c.setFont("Helvetica", 9); c.drawString(2*cm, y, f"Lignes: {len(df)} — Colonnes: {len(df.columns)}"); y -= .8*cm
                    c.setFont("Helvetica-Bold", 10); c.drawString(2*cm, y, "Colonnes (aperçu)"); y -= .6*cm
                    c.setFont("Helvetica", 8); headers = ["colonne","type","% vides","valeurs uniques","min","max"]
                    for _, row in profile_df[headers].head(25).iterrows():
                        line = " | ".join(str(row[h]) for h in headers)
                        for t in simpleSplit(line, "Helvetica", 8, w-4*cm):
                            if y < 2*cm: c.showPage(); y = h - 2*cm; c.setFont("Helvetica", 8)
                            c.drawString(2*cm, y, t); y -= .45*cm
                    c.save()
                    with open(tmp.name, "rb") as fpdf:
                        st.download_button("⬇️ Télécharger le rapport (PDF)", fpdf.read(),
                                           "profil.pdf", mime="application/pdf")
                except Exception:
                    st.info("Pour le PDF, installez 'reportlab'. Le HTML est disponible.")
            log_run("profil", {"fichier": f.name, "lignes": len(r.df), "colonnes": len(r.df.columns)})
    else:
        st.info("Charge un fichier pour obtenir son profil.")

# 1) Anti-jointure
with anomaly_tabs[1]:
    st.subheader("🔗 Lignes sans correspondance entre deux fichiers")
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("Fichier 1 (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="aj_f1")
    with c2:
        f2 = st.file_uploader("Fichier 2 (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="aj_f2")

    with st.expander("🎛️ Préréglage du module"):
        preset_names = sorted(st.session_state[PRESETS_KEY].keys())
        chosen = st.selectbox("Charger un preset", ["(aucun)"] + preset_names, index=0, key="aj_preset_select")

    if f1 and f2:
        r1 = read_any(f1); r2 = read_any(f2)
        if r1 and r2:
            st.markdown(
                '<div class="metric-grid">'
                + f"<div class='metric'><div class='label'>Fichier 1</div><div class='value'>{len(r1.df):,} lignes</div><div class='label'>{r1.meta}</div></div>"
                + f"<div class='metric'><div class='label'>Fichier 2</div><div class='value'>{len(r2.df):,} lignes</div><div class='label'>{r2.meta}</div></div>"
                + f"<div class='metric'><div class='label'>Colonnes F1</div><div class='value'>{len(r1.df.columns)}</div></div>"
                + f"<div class='metric'><div class='label'>Colonnes F2</div><div class='value'>{len(r2.df.columns)}</div></div>"
                + "</div>", unsafe_allow_html=True)

            s1, s2 = st.columns(2)
            with s1: keys1 = st.multiselect("Clés (Fichier 1)", list(r1.df.columns), key="aj_keys1")
            with s2: keys2 = st.multiselect("Clés (Fichier 2)", list(r2.df.columns), key="aj_keys2")

            st.markdown("<div class='badge'>Options de normalisation</div>", unsafe_allow_html=True)
            o1, o2, o3, o4 = st.columns(4)
            with o1: to_lower = st.toggle("Ignorer la casse", value=True, key="aj_lower")
            with o2: strip_ws = st.toggle("Trim", value=True, key="aj_trim")
            with o3: collapse_ws = st.toggle("Espaces multiples → 1", value=True, key="aj_collapse")
            with o4: remove_acc = st.toggle("Supprimer accents", value=False, key="aj_acc")

            if chosen != "(aucun)":
                cfg = st.session_state[PRESETS_KEY].get(chosen, {}).get("anti_join")
                if cfg:
                    keys1 = cfg.get("keys1", keys1); keys2 = cfg.get("keys2", keys2)
                    to_lower = cfg.get("to_lower", to_lower); strip_ws = cfg.get("strip_ws", strip_ws)
                    collapse_ws = cfg.get("collapse_ws", collapse_ws); remove_acc = cfg.get("remove_acc", remove_acc)

            if st.button("🚀 Lancer", type="primary", use_container_width=True):
                if not keys1 or not keys2 or len(keys1) != len(keys2):
                    st.error("Sélectionnez des colonnes et assurez la même cardinalité."); st.stop()
                df1, df2 = r1.df.copy(), r2.df.copy()
                n1, n2 = [], []
                for i, c in enumerate(keys1):
                    nn = f"__n1_{i}"; df1[nn] = df1[c].apply(lambda x: normalize_value(x, to_lower, strip_ws, remove_acc, collapse_ws)); n1.append(nn)
                for i, c in enumerate(keys2):
                    nn = f"__n2_{i}"; df2[nn] = df2[c].apply(lambda x: normalize_value(x, to_lower, strip_ws, remove_acc, collapse_ws)); n2.append(nn)
                with st.spinner("Calcul en cours…"):
                    only_in_1 = anti_join(df1, df2, n1, n2); only_in_2 = anti_join(df2, df1, n2, n1)
                only_in_1 = only_in_1[[c for c in only_in_1.columns if not c.startswith("__n")]]
                only_in_2 = only_in_2[[c for c in only_in_2.columns if not c.startswith("__n")]]
                st.markdown(
                    '<div class="metric-grid">'
                    + f"<div class='metric'><div class='label'>Sans correspondance depuis F1</div><div class='value'>{len(only_in_1):,}</div></div>"
                    + f"<div class='metric'><div class='label'>Sans correspondance depuis F2</div><div class='value'>{len(only_in_2):,}</div></div>"
                    + f"<div class='metric'><div class='label'>Total F1</div><div class='value'>{len(r1.df):,}</div></div>"
                    + f"<div class='metric'><div class='label'>Total F2</div><div class='value'>{len(r2.df):,}</div></div>"
                    + "</div>", unsafe_allow_html=True)
                st.write("**Aperçu — Fichier 1 sans correspondance**"); st.dataframe(only_in_1.head(500), use_container_width=True)
                st.write("**Aperçu — Fichier 2 sans correspondance**"); st.dataframe(only_in_2.head(500), use_container_width=True)
                st.subheader("Export"); out_sep = st.radio("Séparateur CSV", [",", ";", "\t", "|"], index=1, horizontal=True, key="aj_sep")
                colx, coly = st.columns(2)
                with colx:
                    st.download_button("⬇️ Télécharger — F1 sans correspondance", df_to_csv_bytes(only_in_1, out_sep),
                                       "unmatched_from_f1.csv", mime="text/csv")
                with coly:
                    st.download_button("⬇️ Télécharger — F2 sans correspondance", df_to_csv_bytes(only_in_2, out_sep),
                                       "unmatched_from_f2.csv", mime="text/csv")
                with st.expander("Enregistrer ce réglage comme preset"):
                    name = st.text_input("Nom du preset", key="aj_save_name")
                    if st.button("💾 Sauvegarder preset"): save_preset(name, {"anti_join":{
                        "keys1":keys1,"keys2":keys2,"to_lower":to_lower,"strip_ws":strip_ws,"collapse_ws":collapse_ws,"remove_acc":remove_acc}})
                log_run("anti_join", {"f1": f1.name, "f2": f2.name, "unmatched_from_f1": len(only_in_1), "unmatched_from_f2": len(only_in_2)})
    else:
        st.info("Déposez vos deux fichiers pour commencer.")

# 2) Doublons
with anomaly_tabs[2]:
    st.subheader("🧭 Détection de doublons dans un fichier")
    f = st.file_uploader("Fichier (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="dup_f")
    with st.expander("🎛️ Préréglage du module"):
        preset_names = sorted(st.session_state[PRESETS_KEY].keys())
        chosen = st.selectbox("Charger un preset", ["(aucun)"] + preset_names, index=0, key="dup_preset_select")
    if f:
        r = read_any(f)
        if r:
            st.markdown(
                '<div class="metric-grid">'
                + f"<div class='metric'><div class='label'>Lignes</div><div class='value'>{len(r.df):,}</div><div class='label'>{r.meta}</div></div>"
                + f"<div class='metric'><div class='label'>Colonnes</div><div class='value'>{len(r.df.columns)}</div></div>"
                + '<div class="metric"></div><div class="metric"></div>'
                + "</div>", unsafe_allow_html=True)
            keys = st.multiselect("Colonnes à considérer pour l'unicité", list(r.df.columns), key="dup_keys")
            if chosen != "(aucun)": 
                cfg = st.session_state[PRESETS_KEY].get(chosen, {}).get("duplicates")
                if cfg: keys = cfg.get("keys", keys)
            if st.button("🔍 Chercher les doublons", type="primary"):
                if not keys: st.warning("Sélectionnez au moins une colonne."); st.stop()
                with st.spinner("Analyse des doublons…"): res = duplicates(r.df, keys)
                st.write(f"**Doublons trouvés : {len(res):,} lignes**".replace(",", " "))
                st.dataframe(res.head(1000), use_container_width=True)
                out_sep = st.radio("Séparateur d’export", [",", ";", "\t", "|"], index=1, horizontal=True, key="dup_sep")
                st.download_button("⬇️ Télécharger les doublons", df_to_csv_bytes(res, out_sep), "duplicates.csv", mime="text/csv")
                with st.expander("Enregistrer ce réglage comme preset"):
                    name = st.text_input("Nom du preset", key="dup_save_name")
                    if st.button("💾 Sauvegarder preset"): save_preset(name, {"duplicates":{"keys":keys}})
                log_run("duplicates", {"fichier": f.name, "doublons": len(res)})
    else:
        st.info("Charge un fichier pour continuer.")

# 3) Références manquantes
with anomaly_tabs[3]:
    st.subheader("🗂️ Références manquantes (clé étrangère)")
    c1, c2 = st.columns(2)
    with c1: child = st.file_uploader("Table enfant (fact)", type=["csv", "xlsx", "xls"], key="fk_child")
    with c2: parent = st.file_uploader("Table parent (dimension/référentiel)", type=["csv", "xlsx", "xls"], key="fk_parent")
    with st.expander("🎛️ Préréglage du module"):
        preset_names = sorted(st.session_state[PRESETS_KEY].keys())
        chosen = st.selectbox("Charger un preset", ["(aucun)"] + preset_names, index=0, key="fk_preset_select")

    if child and parent:
        rc = read_any(child); rp = read_any(parent)
        if rc and rp:
            s1, s2 = st.columns(2)
            with s1: ck = st.multiselect("Clés dans l'enfant", list(rc.df.columns), key="fk_ck")
            with s2: pk = st.multiselect("Clés correspondantes dans le parent", list(rp.df.columns), key="fk_pk")

            st.markdown("<div class='badge'>Options de normalisation</div>", unsafe_allow_html=True)
            o1, o2, o3, o4 = st.columns(4)
            with o1: fk_lower = st.toggle("Ignorer la casse", value=True, key="fk_lower")
            with o2: fk_trim = st.toggle("Trim", value=True, key="fk_trim")
            with o3: fk_collapse = st.toggle("Espaces → 1", value=True, key="fk_collapse")
            with o4: fk_acc = st.toggle("Supprimer accents", value=False, key="fk_acc")

            if chosen != "(aucun)":
                cfg = st.session_state[PRESETS_KEY].get(chosen, {}).get("fk")
                if cfg:
                    ck = cfg.get("ck", ck); pk = cfg.get("pk", pk)
                    fk_lower = cfg.get("lower", fk_lower); fk_trim = cfg.get("trim", fk_trim)
                    fk_collapse = cfg.get("collapse", fk_collapse); fk_acc = cfg.get("accents", fk_acc)

            if st.button("🚀 Vérifier", type="primary"):
                if not ck or not pk or len(ck)!=len(pk): st.error("Sélectionnez des colonnes et assurez la même cardinalité."); st.stop()
                dfc, dfp = rc.df.copy(), rp.df.copy(); nck, npk = [], []
                for i, c in enumerate(ck): nn=f"__c{i}"; dfc[nn]=dfc[c].apply(lambda x: normalize_value(x,fk_lower,fk_trim,fk_acc,fk_collapse)); nck.append(nn)
                for i, c in enumerate(pk): nn=f"__p{i}"; dfp[nn]=dfp[c].apply(lambda x: normalize_value(x,fk_lower,fk_trim,fk_acc,fk_collapse)); npk.append(nn)
                with st.spinner("Recherche des références manquantes…"): misses = foreign_key_misses(dfc, dfp, nck, npk)
                misses = misses[[c for c in misses.columns if not c.startswith("__")]]
                st.write(f"**Références manquantes : {len(misses):,} lignes**".replace(",", " "))
                st.dataframe(misses.head(1000), use_container_width=True)
                out_sep = st.radio("Séparateur d’export", [",", ";", "\t", "|"], index=1, horizontal=True, key="fk_sep")
                st.download_button("⬇️ Télécharger les références manquantes", df_to_csv_bytes(misses, out_sep), "foreign_key_misses.csv", mime="text/csv")
                with st.expander("Enregistrer ce réglage comme preset"):
                    name = st.text_input("Nom du preset", key="fk_save_name")
                    if st.button("💾 Sauvegarder preset"): save_preset(name, {"fk":{"ck":ck,"pk":pk,"lower":fk_lower,"trim":fk_trim,"collapse":fk_collapse,"accents":fk_acc}})
                log_run("fk_missing", {"child": child.name, "parent": parent.name, "misses": len(misses)})
    else:
        st.info("Déposez les deux fichiers pour continuer.")

# 4) Règles de validation
with anomaly_tabs[4]:
    st.subheader("✅ Règles de validation d'un fichier")
    f = st.file_uploader("Fichier (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="val_f")
    with st.expander("🎛️ Préréglage du module"):
        preset_names = sorted(st.session_state[PRESETS_KEY].keys())
        chosen = st.selectbox("Charger un preset", ["(aucun)"] + preset_names, index=0, key="val_preset_select")
    if f:
        r = read_any(f)
        if r:
            st.markdown(
                '<div class="metric-grid">'
                + f"<div class='metric'><div class='label'>Lignes</div><div class='value'>{len(r.df):,}</div><div class='label'>{r.meta}</div></div>"
                + f"<div class='metric'><div class='label'>Colonnes</div><div class='value'>{len(r.df.columns)}</div></div>"
                + '<div class="metric"></div><div class="metric"></div>'
                + "</div>", unsafe_allow_html=True)

            target_col = st.selectbox("Colonne cible", list(r.df.columns), index=0, key="val_target")
            c1, c2, c3, c4 = st.columns(4)
            with c1: rule_not_null = st.checkbox("Non vide", key="val_notnull")
            with c2: min_v = st.text_input("Min (num.)", placeholder="ex: 0", key="val_min")
            with c3: max_v = st.text_input("Max (num.)", placeholder="ex: 100", key="val_max")
            with c4: regex = st.text_input("Regex", placeholder=r"^(?:A|B|C)$", key="val_regex")
            allowed = st.text_input("Valeurs autorisées (séparées par ,)", placeholder="A,B,C", key="val_allowed")

            if chosen != "(aucun)":
                cfg = st.session_state[PRESETS_KEY].get(chosen, {}).get("validation")
                if cfg:
                    target_col = cfg.get("target", target_col); rule_not_null = cfg.get("notnull", rule_not_null)
                    min_v = cfg.get("min", min_v); max_v = cfg.get("max", max_v)
                    regex = cfg.get("regex", regex); allowed = cfg.get("allowed", allowed)

            if st.button("🧪 Valider", type="primary"):
                df = r.df.copy(); series = df[target_col].astype(str)
                invalid_mask = pd.Series(False, index=df.index); explanations = []
                if rule_not_null:
                    m = series.str.len()==0; invalid_mask |= m; explanations.append(("Vide", m))
                def to_num(s): return pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
                if min_v.strip():
                    try: mv=float(min_v); nums=to_num(series); m=(nums<mv)|nums.isna(); invalid_mask|=m; explanations.append((f"< {mv}", m))
                    except Exception: st.warning("Min non numérique — ignoré.")
                if max_v.strip():
                    try: Mv=float(max_v); nums=to_num(series); m=(nums>Mv)|nums.isna(); invalid_mask|=m; explanations.append((f"> {Mv}", m))
                    except Exception: st.warning("Max non numérique — ignoré.")
                if regex.strip():
                    try: pat=re.compile(regex); m=~series.str.match(pat); invalid_mask|=m; explanations.append(("Regex !match", m))
                    except re.error: st.warning("Regex invalide — ignorée.")
                if allowed.strip():
                    allowed_set={a.strip() for a in allowed.split(",") if a.strip()}
                    if allowed_set: m=~series.isin(allowed_set); invalid_mask|=m; explanations.append(("Hors domaine", m))
                invalid_rows = df.loc[invalid_mask].copy()
                for label, mask in explanations: invalid_rows[f"__{label}"]=mask[invalid_mask].values
                st.write(f"**Lignes invalides : {len(invalid_rows):,}**".replace(",", " "))
                st.dataframe(invalid_rows.head(1000), use_container_width=True)
                out_sep = st.radio("Séparateur d’export", [",", ";", "\t", "|"], index=1, horizontal=True, key="val_sep")
                st.download_button("⬇️ Télécharger les lignes invalides", df_to_csv_bytes(invalid_rows, out_sep), "invalid_rows.csv", mime="text/csv")
                with st.expander("Enregistrer ce réglage comme preset"):
                    name = st.text_input("Nom du preset", key="val_save_name")
                    if st.button("💾 Sauvegarder preset"): save_preset(name, {"validation":{
                        "target":target_col,"notnull":rule_not_null,"min":min_v,"max":max_v,"regex":regex,"allowed":allowed}})
                log_run("validation", {"fichier": f.name, "invalid_rows": len(invalid_rows)})
    else:
        st.info("Charge un fichier pour configurer des règles.")

# --- Notes ---
st.divider()
st.markdown(textwrap.dedent("""
**Conseils**  
- Pour de gros fichiers, préférez CSV et limitez le nombre de colonnes.  
- Exports UTF-8 : au besoin, choisissez le séparateur correct à l’import Excel.  
"""))
