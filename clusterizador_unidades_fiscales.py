from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler


APP_VERSION = "0.1.0"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_CURRENT = DATA_DIR / "data_final_acusatorio20260904.parquet"
DEFAULT_HITOS = DATA_DIR / "baseUnisaAcusatorio.parquet"
DEFAULT_CONFLICT = DATA_DIR / "dim_delito_conflictividad_completa_v2.xlsx"
DEFAULT_RRHH = DATA_DIR / "Fiscales_y_AF_jurisdicciones_implementadas.xlsx"
DEFAULT_TERRITORIO = DATA_DIR / "Base_Ancha_Territorios_Fiscalias_Sedes_2026_superficie_oficial.xlsx"


# -----------------------------------------------------------------------------
# Reglas de medición
# -----------------------------------------------------------------------------
# Regla central: un caso se cuenta SIEMPRE como n_distinct(IdCasoOriginal)
# y una actuación SIEMPRE como n_distinct(IdActuacion). Nunca se cuentan filas.
#
# EstadoInformeConsistencia se usa como taxonomía de hitos/salidas depurada.
# Los hitos se expresan en cantidad bruta de casos y como tasa sobre los casos
# de la unidad en la ventana seleccionada.

HITOS_SQL: Dict[str, str] = {
    "formalizacion": "EstadoInformeConsistencia = 'Formalización'",
    "acusacion": "EstadoInformeConsistencia = 'Acusación Fiscal'",
    "archivo": "lower(EstadoInformeConsistencia) LIKE 'archivo%'",
    "desestimacion": "lower(EstadoInformeConsistencia) LIKE 'desestimación%'",
    "criterio_oportunidad": "lower(EstadoInformeConsistencia) LIKE 'criterio oportunidad%' OR lower(EstadoInformeConsistencia) LIKE 'criteriodeoportunidad%'",
    "incompetencia": "EstadoInformeConsistencia IN ('Incompetencia', 'Derivado_a_organismo_externo')",
    "reparacion_integral": "EstadoInformeConsistencia = 'Reparación Integral'",
    "spp": "EstadoInformeConsistencia = 'Suspensión de proceso a prueba'",
    "conciliacion": "EstadoInformeConsistencia = 'Conciliación'",
    "acuerdo_pleno": "EstadoInformeConsistencia IN ('SentenciaCondenatoriaAcuerdoPleno', 'SentenciaAbsolutoriaAcuerdoPleno')",
    "sentencia_juicio_oral": "EstadoInformeConsistencia IN ('SentenciaCondenatoriaJuicio', 'SentenciaAbsolutoriaJuicio')",
    "sobreseimiento_otros": "EstadoInformeConsistencia = 'Sobreseimiento'",
}

HITO_LABELS = {
    "formalizacion": "Formalización",
    "acusacion": "Acusación fiscal",
    "archivo": "Archivo",
    "desestimacion": "Desestimación",
    "criterio_oportunidad": "Criterio de oportunidad",
    "incompetencia": "Incompetencia / derivación",
    "reparacion_integral": "Reparación integral",
    "spp": "Suspensión del proceso a prueba",
    "conciliacion": "Conciliación",
    "acuerdo_pleno": "Acuerdo pleno",
    "sentencia_juicio_oral": "Sentencia en juicio oral",
    "sobreseimiento_otros": "Otros sobreseimientos",
}

AXIS_COLS = {
    "Actuación": {
        "distrito": "jurisdiccion_actuacion",
        "unidad": "unidadfiscal_actuacion",
        "oficina": "oficina_actuacion",
    },
    "Actual": {
        "distrito": "jurisdiccion_actual",
        "unidad": "unidadfiscal_actual",
        "oficina": "oficina_actual",
    },
    "Ingreso": {
        "distrito": "jurisdiccion_ingreso",
        "unidad": "unidadfiscal_ingreso",
        "oficina": "oficina_ingreso",
    },
}

GRAIN_LABELS = {
    "Distrito": "distrito",
    "Unidad / sede": "unidad",
    "Oficina / área": "oficina",
}


@dataclass(frozen=True)
class Paths:
    current: Path
    hitos: Path
    conflict: Path
    rrhh: Path
    territorio: Optional[Path]


def norm_text(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s.strip()


def sql_string(path: Path) -> str:
    return str(path).replace("'", "''")


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def unit_expr(axis: str, grain: str) -> Tuple[str, str, str]:
    cols = AXIS_COLS[axis]
    district = qident(cols["distrito"])
    unit = qident(cols["unidad"])
    office = qident(cols["oficina"])
    if grain == "distrito":
        expr = f"trim(CAST({district} AS VARCHAR))"
    elif grain == "unidad":
        expr = f"trim(CAST({unit} AS VARCHAR))"
    else:
        expr = (
            f"trim(CAST({unit} AS VARCHAR)) || ' · ' || "
            f"coalesce(nullif(trim(CAST({office} AS VARCHAR)), ''), 'Sin oficina')"
        )
    return expr, district, unit


@st.cache_resource(show_spinner=False)
def con() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


@st.cache_data(show_spinner=False)
def parquet_bounds(path_str: str) -> Tuple[Optional[date], Optional[date]]:
    c = duckdb.connect(database=":memory:")
    q = f"""
        SELECT min(CAST(fechaactuacion AS DATE)) AS dmin,
               max(CAST(fechaactuacion AS DATE)) AS dmax
        FROM read_parquet('{path_str.replace("'", "''")}')
        WHERE fechaactuacion IS NOT NULL
    """
    row = c.execute(q).fetchone()
    c.close()
    return row[0], row[1]


@st.cache_data(show_spinner=False)
def source_metadata(current_path: str, hitos_path: str) -> pd.DataFrame:
    rows = []
    for label, p in [("Base actual", current_path), ("Base de hitos", hitos_path)]:
        dmin, dmax = parquet_bounds(p)
        rows.append({"fuente": label, "fecha_min": dmin, "fecha_max": dmax})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_conflict(path_str: str) -> pd.DataFrame:
    df = pd.read_excel(path_str, sheet_name="dim_conflictividad_completa", engine="openpyxl")
    need = ["IdTipoDelito", "tipo_conflictividad_v2_codigo", "tipo_conflictividad_v2"]
    missing = [x for x in need if x not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en dimensión de conflictividad: {missing}")
    out = df[need].copy()
    out["IdTipoDelito"] = pd.to_numeric(out["IdTipoDelito"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["IdTipoDelito", "tipo_conflictividad_v2"]).drop_duplicates("IdTipoDelito")
    return out


@st.cache_data(show_spinner=False)
def load_rrhh(path_str: str) -> pd.DataFrame:
    df = pd.read_excel(path_str, sheet_name="Base", engine="openpyxl")
    required = ["Distrito", "Unidad/Sede", "Área", "Nombre", "Cargo", "Fecha de alta", "Fecha de baja"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en RRHH: {missing}")
    df = df[required].copy()
    df["Distrito"] = df["Distrito"].replace({"Mar del PLata": "Mar del Plata"})
    for c in ["Distrito", "Unidad/Sede", "Área", "Nombre", "Cargo"]:
        df[c] = df[c].astype("string").str.strip()
    df["Fecha de alta"] = pd.to_datetime(df["Fecha de alta"], errors="coerce").dt.date
    df["Fecha de baja"] = pd.to_datetime(df["Fecha de baja"], errors="coerce").dt.date
    cargo = df["Cargo"].fillna("").map(norm_text)
    df["es_auxiliar"] = cargo.str.contains(r"\baux\b|auxiliar", regex=True)
    df["es_fiscal"] = cargo.str.contains("fiscal", regex=False) & ~df["es_auxiliar"]
    df["es_litigante"] = df["es_auxiliar"] | df["es_fiscal"]
    return df


@st.cache_data(show_spinner=False)
def load_territorio(path_str: str) -> pd.DataFrame:
    df = pd.read_excel(path_str, sheet_name="Base consolidada", engine="openpyxl")
    needed = [
        "Distrito",
        "Territorio",
        "Fiscalía / sede (nombre original)",
        "Superficie territorio km²",
        "Población 2022 territorio",
        "Superficie distrito km²",
        "Población 2022 distrito",
    ]
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en base territorial: {missing}")
    return df[needed].copy()


def active_rrhh(df: pd.DataFrame, cut: date) -> pd.DataFrame:
    alta_ok = df["Fecha de alta"].isna() | (df["Fecha de alta"] <= cut)
    baja_ok = df["Fecha de baja"].isna() | (df["Fecha de baja"] >= cut)
    return df[alta_ok & baja_ok & df["es_litigante"]].copy()


def rrhh_by_grain(df: pd.DataFrame, grain: str, cut: date) -> pd.DataFrame:
    x = active_rrhh(df, cut)
    if grain == "distrito":
        x["unidad"] = x["Distrito"]
    elif grain == "unidad":
        x["unidad"] = x["Unidad/Sede"]
    else:
        x["unidad"] = x["Unidad/Sede"].fillna("") + " · " + x["Área"].fillna("Sin oficina")
    x["unidad_key"] = x["unidad"].map(norm_text)
    out = (
        x.groupby(["unidad_key", "unidad"], dropna=False)
        .agg(
            fiscales=("Nombre", lambda s: s[x.loc[s.index, "es_fiscal"]].nunique()),
            auxiliares_fiscales=("Nombre", lambda s: s[x.loc[s.index, "es_auxiliar"]].nunique()),
            litigantes=("Nombre", "nunique"),
        )
        .reset_index()
    )
    return out


def territorio_by_grain(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    x = df.copy()
    if grain == "distrito":
        out = (
            x.groupby("Distrito", dropna=False)
            .agg(
                poblacion_2022=("Población 2022 distrito", "max"),
                superficie_km2=("Superficie distrito km²", "max"),
            )
            .reset_index()
            .rename(columns={"Distrito": "unidad"})
        )
    else:
        out = (
            x.groupby("Fiscalía / sede (nombre original)", dropna=False)
            .agg(
                poblacion_2022=("Población 2022 territorio", "max"),
                superficie_km2=("Superficie territorio km²", "max"),
                territorio=("Territorio", lambda s: " | ".join(sorted(set(str(v) for v in s.dropna())))),
                distrito_territorial=("Distrito", lambda s: " | ".join(sorted(set(str(v) for v in s.dropna())))),
            )
            .reset_index()
            .rename(columns={"Fiscalía / sede (nombre original)": "unidad"})
        )
    out["unidad_key"] = out["unidad"].map(norm_text)
    out["densidad_hab_km2"] = out["poblacion_2022"] / out["superficie_km2"].replace(0, np.nan)
    return out


def date_clause(start: date, end: date, field: str = "fechaactuacion") -> str:
    return f"CAST({qident(field)} AS DATE) BETWEEN DATE '{start.isoformat()}' AND DATE '{end.isoformat()}'"


@st.cache_data(show_spinner=False)
def build_core_metrics(
    current_path: str,
    axis: str,
    grain: str,
    start_iso: str,
    end_iso: str,
) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    expr, district_col, unit_col = unit_expr(axis, grain)
    q = f"""
    WITH f AS (
        SELECT *, {expr} AS unidad
        FROM read_parquet('{current_path.replace("'", "''")}')
        WHERE {date_clause(start, end)}
          AND IdCasoOriginal IS NOT NULL
          AND coalesce({expr}, '') <> ''
    ),
    caso AS (
        SELECT
            unidad,
            IdCasoOriginal,
            max(CASE WHEN CasoComplejo = 'Caso Complejo' THEN 1 ELSE 0 END) AS complejo,
            max(try_cast(imputados AS DOUBLE)) AS imputados_caso,
            count(DISTINCT idtipodelito) AS delitos_distintos_caso
        FROM f
        GROUP BY 1,2
    ),
    caso_agg AS (
        SELECT
            unidad,
            avg(imputados_caso) AS promedio_imputados_por_caso,
            median(imputados_caso) AS mediana_imputados_por_caso,
            avg(delitos_distintos_caso) AS promedio_delitos_distintos_por_caso,
            sum(complejo) AS casos_complejos,
            avg(complejo) AS tasa_complejidad
        FROM caso
        GROUP BY 1
    ),
    principal AS (
        SELECT
            unidad,
            count(DISTINCT IdCasoOriginal) AS casos,
            count(DISTINCT IdActuacion) AS actuaciones,
            count(DISTINCT CASE WHEN ActuacionAudiencia = 'Audiencia' THEN IdActuacion END) AS audiencias,
            count(DISTINCT CASE WHEN ActuacionAudiencia = 'Audiencia' THEN IdCasoOriginal END) AS casos_con_audiencia,
            count(DISTINCT CASE WHEN formalizado = 'Formalizado' THEN IdCasoOriginal END) AS casos_formalizados_flag,
            max(try_cast(poblacion AS DOUBLE)) AS poblacion_coiron_max
        FROM f
        GROUP BY 1
    )
    SELECT p.*, c.promedio_imputados_por_caso, c.mediana_imputados_por_caso,
           c.promedio_delitos_distintos_por_caso, c.casos_complejos, c.tasa_complejidad
    FROM principal p
    LEFT JOIN caso_agg c USING (unidad)
    ORDER BY casos DESC
    """
    c = duckdb.connect(database=":memory:")
    df = c.execute(q).df()
    c.close()
    for col in ["casos", "actuaciones", "audiencias", "casos_con_audiencia", "casos_formalizados_flag"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["actuaciones_por_caso"] = df["actuaciones"] / df["casos"].replace(0, np.nan)
    df["audiencias_por_caso"] = df["audiencias"] / df["casos"].replace(0, np.nan)
    df["tasa_casos_con_audiencia"] = df["casos_con_audiencia"] / df["casos"].replace(0, np.nan)
    return df


@st.cache_data(show_spinner=False)
def build_hitos_metrics(
    hitos_path: str,
    axis: str,
    grain: str,
    start_iso: str,
    end_iso: str,
) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    expr, _, _ = unit_expr(axis, grain)
    parts = []
    for key, cond in HITOS_SQL.items():
        parts.append(
            f"count(DISTINCT CASE WHEN ({cond}) THEN IdCasoOriginal END) AS casos_{key}"
        )
    q = f"""
        WITH f AS (
            SELECT *, {expr} AS unidad
            FROM read_parquet('{hitos_path.replace("'", "''")}')
            WHERE {date_clause(start, end)}
              AND IdCasoOriginal IS NOT NULL
              AND coalesce({expr}, '') <> ''
        )
        SELECT unidad, {', '.join(parts)}
        FROM f
        GROUP BY 1
    """
    c = duckdb.connect(database=":memory:")
    df = c.execute(q).df()
    c.close()
    return df


@st.cache_data(show_spinner=False)
def build_conflict_metrics(
    current_path: str,
    conflict_path: str,
    axis: str,
    grain: str,
    start_iso: str,
    end_iso: str,
) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    expr, _, _ = unit_expr(axis, grain)
    dim = load_conflict(conflict_path)
    c = duckdb.connect(database=":memory:")
    c.register("dim_conf", dim)
    q = f"""
        WITH f AS (
            SELECT {expr} AS unidad,
                   IdCasoOriginal,
                   try_cast(idtipodelito AS BIGINT) AS idtipodelito
            FROM read_parquet('{current_path.replace("'", "''")}')
            WHERE {date_clause(start, end)}
              AND IdCasoOriginal IS NOT NULL
              AND coalesce({expr}, '') <> ''
        )
        SELECT f.unidad,
               d.tipo_conflictividad_v2_codigo AS codigo,
               d.tipo_conflictividad_v2 AS conflictividad,
               count(DISTINCT f.IdCasoOriginal) AS casos_conflictividad
        FROM f
        JOIN dim_conf d ON f.idtipodelito = d.IdTipoDelito
        GROUP BY 1,2,3
    """
    long = c.execute(q).df()
    c.close()
    if long.empty:
        return pd.DataFrame(columns=["unidad"])
    long["codigo"] = pd.to_numeric(long["codigo"], errors="coerce").astype("Int64")
    # Las categorías no son necesariamente excluyentes a nivel caso. Por eso se
    # usan prevalencias por categoría, no porcentajes que deban sumar 100.
    count_piv = long.pivot_table(index="unidad", columns="codigo", values="casos_conflictividad", aggfunc="sum", fill_value=0)
    count_piv.columns = [f"casos_conflictividad_{int(c)}" for c in count_piv.columns]
    count_piv = count_piv.reset_index()
    labels = (
        long[["codigo", "conflictividad"]]
        .drop_duplicates()
        .sort_values("codigo")
        .assign(codigo=lambda d: d["codigo"].astype(int))
    )
    count_piv.attrs["conflict_labels"] = dict(zip(labels["codigo"], labels["conflictividad"]))
    return count_piv


def merge_matrix(
    core: pd.DataFrame,
    hitos: pd.DataFrame,
    conflict: pd.DataFrame,
    rrhh: Optional[pd.DataFrame],
    territorio: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = core.merge(hitos, on="unidad", how="left").merge(conflict, on="unidad", how="left")
    out["unidad_key"] = out["unidad"].map(norm_text)

    if rrhh is not None and not rrhh.empty:
        rr = rrhh.drop(columns=["unidad"], errors="ignore").copy()
        out = out.merge(rr, on="unidad_key", how="left")
    else:
        out["fiscales"] = np.nan
        out["auxiliares_fiscales"] = np.nan
        out["litigantes"] = np.nan

    if territorio is not None and not territorio.empty:
        tt = territorio.drop(columns=["unidad"], errors="ignore").copy()
        out = out.merge(tt, on="unidad_key", how="left")

    raw_count_cols = [c for c in out.columns if c.startswith("casos_")]
    for col in raw_count_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        if col not in {"casos_complejos", "casos_con_audiencia", "casos_formalizados_flag"}:
            rate_name = "tasa_" + col.removeprefix("casos_")
            out[rate_name] = out[col] / out["casos"].replace(0, np.nan)

    # Prevalencias de conflictividad: casos con al menos un delito de la categoría / casos.
    for col in [c for c in out.columns if c.startswith("casos_conflictividad_")]:
        code = col.rsplit("_", 1)[-1]
        out[f"tasa_conflictividad_{code}"] = out[col] / out["casos"].replace(0, np.nan)

    if "litigantes" in out.columns:
        den = pd.to_numeric(out["litigantes"], errors="coerce").replace(0, np.nan)
        out["casos_por_litigante"] = out["casos"] / den
        out["actuaciones_por_litigante"] = out["actuaciones"] / den
        out["audiencias_por_litigante"] = out["audiencias"] / den
        if "casos_formalizacion" in out:
            out["formalizaciones_por_litigante"] = out["casos_formalizacion"] / den
        if "casos_acusacion" in out:
            out["acusaciones_por_litigante"] = out["casos_acusacion"] / den

    return out


def feature_families(df: pd.DataFrame) -> Dict[str, List[str]]:
    volume = [
        "casos", "actuaciones", "audiencias", "casos_con_audiencia",
    ] + [c for c in df.columns if c.startswith("casos_") and not c.startswith("casos_conflictividad_")]
    profile = [
        "actuaciones_por_caso", "audiencias_por_caso", "tasa_casos_con_audiencia",
        "tasa_complejidad", "promedio_imputados_por_caso", "promedio_delitos_distintos_por_caso",
    ] + [c for c in df.columns if c.startswith("tasa_") and not c.startswith("tasa_conflictividad_")]
    conflict = [c for c in df.columns if c.startswith("tasa_conflictividad_")]
    rrhh = [
        c for c in [
            "litigantes", "fiscales", "auxiliares_fiscales", "casos_por_litigante",
            "actuaciones_por_litigante", "audiencias_por_litigante",
            "formalizaciones_por_litigante", "acusaciones_por_litigante",
        ] if c in df.columns
    ]
    territory = [c for c in ["poblacion_2022", "superficie_km2", "densidad_hab_km2"] if c in df.columns]
    return {
        "Volumen bruto": sorted(set(volume)),
        "Perfil procesal / tasas": sorted(set(profile)),
        "Conflictividad": sorted(set(conflict)),
        "Litigación / RRHH": sorted(set(rrhh)),
        "Territorio (descriptivo)": sorted(set(territory)),
    }


def prepare_X(
    df: pd.DataFrame,
    features: Sequence[str],
    log_counts: bool,
    scaler_name: str,
) -> Tuple[pd.DataFrame, np.ndarray, object]:
    x = df[list(features)].copy()
    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")
        x[c] = x[c].replace([np.inf, -np.inf], np.nan)
        x[c] = x[c].fillna(x[c].median())
        if x[c].isna().all():
            x[c] = 0.0
    if log_counts:
        for c in x.columns:
            if c in {"casos", "actuaciones", "audiencias", "litigantes", "fiscales", "auxiliares_fiscales"} or c.startswith("casos_"):
                vals = x[c].clip(lower=0)
                x[c] = np.log1p(vals)
    scaler = StandardScaler() if scaler_name == "Z-score" else RobustScaler()
    X = scaler.fit_transform(x)
    return x, X, scaler


def best_k(X: np.ndarray, method: str, max_k: int = 8) -> Tuple[int, pd.DataFrame]:
    n = X.shape[0]
    max_k = min(max_k, n - 1)
    rows = []
    if max_k < 2:
        return 1, pd.DataFrame()
    for k in range(2, max_k + 1):
        if method == "kmeans":
            labels = KMeans(n_clusters=k, n_init=50, random_state=42).fit_predict(X)
        else:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except Exception:
            score = np.nan
        rows.append({"k": k, "silhouette": score})
    scores = pd.DataFrame(rows)
    valid = scores.dropna(subset=["silhouette"])
    return (int(valid.loc[valid["silhouette"].idxmax(), "k"]) if not valid.empty else 2), scores


def run_cluster(X: np.ndarray, algo: str, k: Optional[int], eps: float, min_samples: int) -> np.ndarray:
    if algo == "K-means":
        return KMeans(n_clusters=int(k), n_init=50, random_state=42).fit_predict(X)
    if algo == "Jerárquico Ward":
        return AgglomerativeClustering(n_clusters=int(k), linkage="ward").fit_predict(X)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)


def feature_stability(X_df: pd.DataFrame, ref_labels: np.ndarray, k: int, n_iter: int = 30) -> Optional[float]:
    if X_df.shape[1] < 4 or k < 2 or len(set(ref_labels)) < 2:
        return None
    rng = np.random.default_rng(20260904)
    scores = []
    cols = np.array(X_df.columns)
    for i in range(n_iter):
        n_pick = max(2, int(math.ceil(len(cols) * 0.8)))
        chosen = rng.choice(cols, size=n_pick, replace=False)
        Z = StandardScaler().fit_transform(X_df[list(chosen)])
        lab = KMeans(n_clusters=k, n_init=20, random_state=100 + i).fit_predict(Z)
        scores.append(adjusted_rand_score(ref_labels, lab))
    return float(np.mean(scores))


def display_name(feature: str, conflict_labels: Dict[int, str]) -> str:
    if feature.startswith("tasa_conflictividad_") or feature.startswith("casos_conflictividad_"):
        code = int(feature.rsplit("_", 1)[-1])
        pref = "Tasa" if feature.startswith("tasa_") else "Casos"
        return f"{pref}: {conflict_labels.get(code, f'Conflictividad {code}')}"
    replacements = {
        "casos": "Casos",
        "actuaciones": "Actuaciones",
        "audiencias": "Audiencias",
        "actuaciones_por_caso": "Actuaciones por caso",
        "audiencias_por_caso": "Audiencias por caso",
        "tasa_casos_con_audiencia": "Tasa de casos con audiencia",
        "tasa_complejidad": "Tasa de casos complejos",
        "promedio_imputados_por_caso": "Promedio de imputados por caso",
        "promedio_delitos_distintos_por_caso": "Promedio de delitos distintos por caso",
        "litigantes": "Litigantes",
        "casos_por_litigante": "Casos por litigante",
        "actuaciones_por_litigante": "Actuaciones por litigante",
        "audiencias_por_litigante": "Audiencias por litigante",
    }
    if feature in replacements:
        return replacements[feature]
    if feature.startswith("casos_"):
        key = feature.removeprefix("casos_")
        return "Casos con " + HITO_LABELS.get(key, key.replace("_", " "))
    if feature.startswith("tasa_"):
        key = feature.removeprefix("tasa_")
        return "Tasa de " + HITO_LABELS.get(key, key.replace("_", " "))
    return feature.replace("_", " ").capitalize()


def main() -> None:
    st.set_page_config(page_title="Clusterizador de unidades fiscales", layout="wide")
    st.title("Clusterizador de unidades fiscales")
    st.caption(
        "Comparación de distritos, unidades/sedes y oficinas mediante volumen, perfil procesal, "
        "conflictividad, complejidad y carga de litigación."
    )

    with st.sidebar:
        st.header("Fuentes")
        current_path = Path(st.text_input("Base actual (parquet)", str(DEFAULT_CURRENT)))
        hitos_path = Path(st.text_input("Base de hitos (parquet)", str(DEFAULT_HITOS)))
        conflict_path = Path(st.text_input("Dimensión conflictividad", str(DEFAULT_CONFLICT)))
        rrhh_path = Path(st.text_input("Fiscales y auxiliares", str(DEFAULT_RRHH)))
        territory_path_txt = st.text_input("Base territorial (opcional)", str(DEFAULT_TERRITORIO))
        territory_path = Path(territory_path_txt) if territory_path_txt.strip() else None

    required = [current_path, hitos_path, conflict_path, rrhh_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("Faltan archivos requeridos:\n\n" + "\n".join(f"- {x}" for x in missing))
        st.info("Copiá los archivos a la carpeta data/ o corregí las rutas en la barra lateral.")
        st.stop()

    try:
        meta = source_metadata(str(current_path), str(hitos_path))
    except Exception as e:
        st.exception(e)
        st.stop()

    dmaxs = [d for d in meta["fecha_max"].tolist() if pd.notna(d)]
    dmins = [d for d in meta["fecha_min"].tolist() if pd.notna(d)]
    overlap_max = min(dmaxs) if dmaxs else date.today()
    all_min = min(dmins) if dmins else overlap_max - timedelta(days=365)

    with st.expander("Cortes de las fuentes", expanded=False):
        st.dataframe(meta, hide_index=True, use_container_width=True)
        if len(set(dmaxs)) > 1:
            st.warning(
                f"Las fuentes no llegan al mismo día. Para evitar mezclar cortes, el máximo común es {overlap_max:%d/%m/%Y}."
            )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        grain_label = st.selectbox("Jerarquía a comparar", list(GRAIN_LABELS), index=1)
        grain = GRAIN_LABELS[grain_label]
    with col_b:
        axis = st.selectbox("Asignar por", list(AXIS_COLS), index=0, help="Actuación es el eje recomendado para carga y litigación.")
    with col_c:
        default_start = max(all_min, overlap_max - timedelta(days=364))
        start = st.date_input("Desde", value=default_start, max_value=overlap_max)
    with col_d:
        end = st.date_input("Hasta", value=overlap_max, min_value=start, max_value=overlap_max)

    if start > end:
        st.error("La fecha inicial no puede ser posterior a la final.")
        st.stop()

    with st.spinner("Construyendo matriz de unidades..."):
        try:
            core = build_core_metrics(str(current_path), axis, grain, start.isoformat(), end.isoformat())
            hitos = build_hitos_metrics(str(hitos_path), axis, grain, start.isoformat(), end.isoformat())
            conflict = build_conflict_metrics(str(current_path), str(conflict_path), axis, grain, start.isoformat(), end.isoformat())
            rrhh_raw = load_rrhh(str(rrhh_path))
            rrhh = rrhh_by_grain(rrhh_raw, grain, end)
            territorio = None
            if territory_path is not None and territory_path.exists():
                territorio = territorio_by_grain(load_territorio(str(territory_path)), grain)
            matrix = merge_matrix(core, hitos, conflict, rrhh, territorio)
        except Exception as e:
            st.exception(e)
            st.stop()

    if matrix.empty:
        st.warning("No hay unidades para la combinación seleccionada.")
        st.stop()

    conflict_labels = {}
    try:
        dimc = load_conflict(str(conflict_path))
        conflict_labels = dict(
            dimc[["tipo_conflictividad_v2_codigo", "tipo_conflictividad_v2"]]
            .drop_duplicates()
            .dropna()
            .astype({"tipo_conflictividad_v2_codigo": int})
            .values
        )
    except Exception:
        pass

    st.subheader("Matriz analítica")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unidades", len(matrix))
    c2.metric("Casos", f"{int(matrix['casos'].sum()):,}".replace(",", "."))
    c3.metric("Actuaciones", f"{int(matrix['actuaciones'].sum()):,}".replace(",", "."))
    c4.metric("Audiencias", f"{int(matrix['audiencias'].sum()):,}".replace(",", "."))

    rrhh_match = matrix["litigantes"].notna().mean() if "litigantes" in matrix else 0
    if rrhh_match < 0.8:
        st.warning(
            f"Cobertura del empalme de RRHH: {rrhh_match:.0%}. En esta jerarquía conviene revisar equivalencias de nombres antes de interpretar indicadores por litigante."
        )

    families = feature_families(matrix)
    family_choice = st.multiselect(
        "Familias que entran al clustering",
        options=list(families),
        default=["Volumen bruto", "Perfil procesal / tasas", "Conflictividad", "Litigación / RRHH"],
    )
    default_features = []
    for fam in family_choice:
        default_features.extend(families[fam])
    default_features = [f for f in dict.fromkeys(default_features) if f in matrix.columns]

    # Sacamos variables redundantes/auxiliares del default si ya está su equivalente más interpretable.
    default_features = [
        f for f in default_features
        if f not in {"casos_formalizados_flag", "poblacion_coiron_max", "casos_complejos"}
    ]

    feature_options = [
        c for c in matrix.columns
        if c not in {"unidad", "unidad_key", "territorio", "distrito_territorial"}
        and pd.api.types.is_numeric_dtype(matrix[c])
    ]
    labels_map = {c: display_name(c, conflict_labels) for c in feature_options}
    inv_labels = {v: k for k, v in labels_map.items()}
    selected_labels = st.multiselect(
        "Indicadores",
        options=[labels_map[c] for c in feature_options],
        default=[labels_map[c] for c in default_features if c in labels_map],
    )
    features = [inv_labels[x] for x in selected_labels if x in inv_labels]

    if len(features) < 2:
        st.info("Seleccioná al menos dos indicadores para calcular clusters.")
        st.stop()

    with st.expander("Definiciones de medición", expanded=False):
        st.markdown(
            """
- **Casos:** `IdCasoOriginal` distintos.  
- **Actuaciones:** `IdActuacion` distintas.  
- **Audiencias:** `IdActuacion` distintas con `ActuacionAudiencia = 'Audiencia'`.  
- **Hitos/salidas:** casos distintos con cada categoría de `EstadoInformeConsistencia`.  
- **Tasa de un hito:** casos con ese hito / casos de la unidad en la ventana seleccionada.  
- **Conflictividad:** prevalencia de casos con al menos un delito de cada categoría; las categorías pueden superponerse a nivel caso y por eso no se obliga a que sumen 100%.  
- **Litigantes:** personas activas al corte cuyo cargo es fiscal o auxiliar fiscal.
            """
        )

    a, b, ccol, d = st.columns(4)
    with a:
        scaler_name = st.selectbox("Escalado", ["Z-score", "Robusto"], index=0)
    with b:
        log_counts = st.checkbox("Logaritmo en cantidades brutas", value=True, help="Conserva el tamaño pero reduce el dominio de unidades extremadamente grandes.")
    with ccol:
        algo = st.selectbox("Algoritmo", ["Emergente automático", "Jerárquico Ward", "K-means", "DBSCAN"], index=0)
    with d:
        min_cases = st.number_input("Mínimo de casos por unidad", min_value=1, value=30, step=10)

    work = matrix[matrix["casos"] >= min_cases].copy().reset_index(drop=True)
    if len(work) < 3:
        st.warning("Quedaron menos de tres unidades después del filtro mínimo.")
        st.stop()

    raw_x, X, scaler = prepare_X(work, features, log_counts=log_counts, scaler_name=scaler_name)

    scores = pd.DataFrame()
    chosen_k = None
    if algo == "Emergente automático":
        chosen_k, scores = best_k(X, "kmeans")
        cluster_labels = run_cluster(X, "K-means", chosen_k, 0.8, 3)
        algo_used = f"K-means automático (k={chosen_k})"
    elif algo == "Jerárquico Ward":
        auto_k, scores = best_k(X, "ward")
        chosen_k = st.slider("Cantidad de clusters", 2, min(10, len(work) - 1), auto_k)
        cluster_labels = run_cluster(X, algo, chosen_k, 0.8, 3)
        algo_used = f"Ward (k={chosen_k})"
    elif algo == "K-means":
        auto_k, scores = best_k(X, "kmeans")
        chosen_k = st.slider("Cantidad de clusters", 2, min(10, len(work) - 1), auto_k)
        cluster_labels = run_cluster(X, algo, chosen_k, 0.8, 3)
        algo_used = f"K-means (k={chosen_k})"
    else:
        e1, e2 = st.columns(2)
        with e1:
            eps = st.slider("DBSCAN eps", 0.2, 3.0, 0.9, 0.05)
        with e2:
            min_samples = st.slider("DBSCAN min_samples", 2, min(10, len(work)), 3)
        cluster_labels = run_cluster(X, algo, None, eps, min_samples)
        algo_used = f"DBSCAN (eps={eps:.2f}, min={min_samples})"

    work["cluster"] = cluster_labels
    work["cluster_etiqueta"] = work["cluster"].map(lambda x: "Atípica / ruido" if x == -1 else f"Cluster {x + 1}")

    # PCA: visualización, no define por sí misma los clusters.
    ncomp = min(2, X.shape[0], X.shape[1])
    pca = PCA(n_components=ncomp, random_state=42)
    coords = pca.fit_transform(X)
    work["PC1"] = coords[:, 0]
    work["PC2"] = coords[:, 1] if ncomp > 1 else 0.0

    st.subheader("Clusters emergentes")
    m1, m2, m3 = st.columns(3)
    m1.metric("Solución", algo_used)
    n_clusters = len(set(cluster_labels) - {-1})
    m2.metric("Clusters", n_clusters)
    if len(set(cluster_labels)) > 1 and not (set(cluster_labels) == {-1}):
        try:
            sil = silhouette_score(X, cluster_labels) if -1 not in cluster_labels else np.nan
        except Exception:
            sil = np.nan
        m3.metric("Silhouette", "—" if np.isnan(sil) else f"{sil:.3f}")
    else:
        m3.metric("Silhouette", "—")

    if chosen_k is not None and algo != "DBSCAN":
        stab = feature_stability(raw_x, cluster_labels, chosen_k, n_iter=30)
        if stab is not None:
            st.caption(f"Estabilidad ante perturbación del 20% de los indicadores (ARI medio, 30 corridas): **{stab:.3f}**")

    fig = px.scatter(
        work,
        x="PC1",
        y="PC2",
        color="cluster_etiqueta",
        hover_name="unidad",
        hover_data={"casos": ":,.0f", "actuaciones": ":,.0f", "audiencias": ":,.0f", "PC1": ":.2f", "PC2": ":.2f"},
        title="Proyección PCA de las unidades",
    )
    fig.update_traces(marker={"size": 11})
    st.plotly_chart(fig, use_container_width=True, key="pca_clusters")

    if not scores.empty:
        with st.expander("Elección del número de clusters", expanded=False):
            fig_k = px.line(scores, x="k", y="silhouette", markers=True, title="Silhouette por cantidad de clusters")
            st.plotly_chart(fig_k, use_container_width=True, key="silhouette_k")

    # Perfil de clusters en la escala original.
    profile_cols = [f for f in features if f in work.columns]
    prof = work.groupby("cluster_etiqueta", dropna=False)[profile_cols].mean(numeric_only=True)
    prof.insert(0, "n_unidades", work.groupby("cluster_etiqueta").size())
    prof = prof.reset_index()
    prof = prof.rename(columns={c: labels_map.get(c, c) for c in prof.columns})
    st.markdown("**Perfil promedio de cada cluster**")
    st.dataframe(prof, hide_index=True, use_container_width=True)

    # Comparador de vecinos: distancia en el mismo espacio utilizado para clusterizar.
    st.subheader("Comparar una unidad")
    selected_unit = st.selectbox("Unidad", work["unidad"].sort_values().tolist())
    idx = int(work.index[work["unidad"] == selected_unit][0])
    dist = np.sqrt(((X - X[idx]) ** 2).sum(axis=1))
    neigh = work[["unidad", "cluster_etiqueta", "casos", "actuaciones", "audiencias"]].copy()
    neigh["distancia_estandarizada"] = dist
    neigh = neigh[neigh["unidad"] != selected_unit].sort_values("distancia_estandarizada").head(8)
    st.dataframe(neigh, hide_index=True, use_container_width=True)

    with st.expander("Tabla completa", expanded=False):
        show_cols = ["unidad", "cluster_etiqueta"] + [c for c in features if c in work.columns]
        st.dataframe(
            work[show_cols].rename(columns={c: labels_map.get(c, c) for c in show_cols}),
            hide_index=True,
            use_container_width=True,
        )

    export_cols = [c for c in work.columns if c not in {"unidad_key"}]
    csv = work[export_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar matriz y clusters (CSV)",
        data=csv,
        file_name=f"clusters_unidades_{grain}_{start}_{end}.csv",
        mime="text/csv",
    )

    st.caption(f"Clusterizador UNISA · versión {APP_VERSION}")


if __name__ == "__main__":
    main()
