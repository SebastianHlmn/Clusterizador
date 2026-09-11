from __future__ import annotations

from datetime import date
from typing import Sequence, Tuple

import duckdb
import pandas as pd
import streamlit as st

import clusterizador_unidades_fiscales as core


def _sql_literal(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _in_clause(expr: str, values: Sequence[str]) -> str:
    clean = [str(v) for v in values if str(v).strip()]
    if not clean:
        return ""
    return f" AND {expr} IN ({', '.join(_sql_literal(v) for v in clean)})"


def _accusatory_clause(axis: str, only_accusatory: bool) -> str:
    if not only_accusatory:
        return ""
    unit_col = core.qident(core.AXIS_COLS[axis]["unidad"])
    unit_norm = f"lower(trim(CAST({unit_col} AS VARCHAR)))"
    return f" AND ({unit_norm} LIKE 'fiscalía%' OR {unit_norm} LIKE 'fiscalia%')"


def _where_filters(
    *,
    axis: str,
    grain: str,
    start: date,
    end: date,
    analysis_units: Sequence[str],
    only_accusatory: bool,
) -> Tuple[str, str]:
    expr, _, _ = core.unit_expr(axis, grain)
    clause = (
        f"{core.date_clause(start, end)}"
        " AND IdCasoOriginal IS NOT NULL"
        f" AND coalesce({expr}, '') <> ''"
        + _accusatory_clause(axis, only_accusatory)
        + _in_clause(expr, analysis_units)
    )
    return expr, clause


@st.cache_data(show_spinner=False)
def hierarchy_options(
    current_path: str,
    axis: str,
    start_iso: str,
    end_iso: str,
    only_accusatory: bool,
) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    cols = core.AXIS_COLS[axis]
    district = core.qident(cols["distrito"])
    unit = core.qident(cols["unidad"])
    office = core.qident(cols["oficina"])
    accusatory = _accusatory_clause(axis, only_accusatory)
    q = f"""
        SELECT DISTINCT
            trim(CAST({district} AS VARCHAR)) AS distrito,
            trim(CAST({unit} AS VARCHAR)) AS unidad_fiscal,
            coalesce(nullif(trim(CAST({office} AS VARCHAR)), ''), 'Sin oficina') AS oficina
        FROM read_parquet('{current_path.replace("'", "''")}')
        WHERE {core.date_clause(start, end)}
          AND IdCasoOriginal IS NOT NULL
          AND coalesce(trim(CAST({district} AS VARCHAR)), '') <> ''
          AND coalesce(trim(CAST({unit} AS VARCHAR)), '') <> ''
          {accusatory}
        ORDER BY 1, 2, 3
    """
    c = duckdb.connect(database=":memory:")
    df = c.execute(q).df()
    c.close()
    if df.empty:
        return df
    df["oficina_etiqueta"] = df["unidad_fiscal"] + " · " + df["oficina"]
    return df


@st.cache_data(show_spinner=False)
def build_core_metrics(
    current_path: str,
    axis: str,
    grain: str,
    start_iso: str,
    end_iso: str,
    analysis_units: Tuple[str, ...],
    only_accusatory: bool,
) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    expr, clause = _where_filters(
        axis=axis,
        grain=grain,
        start=start,
        end=end,
        analysis_units=analysis_units,
        only_accusatory=only_accusatory,
    )
    q = f"""
    WITH f AS (
        SELECT *, {expr} AS unidad
        FROM read_parquet('{current_path.replace("'", "''")}')
        WHERE {clause}
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
    if not df.empty:
        df["actuaciones_por_caso"] = df["actuaciones"] / df["casos"].replace(0, pd.NA)
        df["audiencias_por_caso"] = df["audiencias"] / df["casos"].replace(0, pd.NA)
        df["tasa_casos_con_audiencia"] = df["casos_con_audiencia"] / df["casos"].replace(0, pd.NA)
    return df


@st.cache_data(show_spinner=False)
def build_hitos_metrics(
    hitos_path: str,
    axis: str,
    grain: str,
    start_iso: str,
    end_iso: str,
    analysis_units: Tuple[str, ...],
    only_accusatory: bool,
) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    expr, clause = _where_filters(
        axis=axis,
        grain=grain,
        start=start,
        end=end,
        analysis_units=analysis_units,
        only_accusatory=only_accusatory,
    )
    parts = [
        f"count(DISTINCT CASE WHEN ({cond}) THEN IdCasoOriginal END) AS casos_{key}"
        for key, cond in core.HITOS_SQL.items()
    ]
    q = f"""
        WITH f AS (
            SELECT *, {expr} AS unidad
            FROM read_parquet('{hitos_path.replace("'", "''")}')
            WHERE {clause}
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
    analysis_units: Tuple[str, ...],
    only_accusatory: bool,
) -> pd.DataFrame:
    start = date.fromisoformat(start_iso)
    end = date.fromisoformat(end_iso)
    expr, clause = _where_filters(
        axis=axis,
        grain=grain,
        start=start,
        end=end,
        analysis_units=analysis_units,
        only_accusatory=only_accusatory,
    )
    dim = core.load_conflict(conflict_path)
    c = duckdb.connect(database=":memory:")
    c.register("dim_conf", dim)
    q = f"""
        WITH f AS (
            SELECT {expr} AS unidad,
                   IdCasoOriginal,
                   try_cast(idtipodelito AS BIGINT) AS idtipodelito
            FROM read_parquet('{current_path.replace("'", "''")}')
            WHERE {clause}
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
    count_piv = long.pivot_table(
        index="unidad", columns="codigo", values="casos_conflictividad", aggfunc="sum", fill_value=0
    )
    count_piv.columns = [f"casos_conflictividad_{int(code)}" for code in count_piv.columns]
    return count_piv.reset_index()
