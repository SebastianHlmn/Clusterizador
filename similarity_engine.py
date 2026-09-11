from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


FAMILY_SPECS: Dict[str, Sequence[str]] = {
    "Volumen": (
        "casos", "actuaciones", "audiencias",
    ),
    "Actividad / avance": (
        "actuaciones_por_caso", "audiencias_por_caso", "tasa_casos_con_audiencia",
    ),
    "Trayectoria / salidas": (
        "tasa_formalizacion", "tasa_acusacion", "tasa_criterio_oportunidad",
        "tasa_archivo", "tasa_desestimacion", "tasa_incompetencia",
        "tasa_reparacion_integral", "tasa_spp", "tasa_conciliacion",
        "tasa_acuerdo_pleno", "tasa_sentencia_juicio_oral", "tasa_sobreseimiento_otros",
    ),
    "Conflictividad": (),
    "Complejidad": (
        "tasa_complejidad", "promedio_imputados_por_caso",
        "promedio_delitos_distintos_por_caso",
    ),
    "Litigación / RRHH": (
        "litigantes", "fiscales", "auxiliares_fiscales",
        "casos_por_litigante", "actuaciones_por_litigante",
        "audiencias_por_litigante", "formalizaciones_por_litigante",
        "acusaciones_por_litigante",
    ),
}

RAW_COUNT_FEATURES = {
    "casos", "actuaciones", "audiencias", "litigantes", "fiscales", "auxiliares_fiscales"
}

PROFILE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Integral": {
        "Volumen": 1.0,
        "Actividad / avance": 1.0,
        "Trayectoria / salidas": 1.0,
        "Conflictividad": 1.0,
        "Complejidad": 1.0,
        "Litigación / RRHH": 1.0,
    },
    "Atención inicial": {
        "Volumen": 0.6,
        "Actividad / avance": 1.0,
        "Trayectoria / salidas": 1.0,
        "Conflictividad": 0.5,
        "Complejidad": 0.5,
        "Litigación / RRHH": 1.0,
    },
    "Funcionamiento procesal": {
        "Volumen": 0.0,
        "Actividad / avance": 1.0,
        "Trayectoria / salidas": 1.0,
        "Conflictividad": 0.3,
        "Complejidad": 0.5,
        "Litigación / RRHH": 0.6,
    },
    "Demanda / conflictividad": {
        "Volumen": 0.4,
        "Actividad / avance": 0.0,
        "Trayectoria / salidas": 0.0,
        "Conflictividad": 1.0,
        "Complejidad": 1.0,
        "Litigación / RRHH": 0.0,
    },
    "Litigación": {
        "Volumen": 0.3,
        "Actividad / avance": 0.7,
        "Trayectoria / salidas": 0.5,
        "Conflictividad": 0.0,
        "Complejidad": 0.3,
        "Litigación / RRHH": 1.0,
    },
}


def available_families(df: pd.DataFrame) -> Dict[str, list[str]]:
    out: Dict[str, list[str]] = {}
    for family, cols in FAMILY_SPECS.items():
        if family == "Conflictividad":
            candidates = [c for c in df.columns if c.startswith("tasa_conflictividad_")]
        else:
            candidates = list(cols)
        valid = [
            c for c in candidates
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2
        ]
        if valid:
            out[family] = valid
    return out


def _robust_standardize(series: pd.Series, log_count: bool) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if log_count:
        x = np.log1p(x.clip(lower=0))
    med = x.median(skipna=True)
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    scale = q3 - q1
    if not np.isfinite(scale) or scale == 0:
        scale = x.std(skipna=True)
    if not np.isfinite(scale) or scale == 0:
        return pd.Series(np.where(x.notna(), 0.0, np.nan), index=x.index)
    return (x - med) / scale


def standardized_matrix(df: pd.DataFrame, families: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    z = pd.DataFrame(index=df.index)
    for cols in families.values():
        for col in cols:
            if col not in z:
                z[col] = _robust_standardize(df[col], col in RAW_COUNT_FEATURES)
    return z


def _family_distance(z: pd.DataFrame, i: int, j: int, cols: Sequence[str]) -> float:
    a = z.loc[i, list(cols)].astype(float)
    b = z.loc[j, list(cols)].astype(float)
    valid = a.notna() & b.notna()
    if valid.sum() == 0:
        return np.nan
    diff = a[valid] - b[valid]
    return float(np.sqrt(np.mean(np.square(diff))))


def rank_neighbors(
    df: pd.DataFrame,
    reference_unit: str,
    weights: Mapping[str, float],
    unit_col: str = "unidad",
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, list[str]]]:
    families = available_families(df)
    active = {
        fam: cols for fam, cols in families.items()
        if float(weights.get(fam, 0.0)) > 0 and cols
    }
    if not active:
        raise ValueError("No hay dimensiones activas con indicadores disponibles.")

    work = df.reset_index(drop=True).copy()
    matches = work.index[work[unit_col].astype(str) == str(reference_unit)].tolist()
    if not matches:
        raise ValueError(f"No se encontró la unidad de referencia: {reference_unit}")
    ref_idx = matches[0]
    z = standardized_matrix(work, active)

    rows = []
    for idx, row in work.iterrows():
        if idx == ref_idx:
            continue
        family_dist: Dict[str, float] = {}
        numerator = 0.0
        denominator = 0.0
        for family, cols in active.items():
            d = _family_distance(z, ref_idx, idx, cols)
            family_dist[family] = d
            w = float(weights.get(family, 0.0))
            if np.isfinite(d) and w > 0:
                numerator += w * (d ** 2)
                denominator += w
        global_d = np.sqrt(numerator / denominator) if denominator > 0 else np.nan
        out = {"unidad": row[unit_col], "distancia_global": global_d}
        out.update({f"dist_{fam}": d for fam, d in family_dist.items()})
        rows.append(out)

    ranking = pd.DataFrame(rows).dropna(subset=["distancia_global"])
    ranking = ranking.sort_values("distancia_global", ascending=True).reset_index(drop=True)
    ranking.insert(0, "puesto", np.arange(1, len(ranking) + 1))
    return ranking, z, active


def pair_detail(
    df: pd.DataFrame,
    z: pd.DataFrame,
    families: Mapping[str, Sequence[str]],
    unit_a: str,
    unit_b: str,
    unit_col: str = "unidad",
) -> pd.DataFrame:
    work = df.reset_index(drop=True).copy()
    ia = work.index[work[unit_col].astype(str) == str(unit_a)].tolist()[0]
    ib = work.index[work[unit_col].astype(str) == str(unit_b)].tolist()[0]
    rows = []
    for family, cols in families.items():
        for col in cols:
            za = z.at[ia, col] if col in z.columns else np.nan
            zb = z.at[ib, col] if col in z.columns else np.nan
            diff = abs(za - zb) if np.isfinite(za) and np.isfinite(zb) else np.nan
            rows.append({
                "dimension": family,
                "indicador": col,
                "valor_referencia": work.at[ia, col] if col in work.columns else np.nan,
                "valor_comparable": work.at[ib, col] if col in work.columns else np.nan,
                "diferencia_estandarizada": diff,
            })
    return pd.DataFrame(rows)


def humanize(name: str) -> str:
    labels = {
        "casos": "Casos",
        "actuaciones": "Actuaciones",
        "audiencias": "Audiencias",
        "actuaciones_por_caso": "Actuaciones por caso",
        "audiencias_por_caso": "Audiencias por caso",
        "tasa_casos_con_audiencia": "Casos con audiencia",
        "tasa_formalizacion": "Tasa de formalización",
        "tasa_acusacion": "Tasa de acusación",
        "tasa_criterio_oportunidad": "Criterio de oportunidad",
        "tasa_archivo": "Archivo",
        "tasa_desestimacion": "Desestimación",
        "tasa_incompetencia": "Incompetencia / derivación",
        "tasa_reparacion_integral": "Reparación integral",
        "tasa_spp": "Suspensión del proceso a prueba",
        "tasa_conciliacion": "Conciliación",
        "tasa_acuerdo_pleno": "Acuerdo pleno",
        "tasa_sentencia_juicio_oral": "Sentencia en juicio oral",
        "tasa_sobreseimiento_otros": "Sobreseimiento",
        "tasa_complejidad": "Casos complejos",
        "promedio_imputados_por_caso": "Imputados por caso",
        "promedio_delitos_distintos_por_caso": "Delitos distintos por caso",
        "litigantes": "Litigantes",
        "fiscales": "Fiscales",
        "auxiliares_fiscales": "Auxiliares fiscales",
        "casos_por_litigante": "Casos por litigante",
        "actuaciones_por_litigante": "Actuaciones por litigante",
        "audiencias_por_litigante": "Audiencias por litigante",
        "formalizaciones_por_litigante": "Formalizaciones por litigante",
        "acusaciones_por_litigante": "Acusaciones por litigante",
    }
    if name in labels:
        return labels[name]
    if name.startswith("tasa_conflictividad_"):
        return "Conflictividad " + name.rsplit("_", 1)[-1]
    return name.replace("_", " ").capitalize()


def build_summary(
    reference: str,
    comparable: str,
    ranking_row: pd.Series,
    detail: pd.DataFrame,
) -> str:
    fam_cols = [c for c in ranking_row.index if c.startswith("dist_")]
    family_pairs = [
        (c.removeprefix("dist_"), float(ranking_row[c]))
        for c in fam_cols if pd.notna(ranking_row[c])
    ]
    family_pairs.sort(key=lambda x: x[1])
    closest = [name for name, _ in family_pairs[:2]]
    furthest = family_pairs[-1][0] if family_pairs else None

    valid_detail = detail.dropna(subset=["diferencia_estandarizada"]).copy()
    biggest = valid_detail.sort_values("diferencia_estandarizada", ascending=False).head(3)
    indicators = ", ".join(humanize(v) for v in biggest["indicador"].tolist())

    text = (
        f"Dentro del universo seleccionado, {comparable} es la unidad más cercana a {reference} "
        f"según la configuración actual."
    )
    if closest:
        text += f" La mayor cercanía aparece en {' y '.join(closest)}."
    if furthest:
        text += f" La principal distancia entre ambas está en {furthest}."
    if indicators:
        text += f" Los indicadores que más las diferencian son {indicators}."
    return text
