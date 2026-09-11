from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import clusterizador_unidades_fiscales as core
import metrics_v4 as metrics


APP_VERSION = "0.4.0"
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configuraciones"
CONFIG_FILE = CONFIG_DIR / "clusterizador_configuraciones.json"


# -----------------------------------------------------------------------------
# Configuraciones guardadas
# -----------------------------------------------------------------------------
def load_saved_configs() -> Dict[str, dict]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_saved_configs(configs: Dict[str, dict]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_FILE)


def load_selected_config_callback() -> None:
    name = st.session_state.get("saved_config_selector", "—")
    configs = load_saved_configs()
    if name in configs:
        st.session_state["_pending_cluster_config"] = configs[name]


def delete_selected_config_callback() -> None:
    name = st.session_state.get("saved_config_selector", "—")
    configs = load_saved_configs()
    if name in configs:
        configs.pop(name, None)
        write_saved_configs(configs)
    st.session_state["saved_config_selector"] = "—"


# -----------------------------------------------------------------------------
# Estado de los filtros jerárquicos
# -----------------------------------------------------------------------------
def _drop_state(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def reset_full_hierarchy_callback() -> None:
    _drop_state("cfg_districts", "cfg_fiscal_units", "cfg_offices")


def district_changed_callback() -> None:
    # Cascada: si cambia el padre, se reconstruyen todos los hijos.
    _drop_state("cfg_fiscal_units", "cfg_offices")


def fiscal_unit_changed_callback() -> None:
    # Cascada: si cambia la unidad/sede, se reconstruyen sus oficinas.
    _drop_state("cfg_offices")


def set_list_and_reset_callback(
    key: str,
    values: Sequence[str],
    reset_children: Sequence[str] = (),
) -> None:
    st.session_state[key] = list(values)
    for child in reset_children:
        st.session_state.pop(child, None)


def ensure_multiselect_state(
    key: str,
    options: Sequence[str],
    preferred: Optional[Sequence[str]] = None,
) -> None:
    opts = list(options)
    if preferred is not None:
        st.session_state[key] = [value for value in preferred if value in opts]
        return
    if key not in st.session_state:
        # Cuando un padre cambia, el hijo se elimina del estado y acá vuelve a
        # inicializarse con todas las opciones válidas del nuevo padre.
        st.session_state[key] = opts.copy()
        return
    st.session_state[key] = [value for value in st.session_state[key] if value in opts]


def selection_row(
    label: str,
    key: str,
    options: Sequence[str],
    *,
    help_text: Optional[str] = None,
    on_change=None,
    reset_children: Sequence[str] = (),
) -> list[str]:
    a, b, c = st.columns([6, 1, 1])
    with a:
        selected = st.multiselect(
            label,
            options=list(options),
            key=key,
            help=help_text,
            on_change=on_change,
        )
    with b:
        st.write("")
        st.write("")
        st.button(
            "Todas",
            key=f"all_{key}",
            use_container_width=True,
            on_click=set_list_and_reset_callback,
            args=(key, list(options), tuple(reset_children)),
        )
    with c:
        st.write("")
        st.write("")
        st.button(
            "Ninguna",
            key=f"none_{key}",
            use_container_width=True,
            on_click=set_list_and_reset_callback,
            args=(key, [], tuple(reset_children)),
        )
    return selected


def safe_date(value: object, fallback: date, lower: date, upper: date) -> date:
    try:
        parsed = date.fromisoformat(str(value))
    except Exception:
        parsed = fallback
    return min(max(parsed, lower), upper)


def build_configuration(
    *,
    grain_label: str,
    axis: str,
    start: date,
    end: date,
    only_accusatory: bool,
    only_fiscal_units: bool,
    selected_districts: Sequence[str],
    selected_fiscal_units: Sequence[str],
    selected_offices: Sequence[str],
    analysis_units: Sequence[str],
    family_choice: Sequence[str],
    features: Sequence[str],
    scaler_name: str,
    log_counts: bool,
    algo: str,
    min_cases: int,
    chosen_k: Optional[int],
    eps: Optional[float],
    min_samples: Optional[int],
) -> dict:
    return {
        "version": APP_VERSION,
        "jerarquia": grain_label,
        "asignar_por": axis,
        "desde": start.isoformat(),
        "hasta": end.isoformat(),
        "solo_acusatorio": bool(only_accusatory),
        "solo_unidades_fiscales": bool(only_fiscal_units),
        "distritos": list(selected_districts),
        "unidades_fiscales": list(selected_fiscal_units),
        "oficinas": list(selected_offices),
        "unidades": list(analysis_units),
        "familias": list(family_choice),
        "indicadores": list(features),
        "escalado": scaler_name,
        "log_cantidades": bool(log_counts),
        "algoritmo": algo,
        "minimo_casos": int(min_cases),
        "k": int(chosen_k) if chosen_k is not None else None,
        "dbscan_eps": float(eps) if eps is not None else None,
        "dbscan_min_samples": int(min_samples) if min_samples is not None else None,
    }


def main() -> None:
    st.set_page_config(page_title="Clusterizador de unidades fiscales", layout="wide")
    st.title("Clusterizador de unidades fiscales")
    st.caption(
        "Comparación de distritos, unidades/sedes y oficinas mediante volumen, perfil procesal, "
        "conflictividad, complejidad y carga de litigación."
    )

    saved_configs = load_saved_configs()
    pending = st.session_state.pop("_pending_cluster_config", None)

    with st.sidebar:
        st.header("Configuraciones")
        config_names = sorted(saved_configs)
        selected_config_name = st.selectbox(
            "Configuración guardada", options=["—"] + config_names, key="saved_config_selector"
        )
        ca, cb = st.columns(2)
        with ca:
            st.button(
                "Cargar",
                disabled=selected_config_name == "—",
                use_container_width=True,
                on_click=load_selected_config_callback,
            )
        with cb:
            st.button(
                "Eliminar",
                disabled=selected_config_name == "—",
                use_container_width=True,
                on_click=delete_selected_config_callback,
            )
        st.caption("Se guardan parámetros del análisis, no las rutas de las bases.")

        st.header("Fuentes")
        current_path = Path(st.text_input("Base actual (parquet)", str(core.DEFAULT_CURRENT)))
        hitos_path = Path(st.text_input("Base de hitos (parquet)", str(core.DEFAULT_HITOS)))
        conflict_path = Path(st.text_input("Dimensión conflictividad", str(core.DEFAULT_CONFLICT)))
        rrhh_path = Path(st.text_input("Fiscales y auxiliares", str(core.DEFAULT_RRHH)))
        territory_path_txt = st.text_input("Base territorial (opcional)", str(core.DEFAULT_TERRITORIO))
        territory_path = Path(territory_path_txt) if territory_path_txt.strip() else None

    required = [current_path, hitos_path, conflict_path, rrhh_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        st.error("Faltan archivos requeridos:\n\n" + "\n".join(f"- {path}" for path in missing))
        st.info("Copiá los archivos a la carpeta data/ o corregí las rutas en la barra lateral.")
        st.stop()

    try:
        meta = core.source_metadata(str(current_path), str(hitos_path))
    except Exception as exc:
        st.exception(exc)
        st.stop()

    dmaxs = [value for value in meta["fecha_max"].tolist() if pd.notna(value)]
    dmins = [value for value in meta["fecha_min"].tolist() if pd.notna(value)]
    overlap_max = min(dmaxs) if dmaxs else date.today()
    all_min = min(dmins) if dmins else overlap_max - timedelta(days=365)
    default_start = max(all_min, overlap_max - timedelta(days=364))

    with st.expander("Cortes de las fuentes", expanded=False):
        st.dataframe(meta, hide_index=True, use_container_width=True)
        if len(set(dmaxs)) > 1:
            st.warning(
                f"Las fuentes no llegan al mismo día. Para evitar mezclar cortes, el máximo común es {overlap_max:%d/%m/%Y}."
            )

    if pending is not None:
        if pending.get("jerarquia") in core.GRAIN_LABELS:
            st.session_state["cfg_grain_label"] = pending["jerarquia"]
        if pending.get("asignar_por") in core.AXIS_COLS:
            st.session_state["cfg_axis"] = pending["asignar_por"]
        st.session_state["cfg_start"] = safe_date(
            pending.get("desde"), default_start, all_min, overlap_max
        )
        st.session_state["cfg_end"] = safe_date(
            pending.get("hasta"), overlap_max, all_min, overlap_max
        )
        if st.session_state["cfg_start"] > st.session_state["cfg_end"]:
            st.session_state["cfg_start"] = st.session_state["cfg_end"]
        st.session_state["cfg_only_accusatory"] = bool(pending.get("solo_acusatorio", True))
        st.session_state["cfg_only_fiscal_units"] = bool(
            pending.get("solo_unidades_fiscales", True)
        )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        grain_label = st.selectbox(
            "Jerarquía a comparar", list(core.GRAIN_LABELS), index=1, key="cfg_grain_label"
        )
        grain = core.GRAIN_LABELS[grain_label]
    with col_b:
        axis = st.selectbox(
            "Asignar por",
            list(core.AXIS_COLS),
            index=0,
            help="Actuación es el eje recomendado para carga y litigación.",
            key="cfg_axis",
            on_change=reset_full_hierarchy_callback,
        )
    with col_c:
        start = st.date_input(
            "Desde",
            value=default_start,
            min_value=all_min,
            max_value=overlap_max,
            key="cfg_start",
            on_change=reset_full_hierarchy_callback,
        )
    with col_d:
        end = st.date_input(
            "Hasta",
            value=overlap_max,
            min_value=all_min,
            max_value=overlap_max,
            key="cfg_end",
            on_change=reset_full_hierarchy_callback,
        )

    if start > end:
        st.error("La fecha inicial no puede ser posterior a la final.")
        st.stop()

    # -------------------------------------------------------------------------
    # Universo y selección jerárquica EN CASCADA
    # -------------------------------------------------------------------------
    st.subheader("Unidades que participan")
    f1, f2 = st.columns(2)
    with f1:
        only_accusatory = st.checkbox(
            "Solo sistema procesal acusatorio",
            value=True,
            key="cfg_only_accusatory",
            help=(
                "Usa descripcion_sistemaprocesal = 'Acusatorio' cuando la fuente conserva esa variable; "
                "si la extracción ya es exclusivamente acusatoria, no agrega un criterio ficticio."
            ),
            on_change=reset_full_hierarchy_callback,
        )
    with f2:
        only_fiscal_units = st.checkbox(
            "Solo unidades fiscales (nombre comienza con 'Fiscalía')",
            value=True,
            key="cfg_only_fiscal_units",
            help=(
                "Filtra la unidad fiscal del eje elegido por nombres que comienzan con Fiscalía/Fiscalia."
            ),
            on_change=reset_full_hierarchy_callback,
        )

    st.caption(
        "Siempre se excluyen Justicia Nacional, Procuradurías y Casación. "
        "La selección funciona en cascada: Distrito → Unidad / sede fiscal → Oficina / área."
    )

    try:
        hierarchy = metrics.hierarchy_options(
            str(current_path),
            axis,
            start.isoformat(),
            end.isoformat(),
            only_accusatory,
            only_fiscal_units,
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

    if hierarchy.empty:
        st.warning("No hay unidades disponibles para los filtros seleccionados.")
        st.stop()

    district_options = sorted(hierarchy["distrito"].dropna().astype(str).unique().tolist())
    preferred_districts = None
    if pending is not None:
        preferred_districts = pending.get("distritos")
        if preferred_districts is None and grain == "distrito":
            preferred_districts = pending.get("unidades", [])
    ensure_multiselect_state("cfg_districts", district_options, preferred_districts)
    selected_districts = selection_row(
        "Distrito",
        "cfg_districts",
        district_options,
        on_change=district_changed_callback,
        reset_children=("cfg_fiscal_units", "cfg_offices"),
    )
    if not selected_districts:
        st.info("Seleccioná al menos un distrito.")
        st.stop()

    # Primer hijo: sólo unidades de los distritos que quedaron seleccionados.
    hierarchy_d = hierarchy[hierarchy["distrito"].isin(selected_districts)].copy()
    selected_fiscal_units: list[str] = []
    selected_offices: list[str] = []

    if grain in {"unidad", "oficina"}:
        unit_options = sorted(
            hierarchy_d["unidad_fiscal"].dropna().astype(str).unique().tolist()
        )
        preferred_units = None
        if pending is not None:
            preferred_units = pending.get("unidades_fiscales")
            if preferred_units is None and grain == "unidad":
                preferred_units = pending.get("unidades", [])
        ensure_multiselect_state("cfg_fiscal_units", unit_options, preferred_units)
        selected_fiscal_units = selection_row(
            "Unidad / sede fiscal",
            "cfg_fiscal_units",
            unit_options,
            on_change=fiscal_unit_changed_callback,
            reset_children=("cfg_offices",),
        )
        if not selected_fiscal_units:
            st.info("Seleccioná al menos una unidad / sede fiscal.")
            st.stop()

    # Segundo hijo: sólo oficinas de las unidades/sedes elegidas.
    if grain == "oficina":
        hierarchy_u = hierarchy_d[
            hierarchy_d["unidad_fiscal"].isin(selected_fiscal_units)
        ].copy()
        office_options = sorted(
            hierarchy_u["oficina_etiqueta"].dropna().astype(str).unique().tolist()
        )
        preferred_offices = None
        if pending is not None:
            preferred_offices = pending.get("oficinas")
            if preferred_offices is None:
                preferred_offices = pending.get("unidades", [])
        ensure_multiselect_state("cfg_offices", office_options, preferred_offices)
        selected_offices = selection_row(
            "Oficina / área",
            "cfg_offices",
            office_options,
            help_text="Se muestra como 'Unidad / sede · Oficina' para distinguir oficinas con igual nombre.",
        )
        if not selected_offices:
            st.info("Seleccioná al menos una oficina / área.")
            st.stop()

    if grain == "distrito":
        analysis_units = selected_districts
    elif grain == "unidad":
        analysis_units = selected_fiscal_units
    else:
        analysis_units = selected_offices

    # Auditoría visible de la cascada antes de calcular indicadores.
    audit = hierarchy_d.copy()
    if grain in {"unidad", "oficina"}:
        audit = audit[audit["unidad_fiscal"].isin(selected_fiscal_units)]
    if grain == "oficina":
        audit = audit[audit["oficina_etiqueta"].isin(selected_offices)]

    st.caption(f"Universo seleccionado: **{len(analysis_units)}** {grain_label.lower()}.")
    with st.expander("Auditar universo jerárquico", expanded=False):
        audit_show = audit[["distrito", "unidad_fiscal", "oficina", "casos"]].copy()
        audit_show = audit_show.rename(
            columns={
                "distrito": "Distrito",
                "unidad_fiscal": "Unidad / sede fiscal",
                "oficina": "Oficina / área",
                "casos": "Casos",
            }
        )
        st.dataframe(audit_show, hide_index=True, use_container_width=True)

    with st.spinner("Construyendo matriz de unidades..."):
        try:
            analysis_units_tuple = tuple(analysis_units)
            core_df = metrics.build_core_metrics(
                str(current_path), axis, grain, start.isoformat(), end.isoformat(),
                analysis_units_tuple, only_accusatory, only_fiscal_units
            )
            hitos = metrics.build_hitos_metrics(
                str(hitos_path), axis, grain, start.isoformat(), end.isoformat(),
                analysis_units_tuple, only_accusatory, only_fiscal_units
            )
            conflict = metrics.build_conflict_metrics(
                str(current_path), str(conflict_path), axis, grain, start.isoformat(), end.isoformat(),
                analysis_units_tuple, only_accusatory, only_fiscal_units
            )
            rrhh_raw = core.load_rrhh(str(rrhh_path))
            rrhh = core.rrhh_by_grain(rrhh_raw, grain, end)
            territorio = None
            if territory_path is not None and territory_path.exists():
                territorio = core.territorio_by_grain(core.load_territorio(str(territory_path)), grain)
            matrix = core.merge_matrix(core_df, hitos, conflict, rrhh, territorio)
        except Exception as exc:
            st.exception(exc)
            st.stop()

    if matrix.empty:
        st.warning("No hay datos para las unidades seleccionadas.")
        st.stop()

    conflict_labels = {}
    try:
        dimc = core.load_conflict(str(conflict_path))
        conflict_labels = dict(
            dimc[["tipo_conflictividad_v2_codigo", "tipo_conflictividad_v2"]]
            .drop_duplicates().dropna()
            .astype({"tipo_conflictividad_v2_codigo": int})
            .values
        )
    except Exception:
        pass

    st.subheader("Matriz analítica")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unidades", len(matrix), delta=f"de {len(analysis_units)} seleccionadas")
    c2.metric("Casos", f"{int(matrix['casos'].sum()):,}".replace(",", "."))
    c3.metric("Actuaciones", f"{int(matrix['actuaciones'].sum()):,}".replace(",", "."))
    c4.metric("Audiencias", f"{int(matrix['audiencias'].sum()):,}".replace(",", "."))

    rrhh_match = matrix["litigantes"].notna().mean() if "litigantes" in matrix else 0
    if rrhh_match < 0.8:
        st.warning(
            f"Cobertura del empalme de RRHH: {rrhh_match:.0%}. Conviene revisar equivalencias de nombres "
            "antes de interpretar indicadores por litigante."
        )

    families = core.feature_families(matrix)
    default_families = [
        "Volumen bruto", "Perfil procesal / tasas", "Conflictividad", "Litigación / RRHH"
    ]
    if pending is not None:
        loaded_families = [family for family in pending.get("familias", []) if family in families]
        st.session_state["cfg_families"] = loaded_families or default_families

    family_choice = st.multiselect(
        "Familias que entran al clustering",
        options=list(families),
        default=default_families,
        key="cfg_families",
    )
    default_features: list[str] = []
    for family in family_choice:
        default_features.extend(families[family])
    default_features = [
        feature for feature in dict.fromkeys(default_features) if feature in matrix.columns
    ]
    default_features = [
        feature for feature in default_features
        if feature not in {"casos_formalizados_flag", "poblacion_coiron_max", "casos_complejos"}
    ]

    feature_options = [
        column for column in matrix.columns
        if column not in {"unidad", "unidad_key", "territorio", "distrito_territorial"}
        and pd.api.types.is_numeric_dtype(matrix[column])
    ]
    labels_map = {name: core.display_name(name, conflict_labels) for name in feature_options}
    inv_labels = {label: name for name, label in labels_map.items()}
    if pending is not None:
        loaded_features = [
            feature for feature in pending.get("indicadores", []) if feature in labels_map
        ]
        selected_for_load = loaded_features or default_features
        st.session_state["cfg_features_labels"] = [
            labels_map[feature] for feature in selected_for_load
        ]

    selected_labels = st.multiselect(
        "Indicadores",
        options=[labels_map[column] for column in feature_options],
        default=[labels_map[column] for column in default_features if column in labels_map],
        key="cfg_features_labels",
    )
    features = [inv_labels[label] for label in selected_labels if label in inv_labels]
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
- **Conflictividad:** prevalencia de casos con al menos un delito de cada categoría; pueden superponerse.  
- **Litigantes:** personas activas al corte cuyo cargo es fiscal o auxiliar fiscal.  
- **Sistema acusatorio:** se usa la variable de sistema cuando está disponible en la fuente.  
- **Unidades fiscales:** filtro separado por nombre de unidad que comienza con `Fiscalía`.  
- **Exclusiones estructurales:** Justicia Nacional, Procuradurías y Casación no entran al universo comparable.
            """
        )

    if pending is not None:
        if pending.get("escalado") in ["Z-score", "Robusto"]:
            st.session_state["cfg_scaler"] = pending["escalado"]
        if "log_cantidades" in pending:
            st.session_state["cfg_log_counts"] = bool(pending["log_cantidades"])
        if pending.get("algoritmo") in ["Emergente automático", "Jerárquico Ward", "K-means", "DBSCAN"]:
            st.session_state["cfg_algo"] = pending["algoritmo"]
        if pending.get("minimo_casos") is not None:
            st.session_state["cfg_min_cases"] = max(1, int(pending["minimo_casos"]))

    a, b, ccol, d = st.columns(4)
    with a:
        scaler_name = st.selectbox("Escalado", ["Z-score", "Robusto"], index=0, key="cfg_scaler")
    with b:
        log_counts = st.checkbox(
            "Logaritmo en cantidades brutas",
            value=True,
            help="Conserva el tamaño pero reduce el dominio de unidades extremadamente grandes.",
            key="cfg_log_counts",
        )
    with ccol:
        algo = st.selectbox(
            "Algoritmo",
            ["Emergente automático", "Jerárquico Ward", "K-means", "DBSCAN"],
            index=0,
            key="cfg_algo",
        )
    with d:
        min_cases = st.number_input(
            "Mínimo de casos por unidad", min_value=1, value=30, step=10, key="cfg_min_cases"
        )

    work = matrix[matrix["casos"] >= min_cases].copy().reset_index(drop=True)
    st.caption(
        f"Participan efectivamente **{len(work)}** de las **{len(matrix)}** unidades con datos "
        f"después de aplicar el mínimo de {int(min_cases)} casos."
    )
    if len(work) < 3:
        st.warning("Quedaron menos de tres unidades después del filtro mínimo.")
        st.stop()

    raw_x, X, _ = core.prepare_X(
        work, features, log_counts=log_counts, scaler_name=scaler_name
    )
    scores = pd.DataFrame()
    chosen_k = None
    eps = None
    min_samples = None

    if algo == "Emergente automático":
        chosen_k, scores = core.best_k(X, "kmeans")
        cluster_labels = core.run_cluster(X, "K-means", chosen_k, 0.8, 3)
        algo_used = f"K-means automático (k={chosen_k})"
    elif algo in {"Jerárquico Ward", "K-means"}:
        method = "ward" if algo == "Jerárquico Ward" else "kmeans"
        auto_k, scores = core.best_k(X, method)
        max_k = min(10, len(work) - 1)
        if pending is not None and pending.get("k") is not None:
            st.session_state["cfg_k"] = min(max(2, int(pending["k"])), max_k)
        elif "cfg_k" in st.session_state:
            st.session_state["cfg_k"] = min(
                max(2, int(st.session_state["cfg_k"])), max_k
            )
        chosen_k = st.slider("Cantidad de clusters", 2, max_k, auto_k, key="cfg_k")
        cluster_labels = core.run_cluster(X, algo, chosen_k, 0.8, 3)
        algo_used = f"{'Ward' if algo == 'Jerárquico Ward' else 'K-means'} (k={chosen_k})"
    else:
        if pending is not None:
            if pending.get("dbscan_eps") is not None:
                st.session_state["cfg_eps"] = min(
                    max(0.2, float(pending["dbscan_eps"])), 3.0
                )
            if pending.get("dbscan_min_samples") is not None:
                st.session_state["cfg_min_samples"] = min(
                    max(2, int(pending["dbscan_min_samples"])), min(10, len(work))
                )
        if "cfg_min_samples" in st.session_state:
            st.session_state["cfg_min_samples"] = min(
                max(2, int(st.session_state["cfg_min_samples"])), min(10, len(work))
            )
        e1, e2 = st.columns(2)
        with e1:
            eps = st.slider("DBSCAN eps", 0.2, 3.0, 0.9, 0.05, key="cfg_eps")
        with e2:
            min_samples = st.slider(
                "DBSCAN min_samples", 2, min(10, len(work)), 3, key="cfg_min_samples"
            )
        cluster_labels = core.run_cluster(X, algo, None, eps, min_samples)
        algo_used = f"DBSCAN (eps={eps:.2f}, min={min_samples})"

    current_cfg = build_configuration(
        grain_label=grain_label,
        axis=axis,
        start=start,
        end=end,
        only_accusatory=only_accusatory,
        only_fiscal_units=only_fiscal_units,
        selected_districts=selected_districts,
        selected_fiscal_units=selected_fiscal_units,
        selected_offices=selected_offices,
        analysis_units=analysis_units,
        family_choice=family_choice,
        features=features,
        scaler_name=scaler_name,
        log_counts=log_counts,
        algo=algo,
        min_cases=int(min_cases),
        chosen_k=chosen_k,
        eps=eps,
        min_samples=min_samples,
    )

    with st.sidebar:
        st.divider()
        st.subheader("Guardar configuración actual")
        config_name = st.text_input(
            "Nombre", placeholder="Ej.: Acusatorio - perfil procesal", key="new_config_name"
        )
        if st.button(
            "Guardar configuración",
            use_container_width=True,
            disabled=not config_name.strip(),
        ):
            configs_now = load_saved_configs()
            configs_now[config_name.strip()] = current_cfg
            write_saved_configs(configs_now)
            st.success(f"Configuración '{config_name.strip()}' guardada.")

    work["cluster"] = cluster_labels
    work["cluster_etiqueta"] = work["cluster"].map(
        lambda value: "Atípica / ruido" if value == -1 else f"Cluster {value + 1}"
    )

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
    if len(set(cluster_labels)) > 1 and set(cluster_labels) != {-1}:
        try:
            sil = silhouette_score(X, cluster_labels) if -1 not in cluster_labels else np.nan
        except Exception:
            sil = np.nan
        m3.metric("Silhouette", "—" if np.isnan(sil) else f"{sil:.3f}")
    else:
        m3.metric("Silhouette", "—")

    if chosen_k is not None and algo != "DBSCAN":
        stability = core.feature_stability(raw_x, cluster_labels, chosen_k, n_iter=30)
        if stability is not None:
            st.caption(
                f"Estabilidad ante perturbación del 20% de los indicadores (ARI medio, 30 corridas): **{stability:.3f}**"
            )

    fig = px.scatter(
        work,
        x="PC1",
        y="PC2",
        color="cluster_etiqueta",
        hover_name="unidad",
        hover_data={
            "casos": ":,.0f",
            "actuaciones": ":,.0f",
            "audiencias": ":,.0f",
            "PC1": ":.2f",
            "PC2": ":.2f",
        },
        title="Proyección PCA de las unidades seleccionadas",
    )
    fig.update_traces(marker={"size": 11})
    st.plotly_chart(fig, use_container_width=True, key="pca_clusters")

    if not scores.empty:
        with st.expander("Elección del número de clusters", expanded=False):
            fig_k = px.line(
                scores,
                x="k",
                y="silhouette",
                markers=True,
                title="Silhouette por cantidad de clusters",
            )
            st.plotly_chart(fig_k, use_container_width=True, key="silhouette_k")

    profile_cols = [feature for feature in features if feature in work.columns]
    prof = work.groupby("cluster_etiqueta", dropna=False)[profile_cols].mean(numeric_only=True)
    prof.insert(0, "n_unidades", work.groupby("cluster_etiqueta").size())
    prof = prof.reset_index().rename(
        columns={column: labels_map.get(column, column) for column in prof.columns}
    )
    st.markdown("**Perfil promedio de cada cluster**")
    st.dataframe(prof, hide_index=True, use_container_width=True)

    st.subheader("Comparar una unidad")
    selected_unit = st.selectbox("Unidad", work["unidad"].sort_values().tolist())
    idx = int(work.index[work["unidad"] == selected_unit][0])
    dist = np.sqrt(((X - X[idx]) ** 2).sum(axis=1))
    neigh = work[
        ["unidad", "cluster_etiqueta", "casos", "actuaciones", "audiencias"]
    ].copy()
    neigh["distancia_estandarizada"] = dist
    neigh = (
        neigh[neigh["unidad"] != selected_unit]
        .sort_values("distancia_estandarizada")
        .head(8)
    )
    st.dataframe(neigh, hide_index=True, use_container_width=True)

    with st.expander("Tabla completa", expanded=False):
        show_cols = ["unidad", "cluster_etiqueta"] + [
            feature for feature in features if feature in work.columns
        ]
        st.dataframe(
            work[show_cols].rename(
                columns={column: labels_map.get(column, column) for column in show_cols}
            ),
            hide_index=True,
            use_container_width=True,
        )

    export_cols = [column for column in work.columns if column != "unidad_key"]
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
