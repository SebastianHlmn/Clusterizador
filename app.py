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


APP_VERSION = "0.2.0"
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configuraciones"
CONFIG_FILE = CONFIG_DIR / "clusterizador_configuraciones.json"


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


def safe_date(value: object, fallback: date, lower: date, upper: date) -> date:
    try:
        d = date.fromisoformat(str(value))
    except Exception:
        d = fallback
    return min(max(d, lower), upper)


def build_configuration(
    *,
    grain_label: str,
    axis: str,
    start: date,
    end: date,
    selected_units: Sequence[str],
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
        "unidades": list(selected_units),
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
            "Configuración guardada",
            options=["—"] + config_names,
            key="saved_config_selector",
        )
        ca, cb = st.columns(2)
        with ca:
            if st.button("Cargar", disabled=selected_config_name == "—", use_container_width=True):
                st.session_state["_pending_cluster_config"] = saved_configs[selected_config_name]
                st.rerun()
        with cb:
            if st.button("Eliminar", disabled=selected_config_name == "—", use_container_width=True):
                saved_configs.pop(selected_config_name, None)
                write_saved_configs(saved_configs)
                st.session_state.pop("saved_config_selector", None)
                st.rerun()
        st.caption("Se guardan parámetros del análisis, no las rutas de las bases.")

        st.header("Fuentes")
        current_path = Path(st.text_input("Base actual (parquet)", str(core.DEFAULT_CURRENT)))
        hitos_path = Path(st.text_input("Base de hitos (parquet)", str(core.DEFAULT_HITOS)))
        conflict_path = Path(st.text_input("Dimensión conflictividad", str(core.DEFAULT_CONFLICT)))
        rrhh_path = Path(st.text_input("Fiscales y auxiliares", str(core.DEFAULT_RRHH)))
        territory_path_txt = st.text_input("Base territorial (opcional)", str(core.DEFAULT_TERRITORIO))
        territory_path = Path(territory_path_txt) if territory_path_txt.strip() else None

    required = [current_path, hitos_path, conflict_path, rrhh_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        st.error("Faltan archivos requeridos:\n\n" + "\n".join(f"- {x}" for x in missing))
        st.info("Copiá los archivos a la carpeta data/ o corregí las rutas en la barra lateral.")
        st.stop()

    try:
        meta = core.source_metadata(str(current_path), str(hitos_path))
    except Exception as e:
        st.exception(e)
        st.stop()

    dmaxs = [d for d in meta["fecha_max"].tolist() if pd.notna(d)]
    dmins = [d for d in meta["fecha_min"].tolist() if pd.notna(d)]
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
        st.session_state["cfg_start"] = safe_date(pending.get("desde"), default_start, all_min, overlap_max)
        st.session_state["cfg_end"] = safe_date(pending.get("hasta"), overlap_max, all_min, overlap_max)
        if st.session_state["cfg_start"] > st.session_state["cfg_end"]:
            st.session_state["cfg_start"] = st.session_state["cfg_end"]

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
        )
    with col_c:
        start = st.date_input(
            "Desde", value=default_start, min_value=all_min, max_value=overlap_max, key="cfg_start"
        )
    with col_d:
        end = st.date_input(
            "Hasta", value=overlap_max, min_value=all_min, max_value=overlap_max, key="cfg_end"
        )

    if start > end:
        st.error("La fecha inicial no puede ser posterior a la final.")
        st.stop()

    with st.spinner("Construyendo matriz de unidades..."):
        try:
            core_df = core.build_core_metrics(str(current_path), axis, grain, start.isoformat(), end.isoformat())
            hitos = core.build_hitos_metrics(str(hitos_path), axis, grain, start.isoformat(), end.isoformat())
            conflict = core.build_conflict_metrics(
                str(current_path), str(conflict_path), axis, grain, start.isoformat(), end.isoformat()
            )
            rrhh_raw = core.load_rrhh(str(rrhh_path))
            rrhh = core.rrhh_by_grain(rrhh_raw, grain, end)
            territorio = None
            if territory_path is not None and territory_path.exists():
                territorio = core.territorio_by_grain(core.load_territorio(str(territory_path)), grain)
            matrix = core.merge_matrix(core_df, hitos, conflict, rrhh, territorio)
        except Exception as e:
            st.exception(e)
            st.stop()

    if matrix.empty:
        st.warning("No hay unidades para la combinación seleccionada.")
        st.stop()

    conflict_labels = {}
    try:
        dimc = core.load_conflict(str(conflict_path))
        conflict_labels = dict(
            dimc[["tipo_conflictividad_v2_codigo", "tipo_conflictividad_v2"]]
            .drop_duplicates()
            .dropna()
            .astype({"tipo_conflictividad_v2_codigo": int})
            .values
        )
    except Exception:
        pass

    unit_options = sorted(matrix["unidad"].dropna().astype(str).unique().tolist())
    if pending is not None:
        requested = pending.get("unidades") or unit_options
        st.session_state["cfg_units"] = [u for u in requested if u in unit_options]
    elif "cfg_units" in st.session_state:
        previous = list(st.session_state["cfg_units"])
        valid = [u for u in previous if u in unit_options]
        st.session_state["cfg_units"] = unit_options if previous and not valid else valid

    st.subheader("Unidades que participan")
    ua, ub, uc = st.columns([6, 1, 1])
    with ua:
        selected_units = st.multiselect(
            "Seleccioná las unidades que entran al clustering",
            options=unit_options,
            default=unit_options,
            key="cfg_units",
        )
    with ub:
        st.write("")
        st.write("")
        if st.button("Todas", use_container_width=True):
            st.session_state["cfg_units"] = unit_options
            st.rerun()
    with uc:
        st.write("")
        st.write("")
        if st.button("Ninguna", use_container_width=True):
            st.session_state["cfg_units"] = []
            st.rerun()

    if not selected_units:
        st.info("Seleccioná al menos tres unidades para calcular clusters.")
        st.stop()

    selected_matrix = matrix[matrix["unidad"].isin(selected_units)].copy()

    st.subheader("Matriz analítica")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unidades seleccionadas", len(selected_matrix), delta=f"de {len(matrix)} disponibles")
    c2.metric("Casos", f"{int(selected_matrix['casos'].sum()):,}".replace(",", "."))
    c3.metric("Actuaciones", f"{int(selected_matrix['actuaciones'].sum()):,}".replace(",", "."))
    c4.metric("Audiencias", f"{int(selected_matrix['audiencias'].sum()):,}".replace(",", "."))

    rrhh_match = selected_matrix["litigantes"].notna().mean() if "litigantes" in selected_matrix else 0
    if rrhh_match < 0.8:
        st.warning(
            f"Cobertura del empalme de RRHH en las unidades seleccionadas: {rrhh_match:.0%}. "
            "Conviene revisar equivalencias de nombres antes de interpretar indicadores por litigante."
        )

    families = core.feature_families(selected_matrix)
    default_families = ["Volumen bruto", "Perfil procesal / tasas", "Conflictividad", "Litigación / RRHH"]
    if pending is not None:
        loaded_families = [f for f in pending.get("familias", []) if f in families]
        st.session_state["cfg_families"] = loaded_families or default_families

    family_choice = st.multiselect(
        "Familias que entran al clustering",
        options=list(families),
        default=default_families,
        key="cfg_families",
    )
    default_features = []
    for fam in family_choice:
        default_features.extend(families[fam])
    default_features = [f for f in dict.fromkeys(default_features) if f in selected_matrix.columns]
    default_features = [
        f for f in default_features
        if f not in {"casos_formalizados_flag", "poblacion_coiron_max", "casos_complejos"}
    ]

    feature_options = [
        c for c in selected_matrix.columns
        if c not in {"unidad", "unidad_key", "territorio", "distrito_territorial"}
        and pd.api.types.is_numeric_dtype(selected_matrix[c])
    ]
    labels_map = {name: core.display_name(name, conflict_labels) for name in feature_options}
    inv_labels = {label: name for name, label in labels_map.items()}
    if pending is not None:
        loaded_features = [f for f in pending.get("indicadores", []) if f in labels_map]
        selected_for_load = loaded_features or default_features
        st.session_state["cfg_features_labels"] = [labels_map[f] for f in selected_for_load if f in labels_map]

    selected_labels = st.multiselect(
        "Indicadores",
        options=[labels_map[c] for c in feature_options],
        default=[labels_map[c] for c in default_features if c in labels_map],
        key="cfg_features_labels",
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
- **Conflictividad:** prevalencia de casos con al menos un delito de cada categoría; pueden superponerse.  
- **Litigantes:** personas activas al corte cuyo cargo es fiscal o auxiliar fiscal.
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

    work = selected_matrix[selected_matrix["casos"] >= min_cases].copy().reset_index(drop=True)
    st.caption(
        f"Participan efectivamente **{len(work)}** de las **{len(selected_units)}** unidades seleccionadas "
        f"después de aplicar el mínimo de {int(min_cases)} casos."
    )
    if len(work) < 3:
        st.warning("Quedaron menos de tres unidades después del filtro mínimo.")
        st.stop()

    raw_x, X, _ = core.prepare_X(work, features, log_counts=log_counts, scaler_name=scaler_name)

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
            st.session_state["cfg_k"] = min(max(2, int(st.session_state["cfg_k"])), max_k)
        chosen_k = st.slider("Cantidad de clusters", 2, max_k, auto_k, key="cfg_k")
        cluster_labels = core.run_cluster(X, algo, chosen_k, 0.8, 3)
        algo_used = f"{'Ward' if algo == 'Jerárquico Ward' else 'K-means'} (k={chosen_k})"
    else:
        if pending is not None:
            if pending.get("dbscan_eps") is not None:
                st.session_state["cfg_eps"] = min(max(0.2, float(pending["dbscan_eps"])), 3.0)
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
        selected_units=selected_units,
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
            "Nombre", placeholder="Ej.: Unidades grandes - perfil procesal", key="new_config_name"
        )
        if st.button("Guardar configuración", use_container_width=True, disabled=not config_name.strip()):
            configs_now = load_saved_configs()
            configs_now[config_name.strip()] = current_cfg
            write_saved_configs(configs_now)
            st.success(f"Configuración '{config_name.strip()}' guardada.")

    work["cluster"] = cluster_labels
    work["cluster_etiqueta"] = work["cluster"].map(
        lambda x: "Atípica / ruido" if x == -1 else f"Cluster {x + 1}"
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
    if len(set(cluster_labels)) > 1 and not (set(cluster_labels) == {-1}):
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
                scores, x="k", y="silhouette", markers=True, title="Silhouette por cantidad de clusters"
            )
            st.plotly_chart(fig_k, use_container_width=True, key="silhouette_k")

    profile_cols = [f for f in features if f in work.columns]
    prof = work.groupby("cluster_etiqueta", dropna=False)[profile_cols].mean(numeric_only=True)
    prof.insert(0, "n_unidades", work.groupby("cluster_etiqueta").size())
    prof = prof.reset_index().rename(columns={c: labels_map.get(c, c) for c in prof.columns})
    st.markdown("**Perfil promedio de cada cluster**")
    st.dataframe(prof, hide_index=True, use_container_width=True)

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

    export_cols = [c for c in work.columns if c != "unidad_key"]
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
