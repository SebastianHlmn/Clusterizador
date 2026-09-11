from __future__ import annotations

import pandas as pd
import streamlit as st

import similarity_engine as sim


st.set_page_config(page_title="Unidades similares", layout="wide")
st.title("Buscar unidades similares")
st.caption(
    "Compara perfiles por dimensiones. Una distancia menor indica mayor cercanía dentro del universo seleccionado."
)

matrix = st.session_state.get("_comparador_matrix")
context = st.session_state.get("_comparador_context", {})

if matrix is None or not isinstance(matrix, pd.DataFrame) or matrix.empty:
    st.info(
        "Primero construí el universo en la pantalla principal. Después volvé a esta página: "
        "el comparador reutiliza exactamente esa matriz y esos filtros."
    )
    st.stop()

work = matrix.copy()
min_cases = int(st.session_state.get("cfg_min_cases", context.get("minimo_casos", 1)) or 1)
if "casos" in work.columns:
    work = work[work["casos"] >= min_cases].copy()
work = work.reset_index(drop=True)

if len(work) < 2:
    st.warning("El universo actual no tiene al menos dos unidades comparables.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Unidades", len(work))
c2.metric("Jerarquía", context.get("jerarquia") or "—")
c3.metric("Asignación", context.get("asignar_por") or "—")
c4.metric("Período", f"{context.get('desde', '—')} a {context.get('hasta', '—')}")

profile = st.selectbox("Criterio de comparación", list(sim.PROFILE_WEIGHTS), index=1)
weights = dict(sim.PROFILE_WEIGHTS[profile])
available = sim.available_families(work)

with st.expander("Ajustar peso de las dimensiones", expanded=False):
    st.caption("Los pesos se aplican por dimensión, no por cantidad de columnas. Así una familia con muchos indicadores no domina por sí sola.")
    cols = st.columns(3)
    for pos, family in enumerate(sim.FAMILY_SPECS):
        if family not in available:
            continue
        with cols[pos % 3]:
            weights[family] = st.slider(
                family,
                min_value=0.0,
                max_value=2.0,
                value=float(weights.get(family, 0.0)),
                step=0.1,
                key=f"sim_weight_{profile}_{family}",
            )

units = sorted(work["unidad"].dropna().astype(str).unique().tolist())
reference = st.selectbox("Unidad de referencia", units)
n_neighbors = st.slider("Cantidad de unidades comparables", 3, min(15, len(units) - 1), min(8, len(units) - 1))

try:
    ranking, z, active_families = sim.rank_neighbors(work, reference, weights)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if ranking.empty:
    st.warning("No fue posible calcular unidades comparables con las dimensiones activas.")
    st.stop()

show = ranking.head(n_neighbors).copy()
rename = {"puesto": "Puesto", "unidad": "Unidad", "distancia_global": "Distancia global"}
for family in active_families:
    rename[f"dist_{family}"] = family
show = show.rename(columns=rename)
num_cols = [c for c in show.columns if c not in {"Puesto", "Unidad"}]
show[num_cols] = show[num_cols].round(3)

st.subheader("Unidades más cercanas")
st.dataframe(show, hide_index=True, use_container_width=True)

comparison_unit = st.selectbox(
    "Ver detalle con",
    ranking.head(n_neighbors)["unidad"].astype(str).tolist(),
)
row = ranking.loc[ranking["unidad"].astype(str) == comparison_unit].iloc[0]
detail = sim.pair_detail(work, z, active_families, reference, comparison_unit)
summary = sim.build_summary(reference, comparison_unit, row, detail)

st.subheader("Lectura rápida")
st.write(summary)

family_rows = []
for family in active_families:
    value = row.get(f"dist_{family}")
    if pd.notna(value):
        family_rows.append({"Dimensión": family, "Distancia": round(float(value), 3)})
family_df = pd.DataFrame(family_rows).sort_values("Distancia")
st.dataframe(family_df, hide_index=True, use_container_width=True)

st.markdown("**Indicadores que más diferencian el par**")
detail_show = detail.dropna(subset=["diferencia_estandarizada"]).copy()
detail_show = detail_show.sort_values("diferencia_estandarizada", ascending=False).head(10)
detail_show["Indicador"] = detail_show["indicador"].map(sim.humanize)
detail_show = detail_show.rename(columns={
    "dimension": "Dimensión",
    "valor_referencia": reference,
    "valor_comparable": comparison_unit,
    "diferencia_estandarizada": "Diferencia estandarizada",
})
detail_show = detail_show[["Dimensión", "Indicador", reference, comparison_unit, "Diferencia estandarizada"]]
detail_show["Diferencia estandarizada"] = detail_show["Diferencia estandarizada"].round(3)
st.dataframe(detail_show, hide_index=True, use_container_width=True)

csv = ranking.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar ranking de similitud (CSV)",
    data=csv,
    file_name=f"unidades_similares_{reference}.csv",
    mime="text/csv",
)

st.caption(
    "La similitud es descriptiva y relativa al universo, período y pesos seleccionados; no expresa desempeño ni calidad institucional."
)
