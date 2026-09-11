from __future__ import annotations

"""Capa de compatibilidad sobre la v0.4.

Corrige la rotulación del universo acusatorio y deja disponible en session_state
la matriz analítica construida por la pantalla principal, para reutilizarla en
el comparador de unidades similares sin volver a redefinir métricas.
"""

import app_v4


APP_VERSION = "0.4.2"

_original_checkbox = app_v4.st.checkbox
if not hasattr(app_v4.core, "_clusterizador_merge_matrix_original"):
    app_v4.core._clusterizador_merge_matrix_original = app_v4.core.merge_matrix


def _checkbox_with_correct_accusatory_label(label, *args, **kwargs):
    if label == "Solo unidades fiscales (nombre comienza con 'Fiscalía')":
        label = "Solo estructura fiscal del acusatorio"
        kwargs["help"] = (
            "Incluye unidades cuyo nombre comienza con 'Unidad Fiscal' o "
            "'Sede Fiscal Descentralizada'. 'Fiscalía Federal' corresponde a la "
            "estructura del sistema mixto y queda fuera de este filtro."
        )
    return _original_checkbox(label, *args, **kwargs)


def _merge_matrix_and_capture(*args, **kwargs):
    original = app_v4.core._clusterizador_merge_matrix_original
    matrix = original(*args, **kwargs)
    app_v4.st.session_state["_comparador_matrix"] = matrix.copy()
    app_v4.st.session_state["_comparador_context"] = {
        "jerarquia": app_v4.st.session_state.get("cfg_grain_label"),
        "asignar_por": app_v4.st.session_state.get("cfg_axis"),
        "desde": str(app_v4.st.session_state.get("cfg_start", "")),
        "hasta": str(app_v4.st.session_state.get("cfg_end", "")),
        "minimo_casos": int(app_v4.st.session_state.get("cfg_min_cases", 1) or 1),
    }
    return matrix


def main() -> None:
    app_v4.st.checkbox = _checkbox_with_correct_accusatory_label
    app_v4.core.merge_matrix = _merge_matrix_and_capture
    app_v4.APP_VERSION = APP_VERSION
    app_v4.main()


if __name__ == "__main__":
    main()
