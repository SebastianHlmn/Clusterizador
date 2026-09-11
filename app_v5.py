from __future__ import annotations

"""Corrección de interfaz sobre la v0.4.

La lógica de selección y clustering sigue en app_v4. Esta capa corrige la
rotulación del filtro organizacional para reflejar la estructura real del
acusatorio en Coirón/UNISA: Unidad Fiscal / Sede Fiscal Descentralizada.
"""

import app_v4


APP_VERSION = "0.4.1"

_original_checkbox = app_v4.st.checkbox


def _checkbox_with_correct_accusatory_label(label, *args, **kwargs):
    if label == "Solo unidades fiscales (nombre comienza con 'Fiscalía')":
        label = "Solo estructura fiscal del acusatorio"
        kwargs["help"] = (
            "Incluye unidades cuyo nombre comienza con 'Unidad Fiscal' o "
            "'Sede Fiscal Descentralizada'. 'Fiscalía Federal' corresponde a la "
            "estructura del sistema mixto y queda fuera de este filtro."
        )
    return _original_checkbox(label, *args, **kwargs)


def main() -> None:
    # app_v4 mantiene toda la lógica de cascada; metrics_v4 contiene el criterio
    # corregido. Sólo sustituimos la etiqueta heredada para no mostrar el criterio
    # invertido en pantalla.
    app_v4.st.checkbox = _checkbox_with_correct_accusatory_label
    app_v4.APP_VERSION = APP_VERSION
    app_v4.main()


if __name__ == "__main__":
    main()
