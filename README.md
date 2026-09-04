# Clusterizador de unidades fiscales — v0.1

Primera versión del clusterizador para comparar unidades fiscales del sistema acusatorio a partir de datos Coirón/UNISA, conflictividad y dotación de fiscales/auxiliares.

## Regla de conteo

- Casos = `COUNT(DISTINCT IdCasoOriginal)`.
- Actuaciones = `COUNT(DISTINCT IdActuacion)`.
- Audiencias = actuaciones distintas con `ActuacionAudiencia == "Audiencia"`.
- Los hitos/salidas se toman de `EstadoInformeConsistencia` y se cuentan por `IdCasoOriginal` distinto.
- Cada hito se conserva en cantidad bruta y se deriva su tasa sobre casos.
- Las prevalencias de conflictividad no tienen por qué sumar 100%: un caso puede tener delitos de más de una categoría.

## Jerarquías

La app permite comparar:

1. Distrito.
2. Unidad / sede.
3. Oficina / área.

Y permite asignar los datos por unidad de **actuación** (recomendado para actividad/litigación), unidad **actual** o unidad de **ingreso**.

## Familias de variables

- Volumen bruto: casos, actuaciones, audiencias y cantidades de casos con hitos.
- Perfil procesal: actuaciones/caso, audiencias/caso, tasas de hitos, complejidad.
- Conflictividad: prevalencia de cada `tipo_conflictividad_v2`.
- Litigación/RRHH: fiscales, auxiliares, litigantes y carga por litigante.
- Territorio: población, superficie y densidad; se incorpora como descriptor y se puede activar si se desea.

## Clusters

- Emergente automático: selecciona `k` por silhouette y ejecuta K-means.
- Jerárquico Ward.
- K-means manual.
- DBSCAN para agrupamientos por densidad y detección de unidades que quedan como ruido.

La app agrega una prueba simple de estabilidad: vuelve a clusterizar 30 veces usando el 80% de los indicadores y calcula ARI contra la solución de referencia.

## Archivos

Crear una carpeta `data` junto a la app y copiar allí:

- `data_final_acusatorio20260904.parquet`
- `baseUnisaAcusatorio.parquet`
- `dim_delito_conflictividad_completa_v2.xlsx`
- `Fiscales_y_AF_jurisdicciones_implementadas.xlsx`
- opcional: `Base_Ancha_Territorios_Fiscalias_Sedes_2026_superficie_oficial.xlsx`

Los nombres/rutas se pueden cambiar desde la barra lateral.

## Windows

Doble clic en `run_windows.bat`. La primera vez crea `.venv`, instala dependencias y levanta Streamlit en el puerto 8501 escuchando en `0.0.0.0`.

Luego, desde otra máquina de la red, se accede con:

`http://IP_DE_LA_MAQUINA:8501`

El firewall de Windows debe permitir conexiones entrantes al puerto 8501.

## Decisiones metodológicas de esta versión

1. Se usa una ventana temporal común para todas las unidades. Esto evita comparar como si fueran equivalentes períodos de exposición distintos desde la implementación.
2. Si las dos fuentes parquet tienen cortes distintos, la app limita el máximo al último día común.
3. Se conservan cantidades brutas y tasas. Para clusterizar, las variables se estandarizan; las cantidades pueden transformarse con `log1p` para reducir el peso de escalas extremadamente grandes sin perder la señal de volumen.
4. El territorio queda fuera del conjunto de variables por defecto: sirve primero para describir/interpretar los clusters y puede activarse después.
5. A nivel oficina/área, la app informa la cobertura del empalme con RRHH porque la nomenclatura de áreas puede requerir un diccionario explícito de equivalencias.
