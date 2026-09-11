# Clusterizador de unidades fiscales — v0.3

Clusterizador para comparar unidades fiscales del sistema acusatorio a partir de datos Coirón/UNISA, conflictividad y dotación de fiscales/auxiliares.

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

## Selección jerárquica de unidades

La selección del universo se hace antes de calcular los indicadores.

- Si se compara por **Distrito**, se seleccionan los distritos participantes.
- Si se compara por **Unidad / sede**, primero se seleccionan distritos y luego las unidades pertenecientes a esos distritos.
- Si se compara por **Oficina / área**, se seleccionan distrito, unidad y finalmente oficina. La oficina se identifica como `Unidad / sede · Oficina` para evitar ambigüedades entre oficinas con igual nombre.

Cada nivel tiene acciones **Todas** y **Ninguna**. El mínimo de casos se aplica después de construir la matriz del universo seleccionado.

## Filtro de sistema acusatorio

La opción **Solo unidades del sistema acusatorio** está activa por defecto.

La regla es explícita: se consideran unidades acusatorias aquellas cuyo valor de `unidadfiscal_actuacion`, `unidadfiscal_actual` o `unidadfiscal_ingreso` —según el eje elegido— comienza con `Fiscalía`, aceptando la forma con o sin tilde y sin distinguir mayúsculas/minúsculas.

El filtro se aplica **antes de la agregación**. Por lo tanto, cuando se compara a nivel Distrito, los casos, actuaciones, audiencias, hitos y conflictividad del distrito se calculan sólo con registros pertenecientes a esas unidades fiscales.

## Configuraciones guardadas

La barra lateral permite guardar, cargar y eliminar configuraciones con nombre. Se guardan:

- jerarquía y eje de asignación;
- fechas de análisis;
- filtro de unidades acusatorias;
- distritos, unidades fiscales y oficinas seleccionadas;
- familias e indicadores seleccionados;
- escalado y transformación logarítmica;
- algoritmo y mínimo de casos;
- cantidad de clusters cuando corresponde;
- parámetros de DBSCAN cuando corresponde.

Las configuraciones se almacenan localmente en `configuraciones/clusterizador_configuraciones.json`. Ese archivo queda fuera de Git mediante `.gitignore`. No se guardan rutas de las bases dentro de las configuraciones.

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

También se puede ejecutar desde el entorno `analisisconsultas` con:

`streamlit run app.py`

## Decisiones metodológicas

1. Se usa una ventana temporal común para todas las unidades.
2. Si las dos fuentes parquet tienen cortes distintos, la app limita el máximo al último día común.
3. Se conservan cantidades brutas y tasas. Para clusterizar, las variables se estandarizan; las cantidades pueden transformarse con `log1p`.
4. El filtro de unidades acusatorias se realiza antes de agregar los indicadores.
5. El territorio queda fuera del conjunto de variables por defecto y se utiliza inicialmente como descriptor.
6. A nivel oficina/área, la app informa la cobertura del empalme con RRHH porque la nomenclatura de áreas puede requerir un diccionario explícito de equivalencias.
