# streamlit_app.py

import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
from sqlalchemy import create_engine
from pyproj import Transformer
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import folium
from streamlit_folium import st_folium

# Crear la conexión a la base de datos usando SQLAlchemy y st.secrets
def create_db_connection():
    db_url = (
        f"postgresql://{st.secrets['arqueocronochile_db']['username']}:"
        f"{st.secrets['arqueocronochile_db']['password']}@"
        f"{st.secrets['arqueocronochile_db']['host']}:"
        f"{st.secrets['arqueocronochile_db']['port']}/"
        f"{st.secrets['arqueocronochile_db']['database']}"
    )
    engine = create_engine(db_url)
    return engine

# Función para obtener los datos combinados de las cuatro tablas
@st.cache_data
def get_combined_data():
    engine = create_db_connection()
    query = """
    SELECT 
        m.cod_lab AS cod_lab, 
        m.id_muestra AS id_muestra, 
        m.año_realizacion, 
        m.tipo_fechado, 
        m.sitio, 
        m.localizacion_administrativa_id, 
        m.unidad, 
        m.nivel, 
        m.estrato, 
        m.profundidad, 
        m.proyecto, 
        m.responsable, 
        m.coleccion, 
        m.información_contextual, 
        m.publicaciones,
        m.compilacion_fechados, 
        m.reporte, 
        m.detalle_muestra, 
        m.especimen, 
        la.id_localizacion_administrativa AS la_id_localizacion_administrativa, 
        la.localidad, 
        la.comuna, 
        la.provincia, 
        la.region, 
        la.latitud_4326, 
        la.longitud_4326, 
        la.este_32718, 
        la.norte_32718, 
        la.este_32719, 
        la.norte_32719, 
        la.geom_4326, 
        la.geom_32718, 
        la.geom_32719, 
        fc.fecha_ap AS fc_fecha_ap, 
        fc.error_ap AS fc_error_ap, 
        fc.instrumento AS fc_instrumento, 
        fc.tipo_muestra AS fc_tipo_muestra, 
        fc.biomolecula, 
        fc.c13_c14, 
        fc.d13c, 
        fc.cn, 
        fc.collagen_yield, 
        fc.d13, 
        fc.d14, 
        ft.instrumento AS ft_instrumento, 
        ft.tipo_muestra AS ft_tipo_muestra, 
        ft.p_gy, 
        ft.error_p_gy, 
        ft.d_gy_año, 
        ft.año_base, 
        ft.edad_ap AS ft_edad_ap, 
        ft.error_edad_ap AS ft_error_edad_ap, 
        ft.fecha_dc
    FROM Muestras m
    JOIN localizacion_administrativa la ON m.localizacion_administrativa_id = la.id_localizacion_administrativa
    LEFT JOIN fechado_carbono fc ON m.id_muestra = fc.id_muestra
    LEFT JOIN fechado_termoluminiscencia ft ON m.id_muestra = ft.id_muestra;
    """
    df = pd.read_sql_query(query, engine)
    return df

# Crear un transformador para convertir de UTM a WGS84
transformer18 = Transformer.from_crs("epsg:32718", "epsg:4326", always_xy=True)
transformer19 = Transformer.from_crs("epsg:32718", "epsg:4326", always_xy=True)

# Función para convertir de UTM 18S a WGS84 utilizando Transformer
@st.cache_data
def utm_to_wgs_transformer18(este, norte):
    lon, lat = transformer18.transform(este, norte)
    return lat, lon

# Función para convertir de UTM 19S a WGS84 utilizando Transformer
@st.cache_data
def utm_to_wgs_transformer19(este, norte):
    lon, lat = transformer19.transform(este, norte)
    return lat, lon
    
# Configuraciones
# Configurar la página para utilizar el layout de ancho completo
st.set_page_config(
    page_title="ArqueoCronoChile",
    page_icon="⏱️",  
    layout="wide",  
    initial_sidebar_state="collapsed",  
)

# Trabajo con el DataFrame
df = get_combined_data()

# Mapa
# Aplicar la conversión utilizando Transformer18S
df['lat'], df['lon'] = zip(*df.apply(lambda x: utm_to_wgs_transformer18(x['este_32718'], x['norte_32718']), axis=1))
df['lat'] = df['lat'].astype(float)
df['lon'] = df['lon'].astype(float)
map_data = df[['lat', 'lon']].dropna()


# Limpieza de dataframe
columnas_a_excluir = ['id_muestra', 'localizacion_administrativa_id', 'la_id_localizacion_administrativa', 'geom_4326', 'geom_32718', 'geom_32719']  # Ejemplo de columnas a excluir
df_filtrado = df.drop(columns=columnas_a_excluir)
    
# Unificacion de fechas ap y error
# Crear la columna 'fecha_ap' combinando 'fc_fecha_ap' y 'ft_edad_ap'
df_filtrado['fecha_ap'] = df_filtrado.apply(
    lambda row: row['fc_fecha_ap'] if pd.notna(row['fc_fecha_ap']) else row['ft_edad_ap'], axis=1)
# Crear la columna 'error_ap' combinando 'fc_error_ap' y 'ft_error_edad_ap'
df_filtrado['error_ap'] = df_filtrado.apply(
    lambda row: row['fc_error_ap'] if pd.notna(row['fc_error_ap']) else row['ft_error_edad_ap'], axis=1)
# Crear la columna 'instrumento' combinando 'fc_instrumento' y 'ft_instrumento'
df_filtrado['instrumento'] = df_filtrado.apply(
    lambda row: row['fc_instrumento'] if pd.notna(row['fc_instrumento']) else row['ft_instrumento'], axis=1)
# Crear la columna 'tipo_muestra' combinando 'fc_tipo_muestra' y 'ft_tipo_muestra'
df_filtrado['tipo_muestra'] = df_filtrado.apply(
    lambda row: row['fc_tipo_muestra'] if pd.notna(row['fc_tipo_muestra']) else row['ft_tipo_muestra'], axis=1)

# Filtrar columnas unidas
columnas_a_excluir2 = ['fc_fecha_ap', 'ft_edad_ap', 'fc_error_ap', 'ft_error_edad_ap', 'fc_tipo_muestra', 'ft_tipo_muestra', 'fc_instrumento', 'ft_instrumento']  # Ejemplo de columnas a excluir
df_filtrado2 = df_filtrado.drop(columns=columnas_a_excluir2)

# Orden final del DataFrame
df_neworder = df_filtrado2[[
    "cod_lab",
    "tipo_fechado",
    "fecha_ap",
    "error_ap",
    "sitio",
    "año_realizacion",
    "instrumento",
    "tipo_muestra",
    "detalle_muestra",
    "especimen",
    "unidad",
    "nivel",
    "estrato",
    "profundidad",
    "proyecto",
    "responsable",
    "coleccion",
    "información_contextual",
    "publicaciones",
    "compilacion_fechados",
    "reporte",
    "localidad",
    "comuna",
    "provincia",
    "region",
    "latitud_4326",
    "longitud_4326",
    "este_32718",
    "norte_32718",
    "este_32719",
    "norte_32719",
    "biomolecula",
    "c13_c14",
    "d13c",
    "cn",
    "collagen_yield",
    "d13",
    "d14",
    "p_gy",
    "error_p_gy",
    "d_gy_año",
    "año_base",
    "fecha_dc"
    ]]

#Función para filtrar dataframe
def filter_dataframe(df_renombrado: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df_renombrado (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    #modify = st.checkbox("Add filters")

    #if not modify:
    #    return df_renombrado

    df_filter = df_renombrado.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df_filter.columns:
        if is_object_dtype(df_filter[col]):
            try:
                df_filter[col] = pd.to_datetime(df_filter[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df_filter[col]):
            df_filter[col] = df_filter[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filtrar Base de Datos según", df_filter.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            if is_categorical_dtype(df_filter[column]) or df_filter[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df_filter[column].unique(),
                    default=list(df_filter[column].unique()),
                )
                df_filter = df_filter[df_filter[column].isin(user_cat_input)]
            elif is_numeric_dtype(df_filter[column]):
                _min = float(df_filter[column].min())
                _max = float(df_filter[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df_filter = df_filter[df_filter[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df_filter[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df_filter[column].min(),
                        df_filter[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df_filter = df_filter.loc[df_filter[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df_filter = df_filter[df_filter[column].astype(str).str.contains(user_text_input)]

    return df_filter

# Creación de la interfaz de usuario con Streamlit
def main():
    with st.container():
        st.title('Visualización de Fechados Arqueológicos')
        st.write("El Proyecto ArqueoCronoChile (ACC) tiene como objetivo...")
    with st.container():
        st.title('Base de datos de Fechados Completa:')
       
        # Visualizacion
        df_renombrado = df_neworder.rename(columns={
            'cod_lab': 'Código Laboratorio',
            'tipo_fechado': 'Tipo de Fechado',
            'fecha_ap': 'Fecha AP',
            'error_ap': 'Error AP',
            'sitio': 'Sitio',
            'año_realizacion': 'Año de Realización',
            'instrumento': 'Instrumento',
            'tipo_muestra': 'Tipo de Muestra',
            'detalle_muestra': 'Detalle de Muestra',
            'especimen': 'Especimen',
            'unidad': 'Unidad de Excavación',
            'nivel': 'Nivel de Excavación',
            'estrato': 'Estrato',
            'profundidad': 'Profundidad (cm)',
            'proyecto': 'Proyecto',
            'responsable': 'Responsable(s)',
            'coleccion': 'Colección',
            'información_contextual': 'Información Contextual',
            'publicaciones': 'Publicaciones',
            'compilacion_fechados': "Compilación de Fechado",
            'reporte': 'Reporte',
            'localidad': 'Localidad',
            'comuna': 'Comuna',
            'provincia': 'Provincia',
            'region': 'Región',
            'latitud_4326': 'Latitud WGS84',
            'longitud_4326': 'Longitud WGS84',
            'este_32718': 'Este WGS84 UTM 18S',
            'norte_32718': 'Norte WGS84 UTM 18S',
            'este_32719': 'Este WGS84 UTM 19S',
            'norte_32719': 'Norte WGS84 UTM 19S',
            'biomolecula': 'Biomolécula',
            'c13_c14': 'c13/c14',
            'cn': 'C/N',
            'collagen_yield': 'Collagen Yield',
            'd13': 'D13',
            'd14': 'D14',
            'p_gy': 'P (Gy)',
            'error_p_gy': 'Error P (Gy)',
            'd_gy_año': 'D (Gy/año)',
            'año_base': 'Año Base',
            'fecha_dc': 'Fecha d.C.',
            })
        st.dataframe(df_renombrado)

        st.divider()

        st.title('Filtrar Base de Datos')
        st.dataframe(filter_dataframe(df_renombrado))


    st.divider()

    with st.container():
        st.title('Mapa de Fechados')
        #Mapa
        map = folium.Map(location=[-34.941444, -71.341944], zoom_start=4)
        location = df[['lat', 'lon']]
        location_list = location.to_dict('records')
        filtered_location_list = [item for item in location_list if not (item['lat'] == float('inf') or item['lon'] == float('inf'))]
        #Usando Folium
        #for ubicacion in filtered_location_list:
        #   punto = ubicacion['lat'], ubicacion['lon']
        #   folium.Marker(punto).add_to(map)
        #st_folium(map)
        st.map(filtered_location_list)
    
    st.divider()

    with st.container():
        #Gráficos
        st.title("Análisis de básico de Base de Datos")
        columns = df_neworder.columns.tolist()
        s = df_neworder["tipo_fechado"].str.strip().value_counts()
        
        with st.container():
            st.write("---")
            col1, col2 = st.columns(2)
            with col1:
                st.title("Gráficos")
                #Barra de tipos de fechado
                trace = go.Bar(x=s.index, y=s.values)
                layout = go.Layout(title="Gráfico de Barras de  Tipos de Fechado")
                data = [trace]
                fig = go.Figure(data=data, layout=layout)
                st.plotly_chart(fig)
            
            with col2:
                #Indice de completitud
                st.title("Índices")
                # Contar celdas NaN por columna
                nan_count_col = df.isna().sum().sum()
                # Contar celdas no NaN (llenas) por columna
                non_nan_count_col = df.notna().sum().sum()
                index_completitud = round((non_nan_count_col * 100) / (non_nan_count_col + nan_count_col), 2) 
                st.metric(label='Completitud General', value= str(index_completitud) + "%")
                completitud = pd.DataFrame({
                    'Estado de Celda':['Vacía', 'Completa'], 
                    'Total': [nan_count_col, non_nan_count_col]})
                st.write(completitud)

            
        


if __name__ == '__main__':
    main()