import pandas as pd
import streamlit as st

from utils.helpers import safe_metric, identificar_colunas_numericas, analisar_coluna


def render_tab_visao_geral(df_filtrado):
    st.header("📋 Visão Geral das Colunas")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        safe_metric("Total de Colunas", len(df_filtrado.columns))

    with col2:
        col_numericas_filtradas = identificar_colunas_numericas(df_filtrado)
        safe_metric("Colunas Numéricas", len(col_numericas_filtradas))

    with col3:
        safe_metric("Total de Registros", len(df_filtrado))

    with col4:
        try:
            memoria_mb = df_filtrado.memory_usage(deep=True).sum() / 1024**2
            safe_metric("Uso de Memória", f"{memoria_mb:.2f} MB")
        except:
            safe_metric("Uso de Memória", "N/A")

    st.subheader("📊 Detalhes de Cada Coluna")

    col_filtro1, col_filtro2 = st.columns(2)

    with col_filtro1:
        tipo_filtro = st.selectbox(
            "Filtrar por tipo",
            ["Todas", "Numéricas", "Texto", "Datas"],
            key="filtro_tipo_tab1"
        )

    with col_filtro2:
        pesquisa_coluna = st.text_input("🔍 Pesquisar coluna", "", key="pesquisa_coluna_tab1")

    colunas_para_mostrar = []

    for col in df_filtrado.columns:
        incluir = True

        if tipo_filtro == "Numéricas":
            incluir = col in col_numericas_filtradas
        elif tipo_filtro == "Texto":
            incluir = df_filtrado[col].dtype == 'object' and col not in col_numericas_filtradas
        elif tipo_filtro == "Datas":
            incluir = pd.api.types.is_datetime64_any_dtype(df_filtrado[col])

        if pesquisa_coluna and pesquisa_coluna.lower() not in col.lower():
            incluir = False

        if incluir:
            colunas_para_mostrar.append(col)

    for col in sorted(colunas_para_mostrar)[:50]:
        analise = analisar_coluna(df_filtrado, col)

        if analise:
            with st.expander(f"**{col}** ({analise['tipo_detalhado'] if 'tipo_detalhado' in analise else analise['tipo']})"):
                col_info1, col_info2 = st.columns(2)

                with col_info1:
                    safe_metric("Tipo", analise['tipo'])
                    safe_metric("Não nulos", analise['nao_nulos'])
                    safe_metric("Valores únicos", analise['valores_unicos'])

                with col_info2:
                    safe_metric("Nulos", analise['nulos'])
                    safe_metric("% Nulos", f"{analise['percentual_nulos']:.1f}%")

                if analise.get('tipo_detalhado') == 'Numérica' and analise['nao_nulos'] > 0:
                    st.subheader("📈 Estatísticas")
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

                    with col_stats1:
                        safe_metric("Média", analise.get('media', 0))
                        safe_metric("Min", analise.get('min', 0))

                    with col_stats2:
                        safe_metric("Mediana", analise.get('mediana', 0))
                        safe_metric("Max", analise.get('max', 0))

                    with col_stats3:
                        safe_metric("Q1 (25%)", analise.get('q1', 0))

                    with col_stats4:
                        safe_metric("Q3 (75%)", analise.get('q3', 0))
