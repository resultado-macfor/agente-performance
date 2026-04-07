import pandas as pd
import streamlit as st
from datetime import datetime

from utils.helpers import identificar_colunas_numericas


def render_tab_visualizar_dados(df_filtrado):
    st.header("📊 Visualizar Dados Completos")

    colunas_vis = st.multiselect(
        "Selecione colunas para visualizar",
        options=sorted(df_filtrado.columns),
        default=sorted(df_filtrado.columns)[:min(10, len(df_filtrado.columns))],
        key="colunas_vis_tab4"
    )

    if not colunas_vis:
        return

    st.subheader("🔍 Filtros Adicionais")

    col_f1, col_f2, col_f3 = st.columns(3)
    df_filtrado_tab4 = df_filtrado.copy()

    with col_f1:
        if 'datasource' in df_filtrado.columns:
            datasources = sorted(df_filtrado['datasource'].dropna().unique())
            ds_selecionados = st.multiselect(
                "Data Sources",
                options=datasources,
                default=datasources[:min(3, len(datasources))],
                key="ds_selecionados_tab4"
            )
            if ds_selecionados:
                df_filtrado_tab4 = df_filtrado_tab4[df_filtrado_tab4['datasource'].isin(ds_selecionados)]

    with col_f2:
        colunas_num_vis = [c for c in colunas_vis if c in identificar_colunas_numericas(df_filtrado)]
        if colunas_num_vis:
            col_filtro = st.selectbox(
                "Filtrar por coluna numérica",
                options=['Nenhum'] + colunas_num_vis,
                key="col_filtro_tab4"
            )
            if col_filtro != 'Nenhum':
                try:
                    col_data = pd.to_numeric(df_filtrado_tab4[col_filtro], errors='coerce').dropna()
                    if len(col_data) > 0:
                        min_val = st.number_input(
                            f"Valor mínimo de {col_filtro}",
                            value=float(col_data.min()),
                            key=f"min_val_{col_filtro}_tab4"
                        )
                        df_filtrado_tab4 = df_filtrado_tab4[
                            pd.to_numeric(df_filtrado_tab4[col_filtro], errors='coerce') >= min_val
                        ]
                except:
                    st.warning(f"Não foi possível filtrar por {col_filtro}")

    with col_f3:
        limite_linhas = st.slider("Linhas para mostrar", 10, 1000, 100, key="limite_linhas_tab4")

    st.subheader(f"📋 Dados ({len(df_filtrado_tab4):,} registros)")

    if len(df_filtrado_tab4) > 0:
        total_pages = max(1, len(df_filtrado_tab4) // limite_linhas + 1)

        col_pg1, _, col_pg3 = st.columns([1, 2, 1])

        with col_pg1:
            page_number = st.number_input(
                "Página",
                min_value=1,
                max_value=total_pages,
                value=1,
                key="page_number_tab4"
            )

        with col_pg3:
            st.caption(f"Total: {len(df_filtrado_tab4):,} registros")

        start_idx = (page_number - 1) * limite_linhas
        end_idx = min(start_idx + limite_linhas, len(df_filtrado_tab4))

        df_display = df_filtrado_tab4[colunas_vis].iloc[start_idx:end_idx].copy()

        for col in colunas_vis:
            if col in identificar_colunas_numericas(df_filtrado):
                try:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else ""
                    )
                except:
                    pass
            elif pd.api.types.is_datetime64_any_dtype(df_filtrado[col]):
                try:
                    df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
                except:
                    pass

        st.dataframe(df_display, use_container_width=True, height=400)
    else:
        st.info("Nenhum dado após filtros")

    st.subheader("📥 Exportar")

    if len(df_filtrado_tab4) > 0:
        csv = df_filtrado_tab4[colunas_vis].to_csv(index=False)
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name=f"dados_filtrados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="download_csv_tab4"
        )
    else:
        st.warning("Nenhum dado para exportar")
