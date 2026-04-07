import pandas as pd
import streamlit as st
import plotly.express as px

from utils.helpers import safe_metric, identificar_colunas_numericas


def render_tab_performance(df_filtrado):
    st.header("🎯 Análise de Performance")

    if 'campaign' not in df_filtrado.columns:
        st.error("❌ Coluna 'campaign' não encontrada.")
        return

    st.subheader("📊 Métricas Gerais")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        try:
            safe_metric("Campanhas", df_filtrado['campaign'].nunique())
        except:
            safe_metric("Campanhas", "Erro")

    with col2:
        if 'date' in df_filtrado.columns:
            try:
                df_date = df_filtrado['date'].dropna()
                if len(df_date) > 0:
                    if not pd.api.types.is_datetime64_any_dtype(df_date):
                        df_date = pd.to_datetime(df_date, errors='coerce')
                    days = (df_date.max() - df_date.min()).days + 1
                    safe_metric("Dias", days)
                else:
                    safe_metric("Dias", 0)
            except:
                safe_metric("Dias", "Erro")

    with col3:
        if 'datasource' in df_filtrado.columns:
            try:
                safe_metric("Data Sources", df_filtrado['datasource'].nunique())
            except:
                safe_metric("Data Sources", "Erro")

    with col4:
        try:
            n = df_filtrado['campaign'].nunique()
            safe_metric("Média Reg/Camp", f"{len(df_filtrado) / n:.1f}" if n > 0 else "0")
        except:
            safe_metric("Média Reg/Camp", "Erro")

    st.subheader("📈 Top Campanhas")

    try:
        campaign_stats = df_filtrado['campaign'].value_counts().head(10)
        fig = px.bar(
            x=campaign_stats.index.astype(str),
            y=campaign_stats.values,
            title="Top 10 Campanhas por Volume",
            labels={'x': 'Campanha', 'y': 'Registros'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao criar gráfico: {str(e)[:100]}")

    st.subheader("💰 Métricas Financeiras")

    financial_metrics = []
    col_numericas_filtradas = identificar_colunas_numericas(df_filtrado)
    for metric in ['spend', 'revenue', 'conversions', 'roas', 'cpc']:
        for col in col_numericas_filtradas:
            if metric in col.lower():
                financial_metrics.append(col)
                break

    if financial_metrics:
        cols = st.columns(min(4, len(financial_metrics)))
        for idx, metric in enumerate(financial_metrics[:4]):
            with cols[idx]:
                if metric in df_filtrado.columns:
                    try:
                        total = pd.to_numeric(df_filtrado[metric], errors='coerce').sum()
                        safe_metric(metric, total)
                    except:
                        safe_metric(metric, "Erro")

    st.subheader("📊 Análise por Categoria")

    col_cat1, col_cat2 = st.columns(2)

    with col_cat1:
        if 'campaign_cliente' in df_filtrado.columns:
            try:
                cliente_stats = df_filtrado['campaign_cliente'].value_counts().head(10)
                fig_cliente = px.pie(
                    values=cliente_stats.values,
                    names=cliente_stats.index,
                    title="Distribuição por Cliente",
                    hole=0.3
                )
                st.plotly_chart(fig_cliente, use_container_width=True)
            except:
                pass

    with col_cat2:
        if 'campaign_tipo_campanha' in df_filtrado.columns:
            try:
                tipo_stats = df_filtrado['campaign_tipo_campanha'].value_counts().head(10)
                fig_tipo = px.bar(
                    x=tipo_stats.index,
                    y=tipo_stats.values,
                    title="Distribuição por Tipo de Campanha"
                )
                fig_tipo.update_xaxes(tickangle=45)
                st.plotly_chart(fig_tipo, use_container_width=True)
            except:
                pass
