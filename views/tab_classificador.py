import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

from utils.helpers import safe_metric
from agent.campaign_classifier import classificar_campanhas_multi_cliente


def render_tab_classificador(df, df_classificado, modelo_texto):
    st.markdown('<div class="campaign-classifier"><h2>🎪 Classificador de Campanhas Multi-Clientes</h2></div>', unsafe_allow_html=True)

    col_intro1, col_intro2 = st.columns(2)

    with col_intro1:
        st.markdown("### 📋 Sobre o Sistema")

    with col_intro2:
        st.markdown("### 🔍 Status da Classificação")

        if 'classificado' in df_classificado.columns:
            total = len(df_classificado)
            classificadas = df_classificado[df_classificado['classificado'] == 'SIM'].shape[0]
            taxa = (classificadas / total * 100) if total > 0 else 0

            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                safe_metric("Total", total)
                safe_metric("Classificadas", classificadas)
            with col_stat2:
                safe_metric("Taxa", f"{taxa:.1f}%")
                if 'campaign_cliente' in df_classificado.columns:
                    safe_metric("Clientes", df_classificado['campaign_cliente'].nunique())
        else:
            st.warning("Nenhuma classificação disponível")

    st.markdown("### 📊 Distribuição por Categoria")

    col_dist1, col_dist2, col_dist3 = st.columns(3)

    with col_dist1:
        if 'campaign_cliente' in df_classificado.columns:
            try:
                cliente_counts = df_classificado['campaign_cliente'].value_counts().head(10)
                fig = px.bar(
                    x=cliente_counts.index, y=cliente_counts.values,
                    title="Top 10 Clientes", color=cliente_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

    with col_dist2:
        if 'campaign_tipo_campanha' in df_classificado.columns:
            try:
                tipo_counts = df_classificado['campaign_tipo_campanha'].value_counts().head(10)
                fig = px.pie(
                    values=tipo_counts.values, names=tipo_counts.index,
                    title="Tipos de Campanha", hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

    with col_dist3:
        if 'campaign_etapa_funil' in df_classificado.columns:
            try:
                etapa_counts = df_classificado['campaign_etapa_funil'].value_counts()
                fig = px.bar(
                    x=etapa_counts.index, y=etapa_counts.values,
                    title="Etapas do Funil", color=etapa_counts.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            except:
                pass

    st.markdown("### 🔍 Explorador de Campanhas")

    col_explorer1, col_explorer2 = st.columns(2)

    with col_explorer1:
        if 'campaign' in df_classificado.columns:
            campanhas = sorted(df_classificado['campaign'].dropna().unique())
            campanha_selecionada = st.selectbox(
                "Selecione uma campanha para análise:",
                options=campanhas[:100],
                key="campanha_selecionada_tab7"
            )

            if campanha_selecionada:
                campanha_data = df_classificado[df_classificado['campaign'] == campanha_selecionada].iloc[0]
                st.markdown("#### 📋 Detalhes da Campanha")
                st.write(f"**Nome:** {campanha_selecionada}")

                categorias_identificadas = {
                    col.replace('campaign_', '').replace('_', ' ').title(): campanha_data[col]
                    for col in df_classificado.columns
                    if col.startswith('campaign_') and col not in ('campaign_classificado', 'categorias_identificadas')
                    and pd.notna(campanha_data[col])
                }

                if categorias_identificadas:
                    st.markdown("#### 🏷️ Categorias Identificadas")
                    for categoria, valor in categorias_identificadas.items():
                        st.write(f"**{categoria}:** {valor}")
                else:
                    st.info("Nenhuma categoria identificada para esta campanha")

    with col_explorer2:
        st.markdown("#### 📈 Estatísticas de Classificação")

        if 'classificado' in df_classificado.columns:
            status_counts = df_classificado['classificado'].value_counts()
            fig_status = px.pie(
                values=status_counts.values, names=status_counts.index,
                title="Status de Classificação",
                color_discrete_sequence=['#10b981', '#ef4444']
            )
            st.plotly_chart(fig_status, use_container_width=True)

        if st.button("🔄 Reclassificar Campanhas", use_container_width=True, key="reclassificar_tab7"):
            with st.spinner("Reclassificando campanhas..."):
                st.session_state.df_classificado = classificar_campanhas_multi_cliente(df)
                st.success("✅ Campanhas reclassificadas!")
                st.rerun()

    st.markdown("### 📥 Exportar Dados Classificados")

    if len(df_classificado) > 0:
        colunas_classificadas = [col for col in df_classificado.columns if col.startswith('campaign_')]
        colunas_base = ['campaign', 'date', 'datasource'] if all(
            col in df_classificado.columns for col in ['campaign', 'date', 'datasource']
        ) else []
        colunas_exportar = colunas_base + colunas_classificadas

        col_export1, col_export2 = st.columns(2)

        with col_export1:
            st.download_button(
                label="📥 Baixar Todos os Dados Classificados",
                data=df_classificado[colunas_exportar].to_csv(index=False),
                file_name=f"campanhas_classificadas_completo_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_all_tab7"
            )

        with col_export2:
            if 'classificado' in df_classificado.columns:
                nao_classificadas = df_classificado[df_classificado['classificado'] == 'NÃO']
                if len(nao_classificadas) > 0:
                    st.download_button(
                        label="📥 Baixar Campanhas Não Classificadas",
                        data=nao_classificadas[['campaign']].to_csv(index=False),
                        file_name=f"campanhas_nao_classificadas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_unclassified_tab7"
                    )

    if modelo_texto and len(df_classificado) > 0:
        st.markdown("### 🤖 Análise de Padrões com Gemini")

        if st.button("🔍 Analisar Padrões de Nomenclatura", use_container_width=True, key="analisar_padroes_tab7"):
            with st.spinner("Analisando padrões de nomenclatura..."):
                try:
                    sample_size = min(50, len(df_classificado))
                    sample_campaigns = df_classificado['campaign'].dropna().sample(sample_size).tolist()

                    clientes_identificados = []
                    if 'campaign_cliente' in df_classificado.columns:
                        clientes_identificados = df_classificado['campaign_cliente'].dropna().unique().tolist()

                    prompt = f"""
                    Analise os seguintes nomes de campanhas de marketing e identifique:

                    1. Padrões comuns de nomenclatura
                    2. Estruturas mais frequentes
                    3. Componentes principais encontrados
                    4. Clientes identificados: {', '.join(clientes_identificados[:10]) if clientes_identificados else 'Nenhum'}
                    5. Problemas de padronização
                    6. Sugestões para melhorar a classificação automática

                    Amostra de nomes de campanhas:
                    {', '.join([str(c) for c in sample_campaigns])}

                    Forneça uma análise detalhada em português com:
                    - Identificação de padrões estruturais
                    - Componentes mais comuns (cliente, produto, objetivo, etc.)
                    - Problemas de inconsistência
                    - Recomendações para padronização futura
                    - Sugestões para melhorar a taxonomia
                    """

                    response = modelo_texto.generate_content(prompt)
                    st.markdown("### 📄 Análise de Padrões")
                    st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
                    st.markdown(response.text)
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Erro na análise: {str(e)[:200]}")
