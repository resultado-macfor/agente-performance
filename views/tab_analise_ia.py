import pandas as pd
import streamlit as st
from datetime import datetime

from utils.helpers import safe_metric
from services.gemini_service import generate_gemini_analysis, generate_slides_description


def render_tab_analise_ia(df_filtrado, modelo_texto):
    st.header("🤖 Análise com Gemini IA")

    if not modelo_texto:
        st.error("❌ Gemini não configurado!")
        return

    if df_filtrado.empty:
        st.warning("📭 Nenhum dado carregado.")
        return

    st.markdown("### 🔍 Filtros para Análise")

    with st.expander("⚙️ Configurar", expanded=True):
        col_filter1, col_filter2 = st.columns(2)

        with col_filter1:
            if 'datasource' in df_filtrado.columns:
                datasources = sorted(df_filtrado['datasource'].dropna().unique())
                selected_ds = st.multiselect(
                    "Data Sources:",
                    options=datasources,
                    default=datasources[:min(3, len(datasources))],
                    key="selected_ds_tab6"
                )
            else:
                selected_ds = None

            date_range = None
            if 'date' in df_filtrado.columns:
                try:
                    date_series = df_filtrado['date'].dropna()
                    if len(date_series) > 0:
                        if not pd.api.types.is_datetime64_any_dtype(date_series):
                            date_series = pd.to_datetime(date_series, errors='coerce')
                        min_date = date_series.min().date()
                        max_date = date_series.max().date()
                        date_range = st.date_input(
                            "Período:",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key="date_range_tab6"
                        )
                except:
                    pass

        with col_filter2:
            selected_campaigns = None
            if 'campaign' in df_filtrado.columns:
                campaigns = sorted(df_filtrado['campaign'].dropna().unique())
                selected_campaigns = st.multiselect(
                    "Campanhas (opcional):",
                    options=campaigns,
                    key="selected_campaigns_tab6"
                )

            df_len = len(df_filtrado)
            max_records_value = min(10000, max(100, df_len)) if df_len > 0 else 5000
            max_records = st.slider(
                "Máximo de registros:",
                min_value=100,
                max_value=min(10000, df_len) if df_len > 0 else 10000,
                value=min(5000, max_records_value),
                step=100,
                key="max_records_tab6"
            )

    df_filtered_ia = df_filtrado.copy()

    if selected_ds and 'datasource' in df_filtered_ia.columns:
        df_filtered_ia = df_filtered_ia[df_filtered_ia['datasource'].isin(selected_ds)]

    if date_range and len(date_range) == 2 and 'date' in df_filtered_ia.columns:
        start_date, end_date = date_range
        if not pd.api.types.is_datetime64_any_dtype(df_filtered_ia['date']):
            df_filtered_ia['date'] = pd.to_datetime(df_filtered_ia['date'], errors='coerce')
        mask = df_filtered_ia['date'].notna()
        df_filtered_ia = df_filtered_ia[
            mask &
            (df_filtered_ia['date'] >= pd.Timestamp(start_date)) &
            (df_filtered_ia['date'] <= pd.Timestamp(end_date))
        ]

    if selected_campaigns and 'campaign' in df_filtered_ia.columns:
        df_filtered_ia = df_filtered_ia[df_filtered_ia['campaign'].isin(selected_campaigns)]

    df_filtered_ia = df_filtered_ia.head(max_records)

    st.markdown("### 📊 Dados Selecionados")

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

    with col_stat1:
        safe_metric("Registros", len(df_filtered_ia))
    with col_stat2:
        if 'campaign' in df_filtered_ia.columns:
            try:
                safe_metric("Campanhas", df_filtered_ia['campaign'].nunique())
            except:
                safe_metric("Campanhas", "Erro")
    with col_stat3:
        if 'datasource' in df_filtered_ia.columns:
            try:
                safe_metric("Data Sources", df_filtered_ia['datasource'].nunique())
            except:
                safe_metric("Data Sources", "Erro")
    with col_stat4:
        if 'date' in df_filtered_ia.columns:
            try:
                ds = df_filtered_ia['date'].dropna()
                if len(ds) > 0:
                    if not pd.api.types.is_datetime64_any_dtype(ds):
                        ds = pd.to_datetime(ds, errors='coerce')
                    safe_metric("Dias", (ds.max() - ds.min()).days + 1)
                else:
                    safe_metric("Dias", 0)
            except:
                safe_metric("Dias", "Erro")

    st.markdown("### 🎯 Configuração")

    analysis_focus = st.selectbox(
        "Foco da Análise:",
        options=["overall", "trends", "efficiency", "complete"],
        format_func=lambda x: {
            "overall": "📈 Performance Geral",
            "trends": "📊 Tendências",
            "efficiency": "💰 Eficiência",
            "complete": "🏆 Análise Completa"
        }[x],
        key="analysis_focus_tab6"
    )

    user_instructions = st.text_area(
        "📝 Instruções (opcional):",
        placeholder="Ex: Foque no ROI, identifique as melhores campanhas, analise tendências por data source...",
        height=100,
        key="user_instructions_tab6"
    )

    st.markdown("### 🚀 Gerar Análise")

    if st.button("🤖 Gerar Análise com Gemini", type="primary", use_container_width=True, key="generate_button_tab6"):
        if df_filtered_ia.empty:
            st.error("❌ Nenhum dado após filtros.")
        else:
            with st.spinner(f"🤖 Analisando {len(df_filtered_ia):,} registros..."):
                try:
                    result = generate_gemini_analysis(modelo_texto, df_filtered_ia, analysis_focus, user_instructions)
                    st.session_state.gemini_analysis = result
                    st.success("✅ Análise concluída!")
                except Exception as e:
                    st.error(f"❌ Erro ao gerar análise: {str(e)[:200]}")

    if st.session_state.gemini_analysis:
        st.markdown("---")
        st.markdown("### 📄 Relatório de Análise")

        col_actions1, col_actions2, col_actions3 = st.columns(3)

        with col_actions1:
            st.download_button(
                label="💾 Baixar Relatório",
                data=st.session_state.gemini_analysis,
                file_name=f"analise_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_report_tab6"
            )

        with col_actions2:
            if st.button("🎬 Gerar Descrição dos Slides", use_container_width=True, type="secondary", key="generate_slides_tab6"):
                with st.spinner("Gerando descrição para slides..."):
                    slides_desc = generate_slides_description(modelo_texto, st.session_state.gemini_analysis, user_instructions)
                    st.session_state.slides_description = slides_desc
                    st.success("✅ Descrição dos slides gerada!")
                    st.rerun()

        with col_actions3:
            if st.button("🔄 Nova Análise", use_container_width=True, key="new_analysis_tab6"):
                st.session_state.gemini_analysis = None
                st.session_state.slides_description = None
                st.rerun()

        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
        st.markdown(st.session_state.gemini_analysis)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.slides_description:
            st.markdown("---")
            st.markdown("### 🎬 Descrição para Slides")
            st.download_button(
                label="📥 Baixar Descrição dos Slides",
                data=st.session_state.slides_description,
                file_name=f"slides_descricao_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="download_slides_tab6"
            )
            st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
            st.markdown(st.session_state.slides_description)
            st.markdown('</div>', unsafe_allow_html=True)
