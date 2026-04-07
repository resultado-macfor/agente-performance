import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.helpers import safe_metric, format_currency
from agent.campaign_classifier import calculate_mom_analysis, create_mom_table


def render_tab_mom(df, modelo_texto):
    st.markdown('<div class="mom-analysis"><h2>📅 Análise MoM (Month-over-Month)</h2></div>', unsafe_allow_html=True)

    st.markdown("### 🎯 Configuração da Análise")

    col_config1, col_config2, col_config3 = st.columns(3)

    with col_config1:
        clientes_disponiveis = (
            ["Todos"] + sorted(df['cliente_identificado'].unique().tolist())
            if 'cliente_identificado' in df.columns else ["Todos"]
        )
        cliente_analise = st.selectbox(
            "👥 Cliente para análise:",
            options=clientes_disponiveis,
            index=0,
            key="cliente_analise_tab8"
        )

    mes_atual_period = None

    with col_config2:
        if 'date' in df.columns:
            df_dates = df['date'].dropna()
            if len(df_dates) > 0:
                if not pd.api.types.is_datetime64_any_dtype(df_dates):
                    df_dates = pd.to_datetime(df_dates, errors='coerce')
                meses_str = [str(m) for m in sorted(df_dates.dt.to_period('M').unique(), reverse=True)]
                mes_atual_period = st.selectbox(
                    "📅 Mês Atual:",
                    options=meses_str[:12],
                    key="mes_atual_tab8"
                )
            else:
                st.info("Sem datas disponíveis")
        else:
            st.info("Coluna de data não encontrada")

    with col_config3:
        mes_anterior_period = None
        if mes_atual_period:
            mes_anterior_period = pd.Period(
                pd.Period(mes_atual_period).to_timestamp() - pd.DateOffset(months=1),
                freq='M'
            )
            st.write(f"**Mês Anterior:** {mes_anterior_period}")

    st.markdown("### 📊 Executar Análise")

    if st.button("📈 Calcular Análise MoM", use_container_width=True, type="primary", key="calcular_mom_tab8"):
        if mes_atual_period and mes_anterior_period:
            with st.spinner(f"Calculando análise MoM para {cliente_analise}..."):
                try:
                    mom_result = calculate_mom_analysis(
                        df, cliente_analise,
                        pd.Period(mes_atual_period),
                        pd.Period(mes_anterior_period)
                    )
                    if mom_result:
                        st.session_state.mom_analysis = mom_result
                        st.success("✅ Análise MoM calculada!")
                    else:
                        st.error("❌ Não foi possível calcular a análise MoM")
                except Exception as e:
                    st.error(f"❌ Erro ao calcular MoM: {str(e)[:200]}")
        else:
            st.error("❌ Selecione o mês atual para análise")

    if not st.session_state.mom_analysis:
        return

    mom_data = st.session_state.mom_analysis

    st.markdown("### 📋 Resultados da Análise")

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)

    with col_res1:
        safe_metric("Cliente", mom_data['cliente'])
    with col_res2:
        safe_metric("Mês Anterior", mom_data['mes_anterior'])
    with col_res3:
        safe_metric("Mês Atual", mom_data['mes_atual'])
    with col_res4:
        total_change = mom_data['total_mes_atual'] - mom_data['total_mes_anterior']
        change_pct = (total_change / mom_data['total_mes_anterior'] * 100) if mom_data['total_mes_anterior'] > 0 else 0
        safe_metric("Variação Registros", total_change, f"{change_pct:.1f}%")

    st.markdown("### 📊 Análise por Plataforma")

    df_mom_table = create_mom_table(mom_data)

    if df_mom_table is not None:
        st.dataframe(df_mom_table.style.format(precision=2), use_container_width=True)

        if mom_data.get('platform_analysis'):
            df_platforms = pd.DataFrame({
                'Plataforma': list(mom_data['platform_analysis'].keys()),
                'Mês Anterior': [d['spend_previous'] for d in mom_data['platform_analysis'].values()],
                'Mês Atual': [d['spend_current'] for d in mom_data['platform_analysis'].values()],
            })

            fig_platforms = go.Figure()
            fig_platforms.add_trace(go.Bar(name='Mês Anterior', x=df_platforms['Plataforma'], y=df_platforms['Mês Anterior'], marker_color='#6366f1'))
            fig_platforms.add_trace(go.Bar(name='Mês Atual', x=df_platforms['Plataforma'], y=df_platforms['Mês Atual'], marker_color='#10b981'))
            fig_platforms.update_layout(
                title="Investimento por Plataforma - Comparativo MoM",
                barmode='group', xaxis_title="Plataforma", yaxis_title="Investimento (R$)", height=500
            )
            st.plotly_chart(fig_platforms, use_container_width=True)

    st.markdown("### 📈 Análise de Métricas")

    metric_data = mom_data.get('metric_analysis', {})
    if metric_data:
        cols_metrics = st.columns(min(3, len(metric_data)))

        for idx, (metric_name, metric_info) in enumerate(metric_data.items()):
            if idx >= 9:
                break
            with cols_metrics[idx % 3]:
                st.markdown('<div class="yoy-metric">', unsafe_allow_html=True)
                st.subheader(metric_name)

                is_money = any(k in metric_name.lower() for k in ('spend', 'cost', 'revenue'))
                col_curr, col_prev = st.columns(2)
                with col_curr:
                    st.metric("Atual", format_currency(metric_info['current']) if is_money else f"{metric_info['current']:,.0f}")
                with col_prev:
                    st.metric("Anterior", format_currency(metric_info['previous']) if is_money else f"{metric_info['previous']:,.0f}")

                change_color = "green" if metric_info['change'] > 0 else "red"
                change_val = format_currency(metric_info['change']) if is_money else f"{metric_info['change']:,.0f}"
                st.markdown(
                    f"**Variação:** <span style='color:{change_color}'>{change_val} ({metric_info['change_pct']:.1f}%)</span>",
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📝 Relatório de Análise")

    if modelo_texto:
        if st.button("📄 Gerar Relatório com Gemini", use_container_width=True, key="gerar_relatorio_mom_tab8"):
            with st.spinner("Gerando relatório de análise MoM..."):
                try:
                    analysis_text = f"""
                    CLIENTE: {mom_data['cliente']}
                    PERÍODO: {mom_data['mes_anterior']} vs {mom_data['mes_atual']}

                    RESUMO GERAL:
                    - Total de registros mês anterior: {mom_data['total_mes_anterior']:,}
                    - Total de registros mês atual: {mom_data['total_mes_atual']:,}
                    - Variação: {mom_data['total_mes_atual'] - mom_data['total_mes_anterior']:,} ({((mom_data['total_mes_atual'] - mom_data['total_mes_anterior']) / mom_data['total_mes_anterior'] * 100) if mom_data['total_mes_anterior'] else 0:.1f}%)

                    ANÁLISE POR PLATAFORMA:
                    """
                    for platform, data in mom_data.get('platform_analysis', {}).items():
                        analysis_text += f"""
                        - {platform}:
                          * Investimento anterior: R$ {data['spend_previous']:,.2f}
                          * Investimento atual: R$ {data['spend_current']:,.2f}
                          * Variação: R$ {data['spend_change']:,.2f} ({data['spend_change_pct']:.1f}%)
                          * Registros anterior: {data['records_previous']:,}
                          * Registros atual: {data['records_current']:,}
                        """

                    analysis_text += "\n\nANÁLISE DE MÉTRICAS:\n"
                    for metric, data in mom_data.get('metric_analysis', {}).items():
                        analysis_text += f"""
                        - {metric}:
                          * Anterior: {data['previous']:,.2f}
                          * Atual: {data['current']:,.2f}
                          * Variação: {data['change']:,.2f} ({data['change_pct']:.1f}%)
                        """

                    prompt = f"""
                    # 📊 RELATÓRIO DE ANÁLISE MoM (Month-over-Month)

                    {analysis_text}

                    ## 🎯 TAREFA:

                    Com base nos dados MoM acima, crie um relatório executivo em português que inclua:

                    1. **📈 RESUMO EXECUTIVO** (1-2 parágrafos com os principais achados)
                    2. **💰 ANÁLISE DE INVESTIMENTO** (comparativo por plataforma, eficiência)
                    3. **📊 ANÁLISE DE PERFORMANCE** (principais métricas e suas variações)
                    4. **🔍 INSIGHTS ESTRATÉGICOS** (3-5 insights baseados nos dados)
                    5. **🎯 RECOMENDAÇÕES** (5-7 recomendações acionáveis para o próximo mês)
                    6. **📅 PRÓXIMOS PASSOS** (plano de ação sugerido)

                    Foque em:
                    - Evolução da redistribuição estratégica de investimento por plataforma
                    - Estratégia eficaz com menor investimento (se aplicável)
                    - Eficiência e diminuição nos custos das campanhas de mídia paga
                    - Performance geral e engajamento

                    Seja específico com os números e forneça interpretações práticas.
                    """

                    response = modelo_texto.generate_content(prompt)
                    st.markdown("### 📄 Relatório Gemini")
                    st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
                    st.markdown(response.text)
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Erro ao gerar relatório: {str(e)[:200]}")
