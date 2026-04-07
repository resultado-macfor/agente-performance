import streamlit as st

from services.gemini_service import generate_yoy_analysis


def render_tab_yoy(modelo_texto):
    st.markdown('<div class="yoy-scenario"><h2>📊 Cenário YoY (Year-over-Year)</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cliente_yoy = st.selectbox(
            "👥 Cliente:",
            options=["Syngenta", "Golden Harvest Brasil", "Nidera", "NK Seeds", "EuroChem", "Grupo Vittia", "Outros"],
            key="cliente_yoy_tab9"
        )
    with col2:
        periodo_ref = st.text_input(
            "📅 Período de referência:",
            placeholder="Ex: Novembro 2025 vs Novembro 2024",
            key="periodo_ref_tab9"
        )

    st.markdown("### 📋 Cole os dados do briefing")
    st.caption("Cole qualquer formato — tabela, texto, números soltos. O Gemini extrai o que encontrar.")

    dados_brutos = st.text_area(
        label="Dados:",
        height=300,
        placeholder=(
            "Ex:\n\n"
            "Investimento nov/25: FB R$60.8k, TikTok R$45.8k, Display R$93.6k, YouTube R$81.9k, PMax R$14.6k\n"
            "Investimento nov/24: FB R$60.4k, TikTok R$33k, Display R$58.3k, YouTube R$60.4k\n\n"
            "Sessões: 938k (25) vs 1.05M (24)\n"
            "Engajamento: 22.5M vs 20.25M\n"
            "Views: 15M vs 11.2M\n\n"
            "Contexto: aumento de budget em display e youtube para suportar lançamento de produto X"
        ),
        key="dados_brutos_tab9",
        label_visibility="collapsed"
    )

    if st.button("🤖 Analisar com Gemini", type="primary", use_container_width=True, key="analisar_yoy_tab9"):
        if not dados_brutos.strip():
            st.error("❌ Cole os dados antes de analisar.")
        elif not modelo_texto:
            st.error("❌ Gemini não configurado.")
        else:
            contexto = f"Cliente: {cliente_yoy}" + (f" | Período: {periodo_ref}" if periodo_ref.strip() else "")
            with st.spinner("🤖 Analisando dados YoY..."):
                try:
                    result = generate_yoy_analysis(modelo_texto, dados_brutos, contexto)
                    st.session_state.yoy_analysis = result
                    st.session_state.yoy_download_name = f"yoy_{cliente_yoy.replace(' ', '_')}.txt"
                    st.success("✅ Análise gerada!")
                except Exception as e:
                    st.error(f"❌ Erro: {str(e)[:200]}")

    if st.session_state.get('yoy_analysis'):
        st.markdown("---")
        st.markdown("### 📄 Análise YoY")

        st.download_button(
            label="💾 Baixar análise",
            data=st.session_state.yoy_analysis,
            file_name=st.session_state.get('yoy_download_name', 'analise_yoy.txt'),
            mime="text/plain",
            key="download_yoy_tab9"
        )

        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
        st.markdown(st.session_state.yoy_analysis)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🔄 Nova análise", use_container_width=True, key="nova_yoy_tab9"):
            st.session_state.yoy_analysis = None
            st.rerun()
