import streamlit as st
from datetime import datetime
from services.gemini_service import review_briefing, analyze_pasted_data, generate_slides_description


def _reset():
    for k in ('briefing_etapa', 'briefing_texto', 'briefing_review',
              'briefing_respostas', 'pasted_data_analysis', 'pasted_slides',
              'file_text_tab10'):
        st.session_state.pop(k, None)


def _ler_arquivo(uploaded_file):
    """Extrai texto de TXT, CSV ou PDF."""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "text/csv":
            import pandas as pd
            return pd.read_csv(uploaded_file).to_string(index=False)
        else:  
            try:
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                import pypdf
                reader = pypdf.PdfReader(uploaded_file)
                return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {str(e)[:100]}")
        return ""


def render_tab_dados_colados(modelo_texto):
    st.markdown('<div class="pasted-data"><h2>📋 Briefing & Análise</h2></div>', unsafe_allow_html=True)

    etapa = st.session_state.get('briefing_etapa', 0)
    if etapa == 0:
        st.caption("Cole o briefing, suba um arquivo ou faça os dois. Qualquer formato funciona.")

        col_in1, col_in2 = st.columns([3, 1])

        with col_in1:
            texto_colado = st.text_area(
                "Briefing:",
                height=240,
                placeholder=(
                    "Cole qualquer coisa — e-mail, tabela, números soltos, relatório parcial...\n\n"
                    "Ex:\n"
                    "Cliente: Syngenta | Nov/25 vs Nov/24\n"
                    "FB: R$60.8k → R$60.4k | TikTok: R$45.8k → R$33k\n"
                    "Sessões: 938k vs 1.05M | Engajamento: 22.5M vs 20.25M\n"
                    "Contexto: lançamento produto X, aumento budget display"
                ),
                key="texto_colado_tab10",
                label_visibility="collapsed"
            )

        with col_in2:
            st.markdown("**Arquivo** *(PDF, TXT, CSV)*")
            uploaded_file = st.file_uploader(
                "Arquivo",
                type=["pdf", "txt", "csv"],
                key="upload_briefing_tab10",
                label_visibility="collapsed"
            )
            if uploaded_file:
                file_text = _ler_arquivo(uploaded_file)
                if file_text:
                    st.session_state.file_text_tab10 = file_text
                    st.success(f"✅ {uploaded_file.name}")

        texto_completo = texto_colado.strip()
        file_saved = st.session_state.get('file_text_tab10', '')
        if file_saved:
            texto_completo = (texto_completo + "\n\n---\n" + file_saved).strip() if texto_completo else file_saved

        if st.button("➡️ Continuar", type="primary", use_container_width=True, key="btn_continuar_tab10"):
            if not texto_completo:
                st.error("❌ Cole ou suba o briefing antes de continuar.")
            elif not modelo_texto:
                st.error("❌ Gemini não configurado.")
            else:
                st.session_state.briefing_texto = texto_completo
                review = review_briefing(modelo_texto, texto_completo)
                st.session_state.briefing_review = review
                st.session_state.briefing_etapa = 1
                st.rerun()

    elif etapa == 1:
        review = st.session_state.get('briefing_review', '')

        st.markdown("### 🔍 Leitura do briefing")
        st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
        st.markdown(review)
        st.markdown('</div>', unsafe_allow_html=True)

        # Só mostra campo de respostas se o Gemini fez perguntas
        tem_perguntas = "Pronto para gerar" not in review and "❓" in review

        respostas = ""
        if tem_perguntas:
            st.markdown("### ✏️ Suas respostas *(opcional — responda o que souber)*")
            respostas = st.text_area(
                "Respostas:",
                height=150,
                placeholder="Responda aqui as perguntas acima. Deixe em branco para pular.",
                key="respostas_tab10",
                label_visibility="collapsed"
            )

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            if st.button("🤖 Gerar Análise", type="primary", use_container_width=True, key="btn_gerar_tab10"):
                contexto = respostas.strip() if respostas.strip() else ""
                with st.spinner("🤖 Gerando análise..."):
                    result, _ = analyze_pasted_data(
                        modelo_texto,
                        st.session_state.briefing_texto,
                        contexto
                    )
                st.session_state.pasted_data_analysis = result
                st.session_state.pasted_slides = None
                st.session_state.briefing_respostas = contexto
                st.session_state.briefing_etapa = 2
                st.rerun()

        with col_b2:
            if st.button("↩️ Editar briefing", use_container_width=True, key="btn_voltar_tab10"):
                st.session_state.briefing_etapa = 0
                st.session_state.pop('briefing_review', None)
                st.rerun()

    elif etapa == 2:
        analysis_result = st.session_state.get('pasted_data_analysis', '')
        slides_result = st.session_state.get('pasted_slides')
        col_a1, col_a2, col_a3 = st.columns(3)

        with col_a1:
            st.download_button(
                "💾 Baixar análise",
                data=analysis_result,
                file_name=f"analise_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_analysis_tab10"
            )

        with col_a2:
            if st.button("🎬 Gerar Apresentação", use_container_width=True, key="btn_slides_tab10",
                         disabled=not analysis_result):
                with st.spinner("🎬 Estruturando slides..."):
                    slides = generate_slides_description(
                        modelo_texto,
                        analysis_result,
                        st.session_state.get('briefing_respostas', '')
                    )
                st.session_state.pasted_slides = slides
                st.rerun()

        with col_a3:
            if st.button("🔄 Nova análise", use_container_width=True, key="btn_nova_tab10"):
                _reset()
                st.rerun()

        tab_analise, tab_slides = st.tabs(["📄 Análise", "🎬 Apresentação"])

        with tab_analise:
            st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
            st.markdown(analysis_result)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab_slides:
            if not slides_result:
                st.info("Clique em **🎬 Gerar Apresentação** para estruturar os slides.")
            else:
                st.download_button(
                    "💾 Baixar slides",
                    data=slides_result,
                    file_name=f"slides_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_slides_tab10"
                )
                st.markdown('<div class="gemini-response">', unsafe_allow_html=True)
                st.markdown(slides_result)
                st.markdown('</div>', unsafe_allow_html=True)
