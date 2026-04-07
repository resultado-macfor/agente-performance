import os
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st

from utils.styles import apply_styles
from utils.helpers import safe_metric, identificar_colunas_numericas
from services.bigquery_service import get_bigquery_client, load_all_columns_data
from services.gemini_service import init_gemini
from agent.campaign_classifier import classificar_campanhas_multi_cliente
from config.constants import LISTA_PRODUTOS, OPCOES_CLIENTES, DATA_SOURCES_OPCOES
from views import (
    render_tab_visao_geral,
    render_tab_visualizar_dados,
    render_tab_performance,
    render_tab_analise_ia,
    render_tab_classificador,
    render_tab_mom,
    render_tab_yoy,
    render_tab_dados_colados,
)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.set_page_config(layout="centered")
    st.title("🔒 Agente Performance")
    senha_input = st.text_input("Digite a senha de acesso:", type="password")
    if st.button("Acessar"):
        senha_correta = os.getenv('senha_per')
        if not senha_correta:
            st.error("Sistema não configurado.")
        elif senha_input == senha_correta:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
    st.stop()

st.set_page_config(layout="wide", page_title="Agente Performance", page_icon="📊")
apply_styles()
st.markdown('<div class="header-gradient"><h1>📊 Agente Performance</h1></div>', unsafe_allow_html=True)

modelo_texto = init_gemini()

defaults = {
    'df_completo': pd.DataFrame(),
    'colunas_numericas': [],
    'gemini_analysis': None,
    'df_classificado': pd.DataFrame(),
    'relatorio_classificacao': None,
    'filtros_aplicados': {},
    'slides_description': None,
    'mom_analysis': None,
    'yoy_analysis': None,
    'pasted_data_analysis': None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.header("⚙️ Configurações")

    if st.button("Testar Conexão BigQuery"):
        with st.spinner("Conectando..."):
            if get_bigquery_client():
                st.success("✅ Conexão OK!")

    st.subheader("👥 Filtro por Cliente")
    filtro_cliente = st.selectbox("Selecione o Cliente:", options=OPCOES_CLIENTES, index=0)

    st.subheader("📱 Data Sources")
    selected_sources = st.multiselect("Data Sources", options=DATA_SOURCES_OPCOES, default=DATA_SOURCES_OPCOES[:3])

    st.subheader("📅 Período")
    periodo = st.radio(
        "Selecione",
        ["Últimos 30 dias", "Últimos 90 dias", "Últimos 180 dias", "Todo período", "Personalizado"],
        index=1
    )

    data_fim = datetime.now().date()
    if periodo == "Últimos 30 dias":
        data_inicio = data_fim - timedelta(days=30)
    elif periodo == "Últimos 90 dias":
        data_inicio = data_fim - timedelta(days=90)
    elif periodo == "Últimos 180 dias":
        data_inicio = data_fim - timedelta(days=180)
    elif periodo == "Todo período":
        data_inicio = None
        data_fim = None
    else:
        col1, col2 = st.columns(2)
        with col1:
            data_inicio = st.date_input("Início", value=datetime.now().date() - timedelta(days=90))
        with col2:
            data_fim = st.date_input("Fim", value=datetime.now().date())

    limite_default = 20000
    max_limit = min(100000, max(limite_default, len(st.session_state.df_completo))) if not st.session_state.df_completo.empty else 100000
    limite = st.slider("Limite de registros", 1000, max_limit, min(limite_default, max_limit), 1000)

    if st.button("📊 Carregar Dados", use_container_width=True, type="primary"):
        with st.spinner("Carregando..."):
            client = get_bigquery_client()
            if client:
                df_loaded = load_all_columns_data(
                    client,
                    data_inicio=data_inicio,
                    data_fim=data_fim,
                    data_sources=selected_sources,
                    filtro_cliente=filtro_cliente,
                    limit=limite
                )
                if not df_loaded.empty:
                    st.session_state.df_completo = df_loaded
                    st.session_state.colunas_numericas = identificar_colunas_numericas(df_loaded)
                    st.session_state.df_classificado = classificar_campanhas_multi_cliente(df_loaded)
                    st.session_state.gemini_analysis = None
                    st.session_state.filtros_aplicados = {}
                    st.success(f"✅ {len(df_loaded):,} registros carregados e classificados")
                    st.rerun()
                else:
                    st.error("Nenhum dado encontrado")
            else:
                st.error("❌ Não foi possível conectar.")



df = st.session_state.df_completo
df_classificado = st.session_state.df_classificado

if df.empty:
    st.warning("📭 Nenhum dado carregado. Use o botão na sidebar para carregar dados.")
    st.stop()



st.markdown("## 🔍 Filtros Avançados por Categoria de Campanha")

filtro_col1, filtro_col2, filtro_col3, filtro_col4 = st.columns(4)

with filtro_col1:
    if 'campaign_produto' in df_classificado.columns:
        produto_selecionado = st.selectbox("📦 Produto:", options=['Todos'] + LISTA_PRODUTOS, key="produto_selectbox")
        if produto_selecionado != 'Todos':
            st.session_state.filtros_aplicados['campaign_produto'] = produto_selecionado
        elif 'campaign_produto' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_produto']

    if 'campaign_cultura' in df_classificado.columns:
        culturas = sorted(df_classificado['campaign_cultura'].dropna().unique())
        cultura_sel = st.selectbox("🌱 Cultura/Setor:", options=['Todas'] + list(culturas))
        if cultura_sel != 'Todas':
            st.session_state.filtros_aplicados['campaign_cultura'] = cultura_sel
        elif 'campaign_cultura' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_cultura']

with filtro_col2:
    if 'campaign_tipo_campanha' in df_classificado.columns:
        tipos = sorted(df_classificado['campaign_tipo_campanha'].dropna().unique())
        tipo_sel = st.selectbox("🎯 Tipo de Campanha:", options=['Todos'] + list(tipos))
        if tipo_sel != 'Todos':
            st.session_state.filtros_aplicados['campaign_tipo_campanha'] = tipo_sel
        elif 'campaign_tipo_campanha' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_tipo_campanha']

    if 'campaign_objetivo' in df_classificado.columns:
        objetivos = sorted(df_classificado['campaign_objetivo'].dropna().unique())
        obj_sel = st.selectbox("🎯 Objetivo:", options=['Todos'] + list(objetivos))
        if obj_sel != 'Todos':
            st.session_state.filtros_aplicados['campaign_objetivo'] = obj_sel
        elif 'campaign_objetivo' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_objetivo']

with filtro_col3:
    if 'campaign_etapa_funil' in df_classificado.columns:
        etapas = sorted(df_classificado['campaign_etapa_funil'].dropna().unique())
        etapa_sel = st.selectbox("📊 Etapa do Funil:", options=['Todas'] + list(etapas))
        if etapa_sel != 'Todas':
            st.session_state.filtros_aplicados['campaign_etapa_funil'] = etapa_sel
        elif 'campaign_etapa_funil' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_etapa_funil']

    if 'campaign_iniciativa' in df_classificado.columns:
        iniciativas = sorted(df_classificado['campaign_iniciativa'].dropna().unique())
        inic_sel = st.selectbox("🚀 Iniciativa:", options=['Todas'] + list(iniciativas))
        if inic_sel != 'Todas':
            st.session_state.filtros_aplicados['campaign_iniciativa'] = inic_sel
        elif 'campaign_iniciativa' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_iniciativa']

with filtro_col4:
    if 'campaign_plataforma' in df_classificado.columns:
        plataformas = sorted(df_classificado['campaign_plataforma'].dropna().unique())
        plat_sel = st.selectbox("🖥️ Plataforma:", options=['Todas'] + list(plataformas))
        if plat_sel != 'Todas':
            st.session_state.filtros_aplicados['campaign_plataforma'] = plat_sel
        elif 'campaign_plataforma' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_plataforma']

    if 'campaign_agencia' in df_classificado.columns:
        agencias = sorted(df_classificado['campaign_agencia'].dropna().unique())
        ag_sel = st.selectbox("🏢 Agência:", options=['Todas'] + list(agencias))
        if ag_sel != 'Todas':
            st.session_state.filtros_aplicados['campaign_agencia'] = ag_sel
        elif 'campaign_agencia' in st.session_state.filtros_aplicados:
            del st.session_state.filtros_aplicados['campaign_agencia']

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    if st.button("✅ Aplicar Filtros", use_container_width=True):
        st.rerun()
with col_btn2:
    if st.button("🔄 Limpar Filtros", use_container_width=True):
        st.session_state.filtros_aplicados = {}
        st.rerun()
with col_btn3:
    busca_campanha = st.text_input("", placeholder="Digite parte do nome da campanha...", key="busca_campanha_input", label_visibility="collapsed")
    if busca_campanha:
        st.session_state.busca_campanha = busca_campanha
    elif 'busca_campanha' not in st.session_state:
        st.session_state.busca_campanha = ""


df_filtrado = df_classificado.copy()
for coluna, valor in st.session_state.filtros_aplicados.items():
    if coluna in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado[coluna] == valor]

busca_termo = st.session_state.get('busca_campanha', '').lower().strip()
if busca_termo and 'campaign' in df_filtrado.columns:
    df_filtrado = df_filtrado[df_filtrado['campaign'].astype(str).str.lower().str.contains(busca_termo, na=False)]


filtros_ativos = [f"{k.replace('campaign_', '')}: {v}" for k, v in st.session_state.filtros_aplicados.items()]
if busca_termo:
    filtros_ativos.append(f"Busca: '{busca_termo}'")

if filtros_ativos:
    st.markdown(f"### 📊 Dados Filtrados: {len(df_filtrado):,} registros")
    st.info(f"**Filtros ativos:** {' | '.join(filtros_ativos)}")
    col_badges = st.columns(min(8, len(filtros_ativos)))
    for idx, filtro in enumerate(filtros_ativos):
        with col_badges[idx % 8]:
            st.markdown(f'<span style="background:#e0f2fe; padding:5px 10px; border-radius:10px; margin:2px; font-size:0.8em">{filtro}</span>', unsafe_allow_html=True)
else:
    st.markdown(f"### 📊 Dados Completos: {len(df_filtrado):,} registros")
    st.info("ℹ️ Nenhum filtro aplicado. Todos os dados estão visíveis.")



tab1, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📋 Visão Geral",
    "📊 Visualizar Dados",
    "🎯 Performance",
    "🤖 Análise com IA",
    "🎪 Classificador Campanhas",
    "📅 Análise MoM",
    "📊 Cenário YoY",
    "📋 Dados Colados",
])

with tab1:
    render_tab_visao_geral(df_filtrado)

with tab4:
    render_tab_visualizar_dados(df_filtrado)

with tab5:
    render_tab_performance(df_filtrado)

with tab6:
    render_tab_analise_ia(df_filtrado, modelo_texto)

with tab7:
    render_tab_classificador(df, df_classificado, modelo_texto)

with tab8:
    render_tab_mom(df, modelo_texto)

with tab9:
    render_tab_yoy(modelo_texto)

with tab10:
    render_tab_dados_colados(modelo_texto)



st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    if not df_filtrado.empty:
        st.caption(f"📊 Dados: {len(df_filtrado):,} registros")
        if st.session_state.filtros_aplicados:
            st.caption(f"🔍 Filtros: {len(st.session_state.filtros_aplicados)} ativos")

with footer_col2:
    if 'campaign' in df_filtrado.columns:
        try:
            st.caption(f"🎯 Campanhas: {df_filtrado['campaign'].nunique()}")
        except:
            st.caption("🎯 Campanhas: Erro")

with footer_col3:
    st.caption(f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}")

if modelo_texto:
    st.sidebar.success("✅ Gemini ativo")
else:
    st.sidebar.info("ℹ️ Gemini inativo")
