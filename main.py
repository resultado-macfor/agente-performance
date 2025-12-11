# app_completo.py - App com TODAS as colunas do BigQuery
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from google.oauth2 import service_account
from google.cloud import bigquery
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io

# Configuração da página
st.set_page_config(
    layout="wide",
    page_title="Analytics Platform",
    page_icon="📊"
)

# CSS personalizado
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton button {
        background-color: #4f46e5 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 12px;
        margin: 5px;
        text-align: center;
    }
    .okr-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        border-left: 4px solid #4f46e5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #4f46e5 !important;
    }
    .column-info {
        background: #f0f9ff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #0ea5e9;
    }
    .data-card {
        background: white;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title("📊 Analytics Platform - TODAS as Colunas")

# =============================================================================
# CONEXÃO E CARREGAMENTO - TODAS AS COLUNAS (COM STREAMLIT SECRETS)
# =============================================================================

@st.cache_resource
def get_bigquery_client():
    """Cria cliente BigQuery usando credenciais do Streamlit Secrets"""
    try:
        # Verifica se os secrets estão configurados
        if "gcp_service_account" not in st.secrets:
            st.error("❌ Credenciais do Google Cloud não encontradas no Streamlit Secrets")
            
            return None
        
        # Carrega as credenciais do Streamlit Secrets
        service_account_info = dict(st.secrets["gcp_service_account"])
        
        # Ajusta a chave privada se necessário
        if isinstance(service_account_info.get("private_key"), str):
            # Garante que a chave tenha quebras de linha corretas
            private_key = service_account_info["private_key"]
            if "\\n" in private_key:
                service_account_info["private_key"] = private_key.replace("\\n", "\n")
        
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        client = bigquery.Client(
            credentials=credentials,
            project=service_account_info["project_id"]
        )
        
        st.success("✅ Conectado ao BigQuery usando credenciais do Streamlit Secrets")
        return client
    
    except KeyError as e:
        st.error(f"❌ Chave não encontrada no secrets: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Erro na conexão com BigQuery: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_all_columns_data(_client, data_inicio=None, data_fim=None, data_sources=None, limit=50000):
    """Carrega TODAS as colunas disponíveis na tabela"""
    try:
        # Primeiro, vamos descobrir quais colunas existem
        st.info("🔍 Analisando estrutura da tabela...")
        
        # Query para obter todas as colunas
        query_schema = """
        SELECT column_name, data_type 
        FROM `macfor-media-flow.ads.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = 'app_view_campaigns'
        """
        
        try:
            schema_df = _client.query(query_schema).to_dataframe()
            todas_colunas = schema_df['column_name'].tolist()
            st.success(f"✅ Encontradas {len(todas_colunas)} colunas na tabela")
        except Exception as schema_error:
            # Se falhar, usar colunas padrão
            st.warning(f"Não foi possível obter schema automático: {schema_error}")
            todas_colunas = "*"
        
        # Construir query dinâmica
        if isinstance(todas_colunas, list):
            colunas_query = ",\n            ".join(todas_colunas)
        else:
            colunas_query = "*"
        
        query = f"""
        SELECT 
            {colunas_query}
        FROM `macfor-media-flow.ads.app_view_campaigns`
        """
        
        conditions = []
        
        if data_inicio:
            conditions.append(f"DATE(date) >= DATE('{data_inicio}')")
        if data_fim:
            conditions.append(f"DATE(date) <= DATE('{data_fim}')")
        if data_sources and len(data_sources) > 0:
            ds_str = ", ".join([f"'{ds}'" for ds in data_sources])
            conditions.append(f"datasource IN ({ds_str})")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY date DESC LIMIT {limit}"
        
        st.info("📥 Carregando dados... Isso pode levar alguns instantes")
        df = _client.query(query).to_dataframe()
        
        if df.empty:
            st.warning("Nenhum dado encontrado")
            return pd.DataFrame()
        
        st.success(f"✅ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
        
        # Identificar e converter colunas numéricas
        colunas_numericas = []
        colunas_texto = []
        colunas_data = []
        
        for col in df.columns:
            # Tentar inferir tipo
            try:
                # Primeiro tenta converter para numérico
                amostra = df[col].dropna().head(100)
                if len(amostra) > 0:
                    # Testa se parece numérico
                    if pd.api.types.is_numeric_dtype(df[col]):
                        colunas_numericas.append(col)
                        # Converter para numérico
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif 'date' in col.lower() or 'data' in col.lower():
                        colunas_data.append(col)
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except:
                            pass
                    else:
                        colunas_texto.append(col)
            except:
                colunas_texto.append(col)
        
        # Log das colunas identificadas
        with st.expander("📋 Informações das Colunas"):
            st.write(f"**Numéricas ({len(colunas_numericas)}):** {', '.join(colunas_numericas[:10])}...")
            st.write(f"**Texto ({len(colunas_texto)}):** {', '.join(colunas_texto[:10])}...")
            st.write(f"**Datas ({len(colunas_data)}):** {', '.join(colunas_data)}")
        
        return df
    
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# FUNÇÕES DE ANÁLISE
# =============================================================================

def identificar_colunas_numericas(df):
    """Identifica automaticamente colunas numéricas"""
    colunas_numericas = []
    
    for col in df.columns:
        try:
            # Tenta converter para numérico
            if pd.api.types.is_numeric_dtype(df[col]):
                colunas_numericas.append(col)
            # Ou se tem pelo menos 50% de valores numéricos
            elif df[col].dropna().apply(lambda x: isinstance(x, (int, float, np.number))).any():
                colunas_numericas.append(col)
        except:
            continue
    
    return colunas_numericas

def analisar_coluna(df, coluna):
    """Analisa uma coluna específica"""
    if coluna not in df.columns:
        return None
    
    dados_coluna = df[coluna]
    analise = {
        'nome': coluna,
        'tipo': str(dados_coluna.dtype),
        'total': len(dados_coluna),
        'nao_nulos': dados_coluna.notna().sum(),
        'nulos': dados_coluna.isna().sum(),
        'percentual_nulos': (dados_coluna.isna().sum() / len(dados_coluna)) * 100,
        'valores_unicos': dados_coluna.nunique()
    }
    
    # Se for numérica
    if pd.api.types.is_numeric_dtype(dados_coluna):
        dados_validos = dados_coluna.dropna()
        if len(dados_validos) > 0:
            analise.update({
                'tipo_detalhado': 'Numérica',
                'min': dados_validos.min(),
                'max': dados_validos.max(),
                'media': dados_validos.mean(),
                'mediana': dados_validos.median(),
                'desvio_padrao': dados_validos.std(),
                'q1': dados_validos.quantile(0.25),
                'q3': dados_validos.quantile(0.75),
                'assimetria': dados_validos.skew(),
                'curtose': dados_validos.kurt()
            })
    # Se for categórica/texto
    elif dados_coluna.dtype == 'object':
        analise.update({
            'tipo_detalhado': 'Texto/Categórica',
            'valores_mais_comuns': dados_coluna.value_counts().head(10).to_dict(),
            'valor_mais_frequente': dados_coluna.mode().iloc[0] if not dados_coluna.mode().empty else None,
            'frequencia_valor_mais_comum': dados_coluna.value_counts().iloc[0] if not dados_coluna.empty else 0
        })
    # Se for data
    elif pd.api.types.is_datetime64_any_dtype(dados_coluna):
        dados_validos = dados_coluna.dropna()
        if len(dados_validos) > 0:
            analise.update({
                'tipo_detalhado': 'Data',
                'data_minima': dados_validos.min(),
                'data_maxima': dados_validos.max(),
                'intervalo_dias': (dados_validos.max() - dados_validos.min()).days
            })
    
    return analise

def criar_visualizacao_coluna(df, coluna):
    """Cria visualização adequada para o tipo de coluna"""
    if coluna not in df.columns:
        return None
    
    dados = df[coluna].dropna()
    
    # Coluna numérica
    if pd.api.types.is_numeric_dtype(df[coluna]):
        fig = px.histogram(
            df, 
            x=coluna,
            nbins=50,
            title=f"Distribuição de {coluna}",
            marginal="box"
        )
        return fig
    
    # Coluna categórica/texto (até 50 categorias)
    elif df[coluna].nunique() <= 50:
        contagem = df[coluna].value_counts().head(20)
        fig = px.bar(
            x=contagem.index,
            y=contagem.values,
            title=f"Top 20 Valores em {coluna}",
            labels={'x': coluna, 'y': 'Contagem'}
        )
        fig.update_xaxes(tickangle=45)
        return fig
    
    # Coluna data
    elif pd.api.types.is_datetime64_any_dtype(df[coluna]):
        contagem_diaria = df.groupby(df[coluna].dt.date).size().reset_index()
        contagem_diaria.columns = ['data', 'contagem']
        
        fig = px.line(
            contagem_diaria,
            x='data',
            y='contagem',
            title=f"Frequência por Data - {coluna}"
        )
        return fig
    
    return None

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

# Inicializar estado
if 'df_completo' not in st.session_state:
    st.session_state.df_completo = pd.DataFrame()
if 'colunas_numericas' not in st.session_state:
    st.session_state.colunas_numericas = []

# Sidebar
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Verificação de conexão
    st.subheader("🔗 Conexão")
    if st.button("Testar Conexão BigQuery"):
        with st.spinner("Conectando..."):
            client = get_bigquery_client()
            if client:
                st.success("✅ Conexão bem-sucedida!")
    
    # Data sources
    data_sources_opcoes = ["facebook", "google ads", "tiktok"]
    selected_sources = st.multiselect(
        "Data Sources",
        options=data_sources_opcoes,
        default=data_sources_opcoes
    )
    
    # Período
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
            data_inicio = st.date_input("Início", value=data_fim - timedelta(days=90))
        with col2:
            data_fim = st.date_input("Fim", value=data_fim)
    
    # Limite
    limite = st.slider("Limite de registros", 1000, 100000, 20000, 1000)
    
    # Botão carregar
    if st.button("🔄 Carregar TODOS os Dados", type="primary", use_container_width=True):
        with st.spinner("Carregando TODAS as colunas... Isso pode demorar"):
            client = get_bigquery_client()
            if client:
                df = load_all_columns_data(
                    client,
                    data_inicio=data_inicio,
                    data_fim=data_fim,
                    data_sources=selected_sources,
                    limit=limite
                )
                
                if not df.empty:
                    st.session_state.df_completo = df
                    # Identificar colunas numéricas
                    st.session_state.colunas_numericas = identificar_colunas_numericas(df)
                    st.success(f"✅ {len(df):,} registros carregados")
                    st.success(f"📊 {len(st.session_state.colunas_numericas)} colunas numéricas identificadas")
                else:
                    st.error("Nenhum dado encontrado")



df = st.session_state.df_completo
colunas_numericas = st.session_state.colunas_numericas

# Abas principais
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Visão Geral das Colunas", 
    "📈 Análise Numérica", 
    "🔍 Explorar Colunas", 
    "📊 Visualizar Dados",
    "📐 Análise Estatística"
])

# =============================================================================
# TAB 1: VISÃO GERAL DAS COLUNAS
# =============================================================================

with tab1:
    st.header("📋 Visão Geral de TODAS as Colunas")
    
    # Estatísticas gerais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Colunas", len(df.columns))
    
    with col2:
        st.metric("Colunas Numéricas", len(colunas_numericas))
    
    with col3:
        st.metric("Total de Registros", f"{len(df):,}")
    
    with col4:
        memoria_mb = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Uso de Memória", f"{memoria_mb:.1f} MB")
    
    # Lista de todas as colunas com informações
    st.subheader("📊 Detalhes de Cada Coluna")
    
    # Filtrar colunas
    col_filtro1, col_filtro2 = st.columns(2)
    
    with col_filtro1:
        tipo_filtro = st.selectbox(
            "Filtrar por tipo",
            ["Todas", "Numéricas", "Texto", "Datas"]
        )
    
    with col_filtro2:
        pesquisa_coluna = st.text_input("🔍 Pesquisar coluna", "")
    
    # Preparar lista de colunas filtradas
    colunas_para_mostrar = []
    
    for col in df.columns:
        incluir = True
        
        # Filtrar por tipo
        if tipo_filtro == "Numéricas":
            incluir = col in colunas_numericas
        elif tipo_filtro == "Texto":
            incluir = df[col].dtype == 'object' and col not in colunas_numericas
        elif tipo_filtro == "Datas":
            incluir = pd.api.types.is_datetime64_any_dtype(df[col])
        
        # Filtrar por pesquisa
        if pesquisa_coluna and pesquisa_coluna.lower() not in col.lower():
            incluir = False
        
        if incluir:
            colunas_para_mostrar.append(col)
    
    # Mostrar informações de cada coluna
    for col in sorted(colunas_para_mostrar)[:50]:  # Limitar a 50 para performance
        analise = analisar_coluna(df, col)
        
        with st.expander(f"**{col}** ({analise['tipo_detalhado'] if 'tipo_detalhado' in analise else analise['tipo']})"):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.write(f"**Tipo:** {analise['tipo']}")
                st.write(f"**Não nulos:** {analise['nao_nulos']:,} ({analise['percentual_nulos']:.1f}% nulos)")
                st.write(f"**Valores únicos:** {analise['valores_unicos']:,}")
            
            with col_info2:
                if analise['tipo_detalhado'] == 'Numérica':
                    st.write(f"**Média:** {analise.get('media', 'N/A'):.2f}")
                    st.write(f"**Min:** {analise.get('min', 'N/A'):.2f}")
                    st.write(f"**Max:** {analise.get('max', 'N/A'):.2f}")
                elif analise['tipo_detalhado'] == 'Texto/Categórica':
                    if analise.get('valor_mais_frequente'):
                        st.write(f"**Valor mais comum:** {analise['valor_mais_frequente']}")
                        st.write(f"**Frequência:** {analise['frequencia_valor_mais_comum']:,}")
                elif analise['tipo_detalhado'] == 'Data':
                    st.write(f"**Período:** {analise.get('data_minima', 'N/A')} a {analise.get('data_maxima', 'N/A')}")
            
            # Botão para visualizar
            if st.button(f"📊 Visualizar {col}", key=f"viz_{col}"):
                st.session_state.coluna_selecionada = col
                st.rerun()

# =============================================================================
# TAB 2: ANÁLISE NUMÉRICA
# =============================================================================

with tab2:
    st.header("📈 Análise de Colunas Numéricas")
    
    if not colunas_numericas:
        st.warning("Nenhuma coluna numérica identificada")
    else:
        st.success(f"✅ {len(colunas_numericas)} colunas numéricas disponíveis para análise")
        
        # Selecionar colunas para análise
        colunas_selecionadas = st.multiselect(
            "Selecione colunas numéricas para análise",
            options=colunas_numericas,
            default=colunas_numericas[:5] if len(colunas_numericas) >= 5 else colunas_numericas
        )
        
        if colunas_selecionadas:
            # Estatísticas descritivas
            st.subheader("📊 Estatísticas Descritivas")
            
            stats_df = df[colunas_selecionadas].describe().T
            stats_df['missing'] = df[colunas_selecionadas].isna().sum()
            stats_df['missing_pct'] = (df[colunas_selecionadas].isna().sum() / len(df) * 100)
            
            # Formatar
            def formatar_numero(x):
                if isinstance(x, (int, np.integer)):
                    return f"{x:,}"
                elif isinstance(x, (float, np.floating)):
                    if abs(x) < 0.01:
                        return f"{x:.4f}"
                    elif abs(x) < 1:
                        return f"{x:.3f}"
                    elif abs(x) < 1000:
                        return f"{x:.2f}"
                    else:
                        return f"{x:,.0f}"
                return str(x)
            
            st.dataframe(
                stats_df.style.format(formatar_numero),
                use_container_width=True
            )
            
            # Histogramas
            st.subheader("📈 Distribuições")
            
            num_cols = min(3, len(colunas_selecionadas))
            cols_vis = st.columns(num_cols)
            
            for idx, col in enumerate(colunas_selecionadas[:num_cols*3]):
                with cols_vis[idx % num_cols]:
                    fig = criar_visualizacao_coluna(df, col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            # Correlações
            if len(colunas_selecionadas) >= 2:
                st.subheader("🔥 Matriz de Correlação")
                
                # Calcular correlações
                correlacao = df[colunas_selecionadas].corr()
                
                fig_corr = px.imshow(
                    correlacao,
                    text_auto='.2f',
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlações entre Variáveis Numéricas"
                )
                fig_corr.update_layout(height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Top correlações
                st.subheader("🔗 Principais Correlações")
                
                correlacoes_fortes = []
                for i in range(len(correlacao.columns)):
                    for j in range(i+1, len(correlacao.columns)):
                        corr = correlacao.iloc[i, j]
                        if abs(corr) > 0.3 and not pd.isna(corr):
                            correlacoes_fortes.append({
                                'Variável 1': correlacao.columns[i],
                                'Variável 2': correlacao.columns[j],
                                'Correlação': corr,
                                'Força': 'Forte' if abs(corr) > 0.7 else 'Moderada'
                            })
                
                if correlacoes_fortes:
                    correlacoes_fortes.sort(key=lambda x: abs(x['Correlação']), reverse=True)
                    df_corr = pd.DataFrame(correlacoes_fortes[:20])
                    st.dataframe(df_corr, use_container_width=True)
                else:
                    st.info("Não foram encontradas correlações fortes (> 0.3)")

# =============================================================================
# TAB 3: EXPLORAR COLUNAS
# =============================================================================

with tab3:
    st.header("🔍 Explorar Colunas Individualmente")
    
    coluna_selecionada = st.selectbox(
        "Selecione uma coluna para explorar",
        options=sorted(df.columns),
        index=0
    )
    
    if coluna_selecionada:
        analise = analisar_coluna(df, coluna_selecionada)
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.metric("Total de Valores", analise['total'])
            st.metric("Valores Não Nulos", f"{analise['nao_nulos']:,}")
            st.metric("Valores Únicos", f"{analise['valores_unicos']:,}")
        
        with col_info2:
            st.metric("Valores Nulos", f"{analise['nulos']:,}")
            st.metric("% Nulos", f"{analise['percentual_nulos']:.1f}%")
        
        # Visualização
        st.subheader("📊 Visualização")
        fig = criar_visualizacao_coluna(df, coluna_selecionada)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Valores
        st.subheader("📋 Amostra de Valores")
        
        col_amostra1, col_amostra2 = st.columns(2)
        
        with col_amostra1:
            st.write("**Primeiros 10 valores:**")
            st.write(df[coluna_selecionada].head(10).tolist())
        
        with col_amostra2:
            st.write("**Últimos 10 valores:**")
            st.write(df[coluna_selecionada].tail(10).tolist())
        
        # Se for categórica, mostrar distribuição
        if analise['tipo_detalhado'] == 'Texto/Categórica' and analise['valores_unicos'] <= 100:
            st.subheader("📊 Distribuição de Valores")
            
            contagem = df[coluna_selecionada].value_counts()
            df_contagem = pd.DataFrame({
                'Valor': contagem.index,
                'Contagem': contagem.values,
                'Percentual': (contagem.values / len(df) * 100)
            })
            
            st.dataframe(
                df_contagem.style.format({'Contagem': '{:,}', 'Percentual': '{:.1f}%'}),
                use_container_width=True
            )

# =============================================================================
# TAB 4: VISUALIZAR DADOS
# =============================================================================

with tab4:
    st.header("📊 Visualizar Dados Completos")
    
    # Selecionar colunas para visualizar
    colunas_vis = st.multiselect(
        "Selecione colunas para visualizar",
        options=sorted(df.columns),
        default=sorted(df.columns)[:10]
    )
    
    if colunas_vis:
        # Filtros
        st.subheader("🔍 Filtros")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            # Filtro por datasource se existir
            if 'datasource' in df.columns:
                datasources = sorted(df['datasource'].unique())
                ds_selecionados = st.multiselect(
                    "Data Sources",
                    options=datasources,
                    default=datasources[:min(3, len(datasources))]
                )
                if ds_selecionados:
                    df_filtrado = df[df['datasource'].isin(ds_selecionados)]
                else:
                    df_filtrado = df.copy()
            else:
                df_filtrado = df.copy()
        
        with col_f2:
            # Filtro por valor mínimo se coluna numérica selecionada
            colunas_num_vis = [c for c in colunas_vis if c in colunas_numericas]
            if colunas_num_vis:
                col_filtro = st.selectbox(
                    "Filtrar por coluna numérica",
                    options=['Nenhum'] + colunas_num_vis
                )
                if col_filtro != 'Nenhum':
                    min_val = st.number_input(
                        f"Valor mínimo de {col_filtro}",
                        value=float(df_filtrado[col_filtro].min()),
                        min_value=float(df_filtrado[col_filtro].min()),
                        max_value=float(df_filtrado[col_filtro].max())
                    )
                    df_filtrado = df_filtrado[df_filtrado[col_filtro] >= min_val]
        
        with col_f3:
            # Limite de linhas
            limite_linhas = st.slider("Linhas para mostrar", 10, 1000, 100)
        
        # Mostrar dados
        st.subheader(f"📋 Dados ({len(df_filtrado):,} registros filtrados)")
        
        # Paginação
        total_pages = max(1, len(df_filtrado) // limite_linhas + 1)
        
        col_pg1, col_pg2, col_pg3 = st.columns([1, 2, 1])
        
        with col_pg1:
            page_number = st.number_input("Página", 1, total_pages, 1)
        
        with col_pg3:
            st.caption(f"Total: {len(df_filtrado):,} registros")
        
        # Calcular índice
        start_idx = (page_number - 1) * limite_linhas
        end_idx = min(start_idx + limite_linhas, len(df_filtrado))
        
        # Formatar DataFrame para exibição
        df_display = df_filtrado[colunas_vis].iloc[start_idx:end_idx].copy()
        
        # Formatar colunas numéricas
        for col in colunas_vis:
            if col in colunas_numericas:
                # Formatar como número com separadores
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) and not pd.isna(x) else x
                )
        
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400
        )
        
        # Download
        st.subheader("📥 Exportar Dados")
        
        csv = df_filtrado[colunas_vis].to_csv(index=False)
        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name=f"dados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# =============================================================================
# TAB 5: ANÁLISE ESTATÍSTICA AVANÇADA
# =============================================================================

with tab5:
    st.header("📐 Análise Estatística Avançada")
    
    if not colunas_numericas:
        st.warning("Nenhuma coluna numérica disponível para análise estatística")
    else:
        # Selecionar variáveis para análise
        st.subheader("🔧 Configurar Análise")
        
        col_analise1, col_analise2 = st.columns(2)
        
        with col_analise1:
            variavel_resposta = st.selectbox(
                "Variável Resposta (Y)",
                options=['Nenhuma'] + colunas_numericas,
                help="Variável que queremos explicar/prever"
            )
        
        with col_analise2:
            variaveis_explicativas = st.multiselect(
                "Variáveis Explicativas (X)",
                options=colunas_numericas,
                help="Variáveis que podem explicar a variável resposta"
            )
        
        # Remover variável resposta das explicativas se selecionada
        if variavel_resposta != 'Nenhuma' and variavel_resposta in variaveis_explicativas:
            variaveis_explicativas = [v for v in variaveis_explicativas if v != variavel_resposta]
        
        if variavel_resposta != 'Nenhuma' and variaveis_explicativas:
            # Preparar dados
            dados_analise = df[[variavel_resposta] + variaveis_explicativas].dropna()
            
            if len(dados_analise) < 10:
                st.warning("Dados insuficientes para análise (necessário pelo menos 10 observações)")
            else:
                st.success(f"✅ {len(dados_analise)} observações válidas para análise")
                
                # Análise de correlação
                st.subheader("🔥 Correlações com Variável Resposta")
                
                correlacoes = {}
                for var in variaveis_explicativas:
                    correlacao = dados_analise[variavel_resposta].corr(dados_analise[var])
                    if not pd.isna(correlacao):
                        correlacoes[var] = correlacao
                
                if correlacoes:
                    # Ordenar por força da correlação
                    df_correl = pd.DataFrame({
                        'Variável': list(correlacoes.keys()),
                        'Correlação': list(correlacoes.values()),
                        'Força': [abs(x) for x in correlacoes.values()]
                    }).sort_values('Força', ascending=False)
                    
                    # Adicionar cores
                    def color_correlation(val):
                        if abs(val) > 0.7:
                            return 'background-color: #10b981; color: white; font-weight: bold'
                        elif abs(val) > 0.5:
                            return 'background-color: #f59e0b; color: white; font-weight: bold'
                        elif abs(val) > 0.3:
                            return 'background-color: #ef4444; color: white; font-weight: bold'
                        return ''
                    
                    st.dataframe(
                        df_correl.style.map(
                            lambda x: color_correlation(x) if isinstance(x, (int, float)) else '',
                            subset=['Correlação']
                        ).format({'Correlação': '{:.3f}'}),
                        use_container_width=True
                    )
                    
                    # Gráfico de dispersão para as top 3 correlações
                    st.subheader("📈 Relações Principais")
                    
                    top_correl = df_correl.head(3)
                    
                    if not top_correl.empty:
                        cols_graf = st.columns(min(3, len(top_correl)))
                        
                        for idx, row in top_correl.iterrows():
                            with cols_graf[idx % len(cols_graf)]:
                                fig_scatter = px.scatter(
                                    dados_analise,
                                    x=row['Variável'],
                                    y=variavel_resposta,
                                    trendline="ols",
                                    title=f"{variavel_resposta} vs {row['Variável']}",
                                    labels={
                                        row['Variável']: row['Variável'],
                                        variavel_resposta: variavel_resposta
                                    }
                                )
                                fig_scatter.update_traces(marker=dict(size=5, opacity=0.6))
                                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Distribuição da variável resposta
                st.subheader(f"📊 Distribuição de {variavel_resposta}")
                
                col_dist1, col_dist2 = st.columns(2)
                
                with col_dist1:
                    # Histograma
                    fig_hist = px.histogram(
                        dados_analise,
                        x=variavel_resposta,
                        nbins=50,
                        title=f"Histograma de {variavel_resposta}"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col_dist2:
                    # Box plot
                    fig_box = px.box(
                        dados_analise,
                        y=variavel_resposta,
                        title=f"Box Plot de {variavel_resposta}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Análise de outliers
                st.subheader("📉 Detecção de Outliers")
                
                # Método IQR
                Q1 = dados_analise[variavel_resposta].quantile(0.25)
                Q3 = dados_analise[variavel_resposta].quantile(0.75)
                IQR = Q3 - Q1
                
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                
                outliers = dados_analise[
                    (dados_analise[variavel_resposta] < limite_inferior) | 
                    (dados_analise[variavel_resposta] > limite_superior)
                ]
                
                col_out1, col_out2, col_out3, col_out4 = st.columns(4)
                
                with col_out1:
                    st.metric("Q1 (25%)", f"{Q1:.2f}")
                
                with col_out2:
                    st.metric("Q3 (75%)", f"{Q3:.2f}")
                
                with col_out3:
                    st.metric("IQR", f"{IQR:.2f}")
                
                with col_out4:
                    st.metric("Outliers", f"{len(outliers)} ({len(outliers)/len(dados_analise)*100:.1f}%)")
                
                if not outliers.empty:
                    with st.expander("📋 Ver Outliers"):
                        st.dataframe(outliers, use_container_width=True)

# =============================================================================
# RODAPÉ
# =============================================================================

st.markdown("---")
st.caption(f"📊 Analytics Platform • {len(df):,} registros • {len(df.columns)} colunas • Atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M')}")
