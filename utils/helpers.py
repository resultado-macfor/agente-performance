import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def safe_metric(label, value, delta=None):
    try:
        if pd.isna(value):
            value = 0

        if isinstance(value, (int, np.integer)):
            display_value = f"{int(value):,}"
        elif isinstance(value, (float, np.floating)):
            if abs(value) < 0.01:
                display_value = f"{value:.4f}"
            elif abs(value) < 1:
                display_value = f"{value:.3f}"
            elif abs(value) < 1000:
                display_value = f"{value:.2f}"
            else:
                display_value = f"{value:,.0f}"
        else:
            try:
                num_val = float(value)
                display_value = f"{num_val:,.2f}"
            except:
                display_value = str(value)

        if delta is not None:
            if pd.isna(delta):
                delta = None
            elif isinstance(delta, (int, float, np.integer, np.floating)):
                delta = f"{delta:+.2f}"

        return st.metric(label, display_value, delta=delta)

    except:
        return st.metric(label, "Erro")


def identificar_colunas_numericas(df):
    if df.empty:
        return []

    colunas_numericas = []

    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                colunas_numericas.append(col)
            else:
                amostra = df[col].dropna().head(10)
                if len(amostra) > 0:
                    try:
                        pd.to_numeric(amostra)
                        colunas_numericas.append(col)
                    except:
                        pass
        except:
            continue

    return colunas_numericas


def analisar_coluna(df, coluna):
    if df.empty or coluna not in df.columns:
        return None

    try:
        dados_coluna = df[coluna]

        total = int(len(dados_coluna))
        nao_nulos = int(dados_coluna.notna().sum())
        nulos = int(dados_coluna.isna().sum())
        valores_unicos = int(dados_coluna.nunique())

        if total > 0:
            percentual_nulos = float((nulos / total) * 100)
        else:
            percentual_nulos = 0.0

        analise = {
            'nome': coluna,
            'tipo': str(dados_coluna.dtype),
            'total': total,
            'nao_nulos': nao_nulos,
            'nulos': nulos,
            'percentual_nulos': percentual_nulos,
            'valores_unicos': valores_unicos
        }

        if pd.api.types.is_numeric_dtype(dados_coluna):
            dados_validos = dados_coluna.dropna()
            if len(dados_validos) > 0:
                analise.update({
                    'tipo_detalhado': 'Numérica',
                    'min': float(dados_validos.min()),
                    'max': float(dados_validos.max()),
                    'media': float(dados_validos.mean()),
                    'mediana': float(dados_validos.median()),
                    'desvio_padrao': float(dados_validos.std()),
                    'q1': float(dados_validos.quantile(0.25)),
                    'q3': float(dados_validos.quantile(0.75))
                })
            else:
                analise.update({'tipo_detalhado': 'Numérica (vazia)'})
        elif dados_coluna.dtype == 'object':
            value_counts = dados_coluna.value_counts()
            analise.update({
                'tipo_detalhado': 'Texto/Categórica',
                'valores_mais_comuns': value_counts.head(10).to_dict(),
                'valor_mais_frequente': value_counts.index[0] if len(value_counts) > 0 else None,
                'frequencia_valor_mais_comum': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            })
        elif pd.api.types.is_datetime64_any_dtype(dados_coluna):
            dados_validos = dados_coluna.dropna()
            if len(dados_validos) > 0:
                analise.update({
                    'tipo_detalhado': 'Data',
                    'data_minima': dados_validos.min(),
                    'data_maxima': dados_validos.max(),
                    'intervalo_dias': int((dados_validos.max() - dados_validos.min()).days)
                })
            else:
                analise.update({'tipo_detalhado': 'Data (vazia)'})
        else:
            analise.update({'tipo_detalhado': 'Outro'})

        return analise

    except Exception as e:
        return {
            'nome': coluna,
            'tipo': 'Erro',
            'tipo_detalhado': f'Erro na análise: {str(e)[:50]}',
            'total': 0,
            'nao_nulos': 0,
            'nulos': 0,
            'percentual_nulos': 0.0,
            'valores_unicos': 0
        }


def criar_visualizacao_coluna(df, coluna):
    if df.empty or coluna not in df.columns:
        return None

    try:
        dados = df[coluna].dropna()

        if len(dados) == 0:
            return None

        if pd.api.types.is_numeric_dtype(df[coluna]):
            dados_numeric = pd.to_numeric(dados, errors='coerce').dropna()
            if len(dados_numeric) == 0:
                return None

            fig = px.histogram(
                x=dados_numeric,
                nbins=min(50, len(dados_numeric)),
                title=f"Distribuição de {coluna}",
                marginal="box"
            )
            return fig

        elif df[coluna].nunique() <= 50:
            contagem = df[coluna].value_counts().head(20)
            if len(contagem) == 0:
                return None

            fig = px.bar(
                x=contagem.index.astype(str),
                y=contagem.values,
                title=f"Top 20 Valores em {coluna}",
                labels={'x': coluna, 'y': 'Contagem'}
            )
            fig.update_xaxes(tickangle=45)
            return fig

        elif pd.api.types.is_datetime64_any_dtype(df[coluna]):
            try:
                dados_dt = pd.to_datetime(dados, errors='coerce').dropna()
                if len(dados_dt) == 0:
                    return None

                contagem_diaria = pd.Series(dados_dt.dt.date).value_counts().sort_index().reset_index()
                contagem_diaria.columns = ['data', 'contagem']

                fig = px.line(
                    contagem_diaria,
                    x='data',
                    y='contagem',
                    title=f"Frequência por Data - {coluna}"
                )
                return fig
            except:
                return None

        return None
    except Exception as e:
        st.error(f"Erro ao criar visualização: {str(e)[:100]}")
        return None


def format_currency(value):
    try:
        if pd.isna(value):
            return "R$ 0"
        return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return f"R$ {value}"


def format_percentage(value):
    try:
        if pd.isna(value):
            return "0.00%"
        return f"{value:.2f}%".replace(".", ",")
    except:
        return f"{value}%"
