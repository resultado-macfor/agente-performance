import re
import pandas as pd

from config.constants import PADROES_CLASSIFICACAO, CLIENTES_PADROES
from utils.helpers import identificar_colunas_numericas, format_currency, format_percentage


def extrair_categorias_campanha(nome_campanha):
    if not nome_campanha or pd.isna(nome_campanha):
        return {}

    nome_str = str(nome_campanha).upper()

    categorias = {
        'iniciativa': None,
        'produto': None,
        'cultura': None,
        'categoria': None,
        'tipo_campanha': None,
        'objetivo': None,
        'etapa_funil': None,
        'editoria': None,
        'po': None,
        'agencia': None,
        'plataforma': None,
        'cliente': None,
    }

    po_pattern = r'\bPO[_-]?(\d+)\b'
    po_match = re.search(po_pattern, nome_str, re.IGNORECASE)
    if po_match:
        categorias['po'] = f"PO{po_match.group(1)}"

    for agencia in PADROES_CLASSIFICACAO['agencia']:
        if agencia in nome_str:
            categorias['agencia'] = agencia
            break

    for plataforma in PADROES_CLASSIFICACAO['plataforma']:
        if plataforma in nome_str:
            categorias['plataforma'] = plataforma
            break

    for cultura in PADROES_CLASSIFICACAO['cultura']:
        if cultura in nome_str:
            categorias['cultura'] = cultura
            break

    for produto in PADROES_CLASSIFICACAO['produto']:
        if produto in nome_str:
            categorias['produto'] = produto
            break

    for tipo in PADROES_CLASSIFICACAO['tipo_campanha']:
        if tipo in nome_str:
            categorias['tipo_campanha'] = tipo
            break

    for objetivo in PADROES_CLASSIFICACAO['objetivo']:
        if objetivo in nome_str:
            categorias['objetivo'] = objetivo
            break

    for etapa in PADROES_CLASSIFICACAO['etapa_funil']:
        if etapa in nome_str:
            categorias['etapa_funil'] = etapa
            break

    separadores = ['_', '-', '|', ' ', '__']
    for sep in separadores:
        if sep in nome_str:
            partes = nome_str.split(sep)
            if len(partes) > 0:
                primeira_parte = partes[0]
                if len(primeira_parte) > 3 and primeira_parte not in PADROES_CLASSIFICACAO['plataforma']:
                    categorias['iniciativa'] = primeira_parte

    for cliente, padroes_cliente in CLIENTES_PADROES.items():
        for padrao in padroes_cliente:
            if padrao in nome_str:
                categorias['cliente'] = cliente
                break
        if categorias['cliente']:
            break

    return categorias


def classificar_campanhas_multi_cliente(df, coluna_campanha='campaign'):
    if df.empty or coluna_campanha not in df.columns:
        return df

    classificacoes = []

    for idx, row in df.iterrows():
        nome_campanha = row[coluna_campanha]
        categorias = extrair_categorias_campanha(nome_campanha)

        categorias_preenchidas = sum(1 for v in categorias.values() if v is not None)
        classificado = 'SIM' if categorias_preenchidas >= 3 else 'NÃO'

        classificacao = {
            'nome_campanha_original': nome_campanha,
            'classificado': classificado,
            'categorias_identificadas': categorias_preenchidas,
        }

        for chave, valor in categorias.items():
            classificacao[f'campaign_{chave}'] = valor

        classificacoes.append(classificacao)

    df_classificado = pd.DataFrame(classificacoes)

    if len(df_classificado) == len(df):
        df_resultado = df.copy()
        for col in df_classificado.columns:
            if col != 'nome_campanha_original':
                df_resultado[col] = df_classificado[col]
        return df_resultado

    return df


def calculate_mom_analysis(df, cliente, mes_atual, mes_anterior):
    if df.empty:
        return None

    if cliente != "Todos" and 'cliente_identificado' in df.columns:
        df_filtered = df[df['cliente_identificado'] == cliente].copy()
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        return None

    if 'date' not in df_filtered.columns:
        return None

    df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
    df_filtered['mes'] = df_filtered['date'].dt.to_period('M')

    df_mes_atual = df_filtered[df_filtered['mes'] == mes_atual]
    df_mes_anterior = df_filtered[df_filtered['mes'] == mes_anterior]

    analysis_results = {
        'cliente': cliente,
        'mes_atual': str(mes_atual),
        'mes_anterior': str(mes_anterior),
        'total_mes_atual': len(df_mes_atual),
        'total_mes_anterior': len(df_mes_anterior),
        'platform_analysis': {},
        'metric_analysis': {},
    }

    if 'datasource' in df_filtered.columns:
        platforms = df_filtered['datasource'].unique()

        for platform in platforms:
            platform_current = df_mes_atual[df_mes_atual['datasource'] == platform]
            platform_previous = df_mes_anterior[df_mes_anterior['datasource'] == platform]

            spend_current = 0
            spend_previous = 0

            spend_cols = [col for col in df_filtered.columns if 'spend' in col.lower() or 'cost' in col.lower()]
            if spend_cols:
                spend_col = spend_cols[0]
                spend_current = pd.to_numeric(platform_current[spend_col], errors='coerce').sum()
                spend_previous = pd.to_numeric(platform_previous[spend_col], errors='coerce').sum()

            analysis_results['platform_analysis'][platform] = {
                'spend_current': spend_current,
                'spend_previous': spend_previous,
                'spend_change': spend_current - spend_previous,
                'spend_change_pct': ((spend_current - spend_previous) / spend_previous * 100) if spend_previous > 0 else 0,
                'records_current': len(platform_current),
                'records_previous': len(platform_previous),
            }

    metric_cols = identificar_colunas_numericas(df_filtered)
    priority_metrics = ['spend', 'revenue', 'conversions', 'impressions', 'clicks', 'cpc', 'cpm', 'ctr', 'roas']

    for metric_name in priority_metrics:
        for col in metric_cols:
            if metric_name in col.lower():
                metric_current = pd.to_numeric(df_mes_atual[col], errors='coerce').sum()
                metric_previous = pd.to_numeric(df_mes_anterior[col], errors='coerce').sum()

                change = metric_current - metric_previous
                change_pct = (change / metric_previous * 100) if metric_previous > 0 else 0

                analysis_results['metric_analysis'][col] = {
                    'current': metric_current,
                    'previous': metric_previous,
                    'change': change,
                    'change_pct': change_pct,
                }
                break

    return analysis_results


def create_mom_table(analysis_results):
    if not analysis_results or 'platform_analysis' not in analysis_results:
        return None

    platform_data = analysis_results['platform_analysis']
    if not platform_data:
        return None

    platforms = list(platform_data.keys())

    table_data = {
        'Plataforma': platforms,
        f'Investimento {analysis_results["mes_anterior"]}': [],
        f'Investimento {analysis_results["mes_atual"]}': [],
        'Variação MoM': [],
        'Variação %': [],
        'Registros Anterior': [],
        'Registros Atual': [],
    }

    total_current = 0
    total_previous = 0

    for platform in platforms:
        data = platform_data[platform]

        table_data[f'Investimento {analysis_results["mes_anterior"]}'].append(format_currency(data['spend_previous']))
        table_data[f'Investimento {analysis_results["mes_atual"]}'].append(format_currency(data['spend_current']))
        table_data['Variação MoM'].append(format_currency(data['spend_change']))
        table_data['Variação %'].append(format_percentage(data['spend_change_pct']))
        table_data['Registros Anterior'].append(f"{data['records_previous']:,}")
        table_data['Registros Atual'].append(f"{data['records_current']:,}")

        total_current += data['spend_current']
        total_previous += data['spend_previous']

    table_data['Plataforma'].append('TOTAL')
    table_data[f'Investimento {analysis_results["mes_anterior"]}'].append(format_currency(total_previous))
    table_data[f'Investimento {analysis_results["mes_atual"]}'].append(format_currency(total_current))

    total_change = total_current - total_previous
    total_change_pct = (total_change / total_previous * 100) if total_previous > 0 else 0

    table_data['Variação MoM'].append(format_currency(total_change))
    table_data['Variação %'].append(format_percentage(total_change_pct))
    table_data['Registros Anterior'].append(f"{analysis_results['total_mes_anterior']:,}")
    table_data['Registros Atual'].append(f"{analysis_results['total_mes_atual']:,}")

    return pd.DataFrame(table_data)
