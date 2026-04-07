import json
import os
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

from agent.campaign_classifier import classificar_campanhas_multi_cliente


@st.cache_resource
def get_bigquery_client():
    try:
        service_account_info = None

        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            service_account_info = dict(st.secrets["gcp_service_account"])
            if isinstance(service_account_info.get("private_key"), str):
                service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

        elif all(key in os.environ for key in ['type', 'project_id', 'private_key', 'client_email', 'token_uri']):
            service_account_info = {
                "type": os.environ['type'],
                "project_id": os.environ['project_id'],
                "private_key_id": os.environ.get('private_key_id', ''),
                "private_key": os.environ['private_key'].replace('\\n', '\n'),
                "client_email": os.environ['client_email'],
                "client_id": os.environ.get('client_id', ''),
                "auth_uri": os.environ.get('auth_uri', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.environ['token_uri'],
                "auth_provider_x509_cert_url": os.environ.get('auth_provider_x509_cert_url', 'https://www.googleapis.com/oauth2/v1/certs'),
                "client_x509_cert_url": os.environ.get('client_x509_cert_url', ''),
                "universe_domain": os.environ.get('universe_domain', 'googleapis.com')
            }

        elif 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
            credentials_json = os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON']
            service_account_info = json.loads(credentials_json)

        else:
            st.error("❌ Credenciais não encontradas!")
            return None

        if not service_account_info:
            st.error("❌ Não foi possível obter as credenciais")
            return None

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        client = bigquery.Client(
            credentials=credentials,
            project=service_account_info["project_id"]
        )

        return client

    except Exception as e:
        st.error(f"❌ Erro na conexão: {str(e)}")
        return None


@st.cache_data(ttl=3600)
def load_all_columns_data(_client, data_inicio=None, data_fim=None, data_sources=None, filtro_cliente="Todos", limit=50000):
    try:
        st.info("🔍 Carregando dados...")

        query = """
        SELECT *
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

        query += " ORDER BY date DESC"

        df = _client.query(query).to_dataframe()

        if df.empty:
            st.warning("Nenhum dado encontrado")
            return pd.DataFrame()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        def identificar_cliente_por_account_name(account_name):
            if pd.isna(account_name):
                return "Outros"

            account_str = str(account_name).strip()
            account_upper = account_str.upper()

            if "SYNGENTA" in account_upper:
                return "Syngenta"

            return account_str

        if 'account_name' in df.columns:
            df['cliente_identificado'] = df['account_name'].apply(identificar_cliente_por_account_name)

            if filtro_cliente != "Todos":
                df = df[df['cliente_identificado'] == filtro_cliente].copy()
        else:
            df['cliente_identificado'] = "Desconhecido"

        df = df.head(limit)
        df_classificado = classificar_campanhas_multi_cliente(df)

        return df_classificado

    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return pd.DataFrame()
