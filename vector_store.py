import logging
import os

import pandas as pd
import streamlit as st
from graphrag.model.community_report import CommunityReport
from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_relationships, read_indexer_text_units, \
    read_indexer_communities, embed_community_reports, read_indexer_reports
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings, \
    read_community_reports
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.vector_stores.base import BaseVectorStore, VectorStoreDocument
from graphrag.vector_stores.lancedb import LanceDBVectorStore

LANCEDB_URI = "lancedb"
# Constants and configurations

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"


def get_embedder():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    text_embedder = OpenAIEmbedding(
        api_key=openai_api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model="text-embedding-3-small",
        deployment_name="text-embedding-3-small",
        max_retries=20,
    )
    return text_embedder


def embedding_reports(reports_df, nodes_df, content_embedding_col: str = "full_content_embedding"):
    nodes_df.loc[:, "community"] = nodes_df["community"].fillna(-1)
    nodes_df.loc[:, "community"] = nodes_df["community"].astype(int)

    nodes_df = nodes_df.groupby(["title"]).agg({"community": "max"}).reset_index()
    filtered_community_df = nodes_df["community"].drop_duplicates()

    # todo: pre 1.0 back-compat where community was a string
    reports_df.loc[:, "community"] = reports_df["community"].fillna(-1)
    reports_df.loc[:, "community"] = reports_df["community"].astype(int)

    reports_df = reports_df.merge(
        filtered_community_df, on="community", how="inner"
    )

    embedder = get_embedder()

    reports_df = embed_community_reports(
        reports_df, embedder, embedding_col=content_embedding_col
    )
    return read_community_reports(
        df=reports_df,
        id_col="id",
        short_id_col="community",
        summary_embedding_col=None,
        content_embedding_col=content_embedding_col,
    )


@st.cache_data(show_spinner=False)
def load_data(input_dir, community_level):
    """Load all required data from parquet files."""
    entity_df = pd.read_parquet(f"{input_dir}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{input_dir}/{ENTITY_EMBEDDING_TABLE}.parquet")
    report_df = pd.read_parquet(f"{input_dir}/{COMMUNITY_REPORT_TABLE}.parquet")
    relationship_df = pd.read_parquet(f"{input_dir}/{RELATIONSHIP_TABLE}.parquet")
    text_unit_df = pd.read_parquet(f"{input_dir}/{TEXT_UNIT_TABLE}.parquet")
    communities_df = pd.read_parquet(f"{input_dir}/create_final_communities.parquet")

    entities = read_indexer_entities(entity_df, entity_embedding_df, community_level)
    relationships = read_indexer_relationships(relationship_df)
    text_units = read_indexer_text_units(text_unit_df)
    communities = read_indexer_communities(
        final_communities=communities_df,
        final_nodes=entity_df,  # entity_df contains the node information
        final_community_reports=report_df
    )

    report_embedding_store = LanceDBVectorStore(collection_name="report_description_embeddings")
    report_embedding_store.connect(db_uri=LANCEDB_URI)
    if check_collection_exists(report_embedding_store):
        # no more need to embeddings again
        logging.info("No need to embedding reports, directly read from vector store")
        reports_embedding = read_indexer_reports(report_df, entity_df, community_level, True)
    else:
        reports_embedding = embedding_reports(report_df, entity_df)

    return entities, communities, reports_embedding, relationships, text_units


def store_reports_semantic_embeddings(
    reports: list[CommunityReport],
    vectorstore: BaseVectorStore,
) -> BaseVectorStore:
    """Store entity semantic embeddings in a vectorstore."""
    documents = [
        VectorStoreDocument(
            id=report.id,
            text=report.full_content,
            vector=report.full_content_embedding,
            attributes=(
                {"title": report.title, "community_id": report.community_id}
            ),
        )
        for report in reports
    ]
    vectorstore.load_documents(documents=documents)
    return vectorstore


@st.cache_resource(show_spinner=False)
def setup_vector_store(input_dir, community_level):
    """Set up and initialize vector store.
    Checks if collections exist before rebuilding them to avoid unnecessary recomputation.
    """
    description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=LANCEDB_URI)

    report_embedding_store = LanceDBVectorStore(collection_name="report_description_embeddings")
    report_embedding_store.connect(db_uri=LANCEDB_URI)

    entities, _, reports, _, _ = load_data(input_dir, community_level)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities,
        vectorstore=description_embedding_store
    )

    return description_embedding_store, entity_description_embeddings


def check_collection_exists(vector_store: LanceDBVectorStore) -> bool:
    """
    Check if a LanceDB collection exists and has data.

    Args:
        vector_store: LanceDBVectorStore instance to check

    Returns:
        bool: True if collection exists and has data, False otherwise
    """
    try:
        # Try to access the collection and check if it has any data
        sample = vector_store.document_collection.head(1)
        return len(sample) > 0
    except Exception as e:
        # If collection doesn't exist or any other error occurs, return False
        logging.debug(f"Collection check failed for {vector_store.collection_name}: {str(e)}")
        return False
