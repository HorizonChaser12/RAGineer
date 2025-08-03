import logging
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Union, Optional
from typing import List, Dict, Any, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
import datetime
import uvicorn
import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="azure_rag_system_api.log",
    filemode="a",
)

logger = logging.getLogger(__name__)

load_dotenv()

# API Provider Configuration
API_PROVIDER = os.getenv("API_PROVIDER", "azure").lower()  # 'azure' or 'google'

# Hugging Face configuration
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    logger.warning(
        "Hugging Face token not found in .env file. Sentence transformers and reranker may not work properly."
    )
else:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HUGGINGFACE_TOKEN

# Azure OpenAI environment variables
AZURE_CONFIG = {
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "llm_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    "embedding_deployment": os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS"),
    "embedding_api_version": os.getenv("AZURE_EMBEDDING_API_VERSION"),
    "embedding_endpoint": os.getenv("AZURE_EMBEDDING_ENDPOINT"),
}

# Google API environment variables
GOOGLE_CONFIG = {
    "api_key": os.getenv("GOOGLE_API_KEY"),
    "llm_model": "gemini-2.0-flash",
    "embedding_model": "gemini-embedding-001",
}


def validate_credentials():
    logger.info(f"Current API Provider: {API_PROVIDER}")
    if API_PROVIDER == "azure" and (
        not AZURE_CONFIG["api_key"] or not AZURE_CONFIG["endpoint"]
    ):
        logger.warning(
            "Azure OpenAI credentials not found in .env file. Ensure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set."
        )
        return False
    elif API_PROVIDER == "google" and not GOOGLE_CONFIG["api_key"]:
        logger.warning(
            "Google API credentials not found in .env file. Ensure GOOGLE_API_KEY is set."
        )
        return False
    elif API_PROVIDER == "google" and GOOGLE_CONFIG["api_key"]:
        logger.info("Google API credentials found.")
    return True


validate_credentials()


# --- Utility function for JSON serialization ---
def make_serializable(obj: Any) -> Any:
    """ "
    Recursively converts non-serializable objects (like datetime, numpy types)
    in a data structure to JSON-serializable types.
    """

    if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):

        return obj.isoformat()

    if isinstance(obj, dict):

        return {make_serializable(k): make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):

        return [make_serializable(i) for i in obj]

    if isinstance(obj, (np.ndarray, np.generic)):

        return obj.tolist()

    if isinstance(obj, torch.Tensor):

        return obj.detach().cpu().numpy().tolist()

    return obj


class EnhancedAdaptiveRAGSystem:

    def __init__(
        self,
        excel_file_path: str,
        temperature: float = 0.7,
        concise_prompt: bool = False,
        index_file: str = "faiss_index.bin",
        use_sentence_transformers: bool = True,
        use_reranker: bool = True,
        sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):

        self.excel_file_path = excel_file_path

        self.concise_prompt = concise_prompt

        self.index_file = index_file

        self.chunk_to_original_doc_mapping: List[int] = []

        self.use_sentence_transformers = use_sentence_transformers

        self.use_reranker = use_reranker

        # Initialize sentence transformer and reranker models

        if self.use_sentence_transformers:

            logger.info(
                f"Loading sentence transformer model: {sentence_transformer_model}"
            )

            try:
                if not HUGGINGFACE_TOKEN:
                    logger.warning(
                        "No Hugging Face token available. Model download may fail."
                    )

                self.sentence_transformer = SentenceTransformer(
                    sentence_transformer_model, token=HUGGINGFACE_TOKEN
                )

                logger.info("Sentence transformer model loaded successfully")

            except Exception as e:

                logger.error(f"Failed to load sentence transformer: {e}")

                self.sentence_transformer = None

                self.use_sentence_transformers = False

        else:

            self.sentence_transformer = None

        if self.use_reranker:

            logger.info(f"Loading reranker model: {reranker_model}")

            try:
                if not HUGGINGFACE_TOKEN:
                    logger.warning(
                        "No Hugging Face token available. Model download may fail."
                    )

                self.reranker = CrossEncoder(reranker_model, token=HUGGINGFACE_TOKEN)

                logger.info("Reranker model loaded successfully")

            except Exception as e:

                logger.error(f"Failed to load reranker: {e}")

                self.reranker = None

                self.use_reranker = False

        else:

            self.reranker = None

        # Initialize models based on configuration
        logger.info("Initializing AI models")

        try:
            if API_PROVIDER == "google":
                logger.info("Initializing Google Gemini models")
                logger.info(
                    f"Using Google API key (first 4 chars): {GOOGLE_CONFIG['api_key'][:4]}..."
                )
                try:
                    self.embedding_model = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",  # Required model name for Google's text embeddings
                        google_api_key=GOOGLE_CONFIG["api_key"],
                    )
                    logger.info("Google embeddings model initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Google embeddings model: {e}")
                    raise

                try:
                    self.llm = GoogleGenerativeAI(
                        model=GOOGLE_CONFIG["llm_model"],
                        google_api_key=GOOGLE_CONFIG["api_key"],
                        temperature=temperature,
                    )
                    logger.info("Google Gemini chat model initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Google chat model: {e}")
                    raise

                logger.info("Google Gemini Models initialized successfully.")
            else:
                logger.info("Initializing Azure OpenAI models")
                self.embedding_model = AzureOpenAIEmbeddings(
                    azure_deployment=AZURE_CONFIG["embedding_deployment"],
                    openai_api_version=AZURE_CONFIG["embedding_api_version"],
                    azure_endpoint=AZURE_CONFIG["embedding_endpoint"],
                    api_key=AZURE_CONFIG["api_key"],
                )

                self.llm = AzureChatOpenAI(
                    azure_deployment=AZURE_CONFIG["llm_deployment"],
                    openai_api_version=AZURE_CONFIG["api_version"],
                    azure_endpoint=AZURE_CONFIG["endpoint"],
                    api_key=AZURE_CONFIG["api_key"],
                    temperature=temperature,
                )
                logger.info("Azure OpenAI Models initialized successfully.")

        except Exception as e:

            logger.error(f"Fatal: Failed to initialize Azure OpenAI models: {e}.")

            self.embedding_model = None

            self.llm = None

        self.data = None

        self.metadata: Optional[List[Dict[Any, Any]]] = None

        self.index = None

        self.st_embeddings = None  # Store sentence transformer embeddings

        # Use 1536 for Azure OpenAI Ada-002 embeddings, but will be updated based on actual model

        self.dimension = 1536

        self.column_info: Dict[str, Dict[str, Any]] = {}

        if (self.embedding_model and self.llm) or (self.sentence_transformer):

            self._load_data()

            try:

                logger.info(f"Attempting to load FAISS index from {self.index_file}...")

                self._load_index(self.index_file)

                if (
                    not self.chunk_to_original_doc_mapping
                    and self.data is not None
                    and "combined_text" in self.data.columns
                ):

                    logger.info(
                        "Re-populating chunk_to_original_doc_mapping after loading index."
                    )

                    self._populate_chunk_mapping_from_data()

            except Exception as e:

                logger.warning(
                    f"Could not load persisted index from {self.index_file} (Reason: {e}). Building new index..."
                )

                self._build_index()

        else:

            logger.error(
                "Skipping data loading and index building due to model initialization failure."
            )

        if self.llm:

            self.response_template = """

   You are a helpful technical support assistant. Your goal is to provide comprehensive and accurate answers based on the information available.

   Here's an overview of the dataset you are working with:

   {dataset_overview}

   For the user's specific query, the following documents have been retrieved as potentially relevant:

   {retrieved_documents_context}

   Additionally, here's an analysis of patterns found within these retrieved documents:

   {pattern_analysis_summary}

   User Query: {query}

   Based on all the information above (the dataset overview, the specific retrieved documents, and the pattern analysis),

   provide a professional, conversational response that addresses the user's query.

   If the query is general, use the dataset overview more. If specific, focus on the retrieved documents.

   Ensure your response includes:

   1. A clear summary of the identified issue or topic from the query.

   2. Relevant information from the dataset, citing document IDs if referring to specific retrieved documents.

   3. Insights from past occurrences, root causes, and solutions if applicable and found in the data.

   4. Any common factors or patterns if they are significant.

   5. Preventative measures or recommendations if appropriate.

   Make your response easy to understand. Explain complex terms if necessary.

   Format your response in clear paragraphs.

   If a unrelevant question is asked for the data available then clearly say that you arent able to answer rather than hallucinating.

   If I ask any kind of question that is not related to the data or not even has a answer then simply tell that you can't get the data you are asking for.

   Response:

   """

            self.prompt = PromptTemplate(
                input_variables=[
                    "dataset_overview",
                    "retrieved_documents_context",
                    "pattern_analysis_summary",
                    "query",
                ],
                template=self.response_template,
                validate_template=True,
            )

            self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

            logger.info("LLMChain initialized with comprehensive prompt.")

        else:

            self.chain = None

            logger.error(
                "LLMChain could not be initialized because LLM is not available."
            )

    def _generate_dataset_overview_summary(self) -> str:
        """Generates a textual summary of the dataset's structure."""

        if self.data is None or self.metadata is None:

            return "Dataset information is currently unavailable."

        num_records = len(self.metadata)

        summary_parts = [
            f"The dataset contains {num_records} records (e.g., rows or entries)."
        ]

        if not self.column_info:

            summary_parts.append("Column details are not analyzed.")

            return "\n".join(summary_parts)

        summary_parts.append("It has the following columns:")

        for col_name, info in self.column_info.items():

            col_desc = f"- '{col_name}': Type: {info.get('data_type', 'N/A')}"

            if "semantic_type" in info:

                col_desc += f", Semantic Role: {info.get('semantic_type')}"

            if (
                "categories" in info
                and isinstance(info["categories"], list)
                and info["categories"]
            ):

                preview_cats = info["categories"][:3]

                etc_cats = "..." if len(info["categories"]) > 3 else ""

                col_desc += f" (e.g., {', '.join(map(str, preview_cats))}{etc_cats})"

            summary_parts.append(col_desc)

        return "\n".join(summary_parts)

    def _load_data(self):

        logger.info(f"Loading data from {self.excel_file_path}...")

        if not os.path.exists(self.excel_file_path):

            logger.error(f"Excel file not found: {self.excel_file_path}")

            self.data = pd.DataFrame(
                {"Error": [f"File not found: {self.excel_file_path}"]}
            )

            self._prepare_data()

            self.metadata = self.data.to_dict(orient="records")

            logger.warning("Proceeding with dummy data due to missing Excel file.")

            return

        try:

            excel_data = pd.read_excel(self.excel_file_path, sheet_name=None)

            self.data = None

            for sheet_name, df in excel_data.items():

                if not df.empty:

                    self.data = df

                    logger.info(
                        f"Using sheet '{sheet_name}' ({len(df)}x{len(df.columns)})"
                    )

                    break

            if self.data is None:

                logger.error("No non-empty sheets in Excel. Creating dummy data.")

                self.data = pd.DataFrame({"Error": ["No non-empty sheets in Excel."]})

            self._prepare_data()

            self.metadata = self.data.to_dict(orient="records")

            logger.info(f"Loaded {len(self.data)} records.")

        except Exception as e:

            logger.error(f"Error loading data: {e}", exc_info=True)

            self.data = pd.DataFrame({"Error": [f"Error loading data: {str(e)}"]})

            self._prepare_data()

            self.metadata = self.data.to_dict(orient="records")

    def _prepare_data(self):

        if self.data is None:

            logger.error("Cannot prepare data: self.data is None.")

            return

        self.data = self.data.fillna("")

        self.data = self.data.dropna(how="all").dropna(axis=1, how="all")

        self._analyze_columns()

        self.data["combined_text"] = self.data.apply(
            lambda row: " ".join(
                f"{col}: {val}"
                for col, val in row.items()
                if str(val).strip() != "" and col != "combined_text"
            ),
            axis=1,
        )

        logger.info("Data preparation complete. 'combined_text' created.")

    def _analyze_columns(self):

        if self.data is None:
            return

        logger.info("Analyzing data columns...")

        self.column_info = {}

        for col in self.data.columns:

            if col == "combined_text":
                continue

            col_data = self.data[col].astype(str)

            original_col_data = self.data[col]

            data_type = "text"

            if pd.api.types.is_numeric_dtype(original_col_data.infer_objects()):

                data_type = "numeric"

            elif self._is_date_column(original_col_data):

                data_type = "date"

            empty_count = (original_col_data.isna()).sum() + (
                original_col_data.astype(str) == ""
            ).sum()

            sparsity = empty_count / max(1, len(original_col_data))

            unique_values = original_col_data.nunique(dropna=False)

            value_diversity = unique_values / max(1, len(original_col_data))

            self.column_info[col] = {
                "data_type": data_type,
                "sparsity": sparsity,
                "value_diversity": value_diversity,
                "unique_values_count": unique_values,
            }

            if data_type == "text":

                avg_len = col_data.str.len().mean() if not col_data.empty else 0

                if value_diversity > 0.8 and unique_values > 0.8 * len(
                    original_col_data
                ):

                    self.column_info[col]["semantic_type"] = (
                        "identifier" if avg_len < 50 else "description"
                    )

                elif value_diversity < 0.2 and unique_values < 20:

                    self.column_info[col]["semantic_type"] = "category"

                    if unique_values > 0:

                        self.column_info[col]["categories"] = (
                            original_col_data.dropna().unique().tolist()
                            if unique_values < 20
                            else "Too many to list"
                        )

                else:

                    self.column_info[col]["semantic_type"] = "general_text"

            elif data_type == "date":

                self.column_info[col]["semantic_type"] = "date"

        logger.info(f"Column analysis complete: {self.column_info}")

    def _is_date_column(self, series: pd.Series) -> bool:

        if series.empty:
            return False

        try:

            non_null_series = series.dropna()

            if non_null_series.empty:
                return False

            sample_size = min(len(non_null_series), 20)

            sample = non_null_series.sample(sample_size)

            converted_sample = pd.to_datetime(sample, errors="coerce")

            success_rate = converted_sample.notna().mean()

            return success_rate > 0.7

        except Exception as e:

            logger.debug(f"Date column check error: {e}")

            return False

    def _populate_chunk_mapping_from_data(self):
        """Recreate the chunk mapping from data if index exists but mapping was lost"""

        if self.data is None or "combined_text" not in self.data.columns:

            logger.warning(
                "Cannot populate chunk mapping: data or combined_text column missing"
            )

            return

        self.chunk_to_original_doc_mapping = list(range(len(self.data)))

        logger.info(
            f"Populated chunk mapping with {len(self.chunk_to_original_doc_mapping)} entries"
        )

    def _build_index(self):
        """Build FAISS index and optionally sentence transformer embeddings"""

        if self.data is None or "combined_text" not in self.data.columns:

            logger.error("Cannot build index: data or combined_text column missing")

            return

        logger.info("Building indices from Excel data...")

        documents = self.data["combined_text"].tolist()

        self.chunk_to_original_doc_mapping = list(range(len(documents)))

        # Build sentence transformer embeddings if enabled

        if self.use_sentence_transformers and self.sentence_transformer:

            try:

                logger.info("Generating sentence transformer embeddings...")

                self.st_embeddings = self.sentence_transformer.encode(
                    documents, convert_to_tensor=False
                )

                logger.info(
                    f"Generated {len(self.st_embeddings)} sentence transformer embeddings"
                )

            except Exception as e:

                logger.error(f"Failed to generate sentence transformer embeddings: {e}")

                self.st_embeddings = None

        # Build Azure OpenAI FAISS index if available

        if self.embedding_model:

            try:

                logger.info("Generating embeddings...")

                embeddings = self.embedding_model.embed_documents(documents)

                self.dimension = len(embeddings[0])

                logger.info(
                    f"Creating FAISS index with embedding dimension: {self.dimension}"
                )

                self.index = faiss.IndexFlatL2(self.dimension)

                faiss.normalize_L2(np.array(embeddings, dtype=np.float32))

                self.index.add(np.array(embeddings, dtype=np.float32))

                logger.info(
                    f"FAISS index built successfully with {len(documents)} vectors"
                )

                try:

                    faiss.write_index(self.index, self.index_file)

                    logger.info(f"FAISS index persisted to {self.index_file}")

                except Exception as e:

                    logger.error(f"Failed to persist FAISS index: {e}")

            except Exception as e:

                logger.error(f"Failed to build FAISS index: {e}")

                self.index = None

    def _load_index(self, index_path: str):
        """Load a FAISS index from disk"""

        if not os.path.exists(index_path):

            raise FileNotFoundError(f"Index file not found: {index_path}")

        self.index = faiss.read_index(index_path)

        if self.index.ntotal == 0:

            raise ValueError(f"Loaded index is empty: {index_path}")

        self.dimension = self.index.d

        logger.info(
            f"Successfully loaded FAISS index from {index_path}. N_vectors: {self.index.ntotal}, Dimension: {self.dimension}"
        )

    def _sentence_transformer_retrieve(
        self, query: str, k: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve using sentence transformers"""

        if (
            not self.use_sentence_transformers
            or self.sentence_transformer is None
            or self.st_embeddings is None
        ):

            return []

        logger.info(f"Performing sentence transformer retrieval for: {query[:100]}...")

        try:

            # Encode query

            query_embedding = self.sentence_transformer.encode([query])

            # Calculate cosine similarities

            similarities = cosine_similarity(query_embedding, self.st_embeddings)[0]

            # Get top k results

            top_indices = np.argsort(similarities)[::-1][:k]

            retrieved_docs = []

            for i, idx in enumerate(top_indices):

                if idx >= len(self.chunk_to_original_doc_mapping):

                    continue

                doc_idx = self.chunk_to_original_doc_mapping[idx]

                if doc_idx >= len(self.metadata):

                    continue

                similarity_score = float(similarities[idx])

                doc_content = self.metadata[doc_idx].copy()

                if "combined_text" in doc_content:

                    del doc_content["combined_text"]

                retrieved_docs.append(
                    {
                        "id": doc_idx,
                        "content": doc_content,
                        "similarity": similarity_score,
                        "retrieval_method": "sentence_transformer",
                    }
                )

            logger.info(
                f"Sentence transformer retrieved {len(retrieved_docs)} documents."
            )

            return retrieved_docs

        except Exception as e:

            logger.error(
                f"Error during sentence transformer retrieval: {e}", exc_info=True
            )

            return []

    def _embeddings_retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using embeddings model and FAISS"""

        if self.index is None or self.embedding_model is None:

            return []

        logger.info(f"Performing Azure OpenAI retrieval for: {query[:100]}...")

        try:

            query_embedding = self.embedding_model.embed_query(query)

            query_embedding_np = np.array([query_embedding], dtype=np.float32)

            faiss.normalize_L2(query_embedding_np)

            distances, indices = self.index.search(
                query_embedding_np, min(k, self.index.ntotal)
            )

            retrieved_docs = []

            for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):

                if idx >= len(self.chunk_to_original_doc_mapping):

                    continue

                doc_idx = self.chunk_to_original_doc_mapping[idx]

                if doc_idx >= len(self.metadata):

                    continue

                similarity = 1.0 - min(1.0, float(distance) / 2.0)

                doc_content = self.metadata[doc_idx].copy()

                if "combined_text" in doc_content:

                    del doc_content["combined_text"]

                retrieved_docs.append(
                    {
                        "id": doc_idx,
                        "content": doc_content,
                        "similarity": similarity,
                        "retrieval_method": "azure_openai",
                    }
                )

            logger.info(f"Azure OpenAI retrieved {len(retrieved_docs)} documents.")

            return retrieved_docs

        except Exception as e:

            logger.error(f"Error during Azure OpenAI retrieval: {e}", exc_info=True)

            return []

    def _rerank_documents(
        self, query: str, documents: List[Dict[str, Any]], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Re-rank documents using cross-encoder"""

        if not self.use_reranker or self.reranker is None or not documents:

            return documents

        logger.info(f"Re-ranking {len(documents)} documents...")

        try:

            # Prepare query-document pairs for reranking

            query_doc_pairs = []

            for doc in documents:

                # Use combined_text if available, otherwise concatenate content

                if "combined_text" in self.metadata[doc["id"]]:

                    doc_text = self.metadata[doc["id"]]["combined_text"]

                else:

                    doc_text = " ".join(
                        f"{k}: {v}" for k, v in doc["content"].items() if str(v).strip()
                    )

                query_doc_pairs.append([query, doc_text])

            # Get reranking scores

            rerank_scores = self.reranker.predict(query_doc_pairs)

            # Add rerank scores to documents and sort

            for i, doc in enumerate(documents):

                doc["rerank_score"] = float(rerank_scores[i])

            # Sort by rerank score (higher is better for cross-encoder)

            reranked_docs = sorted(
                documents, key=lambda x: x["rerank_score"], reverse=True
            )

            # Limit to top_k if specified

            if top_k:

                reranked_docs = reranked_docs[:top_k]

            logger.info(
                f"Re-ranking complete. Top document rerank score: {reranked_docs[0]['rerank_score']:.4f}"
            )

            return reranked_docs

        except Exception as e:

            logger.error(f"Error during re-ranking: {e}", exc_info=True)

            return documents

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced retrieval combining multiple methods"""

        logger.info(f"Starting enhanced retrieval for query: {query[:100]}...")

        all_retrieved_docs = []

        # Method 1: Sentence Transformer retrieval

        if self.use_sentence_transformers:

            st_docs = self._sentence_transformer_retrieve(
                query, k * 2
            )  # Get more for diversity

            all_retrieved_docs.extend(st_docs)

        # Method 2: Embeddings-based retrieval

        embedding_docs = self._embeddings_retrieve(query, k * 2)

        all_retrieved_docs.extend(embedding_docs)

        # Remove duplicates based on document ID

        seen_ids = set()

        unique_docs = []

        for doc in all_retrieved_docs:

            if doc["id"] not in seen_ids:

                unique_docs.append(doc)

                seen_ids.add(doc["id"])

        logger.info(f"Combined retrieval found {len(unique_docs)} unique documents")

        # Re-rank if enabled

        if self.use_reranker:

            final_docs = self._rerank_documents(query, unique_docs, k)

        else:

            # Sort by similarity and take top k

            final_docs = sorted(
                unique_docs, key=lambda x: x.get("similarity", 0), reverse=True
            )[:k]

        logger.info(f"Final retrieval returned {len(final_docs)} documents")

        return final_docs

    def format_retrieved_document_for_llm(self, doc: Dict) -> str:
        """Format a retrieved document for inclusion in the LLM context"""

        formatted_content = [f"DOCUMENT ID: {doc['id']}"]

        # Add retrieval method and scores

        if "retrieval_method" in doc:

            formatted_content.append(f"Retrieval Method: {doc['retrieval_method']}")

        if "similarity" in doc:

            formatted_content.append(f"Similarity Score: {doc['similarity']:.4f}")

        if "rerank_score" in doc:

            formatted_content.append(f"Rerank Score: {doc['rerank_score']:.4f}")

        if "content" not in doc or not doc["content"]:

            return (
                "\n".join(formatted_content)
                + "\nNo content available for this document."
            )

        for key, value in doc["content"].items():

            value_str = str(value) if value is not None else ""

            if value_str.strip():

                formatted_key = key.replace("_", " ").title()

                formatted_content.append(f"{formatted_key}: {value_str}")

        return "\n".join(formatted_content)

    def analyze_patterns(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:

        if not retrieved_docs:
            return {"count": 0, "patterns": {}, "date_range": None}

        analysis = {"count": len(retrieved_docs), "patterns": {}, "date_range": None}

        for col_name, info in self.column_info.items():

            if info.get("semantic_type") == "category" or (
                info.get("data_type") == "text"
                and info.get("value_diversity", 1.0) < 0.5
            ):

                value_counts = {}

                for doc in retrieved_docs:

                    value = doc["content"].get(col_name)

                    if value is not None and str(value).strip():

                        value_str = str(value)

                        value_counts[value_str] = value_counts.get(value_str, 0) + 1

                if value_counts:
                    analysis["patterns"][col_name] = value_counts

            if info.get("data_type") == "date":

                dates = []

                for doc in retrieved_docs:

                    date_val = doc["content"].get(col_name)

                    if date_val:

                        try:
                            dt = pd.to_datetime(date_val, errors="coerce")

                        except:
                            dt = None

                        if pd.notna(dt):
                            dates.append(dt)

                if dates:

                    min_date, max_date = min(dates), max(dates)

                    if analysis["date_range"] is None:

                        analysis["date_range"] = {
                            "column": col_name,
                            "min_date": min_date.strftime("%Y-%m-%d"),
                            "max_date": max_date.strftime("%Y-%m-%d"),
                            "span_days": (max_date - min_date).days,
                        }

        logger.info(f"Pattern analysis complete: {analysis}")

        return analysis

    def generate_response(self, query: str, k: int = 3) -> Dict[str, Any]:

        if not self.chain:

            logger.error("Cannot generate response: LLM chain not initialized.")

            return {
                "response": "System error: Unable to process request.",
                "retrieved_docs": [],
                "pattern_analysis": {"count": 0},
            }

        logger.info(f"Generating enhanced response for query: {query[:100]}..., k={k}")

        dataset_overview_summary = self._generate_dataset_overview_summary()

        retrieved_docs = self.retrieve(query, k)

        if retrieved_docs:

            retrieved_documents_llm_context = "\n\n===\n\n".join(
                [self.format_retrieved_document_for_llm(doc) for doc in retrieved_docs]
            )

        else:

            retrieved_documents_llm_context = (
                "No specific documents were found to be highly relevant to this query."
            )

            logger.warning("No relevant documents found for the query to pass to LLM.")

        pattern_analysis = self.analyze_patterns(retrieved_docs)

        pattern_analysis_llm_summary_parts = [
            "Summary of Patterns Found in Retrieved Documents:"
        ]

        if pattern_analysis["count"] > 0:

            pattern_analysis_llm_summary_parts.append(
                f"- Number of similar records found: {pattern_analysis['count']}"
            )

            if pattern_analysis.get("date_range"):

                dr = pattern_analysis["date_range"]

                pattern_analysis_llm_summary_parts.append(
                    f"- These records span from {dr['min_date']} to {dr['max_date']} ({dr['span_days']} days) in the '{dr['column']}' field"
                )

            if pattern_analysis.get("patterns"):

                pattern_analysis_llm_summary_parts.append(
                    "- Common patterns identified:"
                )

                for field, value_counts in pattern_analysis["patterns"].items():

                    top_values = sorted(
                        value_counts.items(), key=lambda x: x[1], reverse=True
                    )[:3]

                    field_display = field.replace("_", " ").title()

                    values_display = ", ".join(
                        [f"{val} ({count}x)" for val, count in top_values]
                    )

                    pattern_analysis_llm_summary_parts.append(
                        f" * {field_display}: {values_display}"
                    )

        else:

            pattern_analysis_llm_summary_parts.append(
                "- No specific patterns identified in the retrieved documents."
            )

        pattern_analysis_llm_summary = "\n".join(pattern_analysis_llm_summary_parts)

        try:

            logger.info("Calling LLM chain to generate response...")

            response = self.chain.run(
                dataset_overview=dataset_overview_summary,
                retrieved_documents_context=retrieved_documents_llm_context,
                pattern_analysis_summary=pattern_analysis_llm_summary,
                query=query,
            )

            logger.info(
                f"LLM response generated successfully. Length: {len(response)} characters"
            )

        except Exception as e:

            logger.error(f"Error generating LLM response: {e}", exc_info=True)

            response = f"I apologize, but I encountered an error while processing your query: {str(e)}"

        # Prepare response with serializable data

        serializable_retrieved_docs = make_serializable(retrieved_docs)

        serializable_pattern_analysis = make_serializable(pattern_analysis)

        return {
            "response": response,
            "retrieved_docs": serializable_retrieved_docs,
            "pattern_analysis": serializable_pattern_analysis,
            "dataset_overview": dataset_overview_summary,
            "query": query,
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information"""

        status = {
            "embedding_model_ready": self.embedding_model is not None,
            "llm_ready": self.llm is not None,
            "data_loaded": self.data is not None and not self.data.empty,
            "faiss_index_ready": self.index is not None,
            "sentence_transformer_ready": self.sentence_transformer is not None,
            "reranker_ready": self.reranker is not None,
            "total_documents": len(self.metadata) if self.metadata else 0,
            "index_dimension": self.dimension,
            "features_enabled": {
                "sentence_transformers": self.use_sentence_transformers,
                "reranker": self.use_reranker,
                "azure_openai": self.embedding_model is not None
                and self.llm is not None,
            },
        }

        if self.data is not None:

            status["data_shape"] = list(self.data.shape)

            status["columns"] = list(self.data.columns)

            status["column_info"] = self.column_info

        if self.index:

            status["faiss_index_size"] = self.index.ntotal

        if self.st_embeddings is not None:

            status["sentence_transformer_embeddings_count"] = len(self.st_embeddings)

        return make_serializable(status)

    def _perform_data_analysis(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform statistical analysis on the dataframe based on the query"""
        try:
            analysis_results = {
                "metrics": {},
                "summary": {},
                "sample_data": None
            }

            # Parse the query to identify key terms
            query_lower = query.lower()
            query_year = None
            
            # Extract year if present in query
            import re
            year_match = re.search(r'\b20\d{2}\b', query)
            if year_match:
                query_year = int(year_match.group())
            
            # Check if query is about specific component
            if 'mobile app' in query_lower:
                component_filter = df['Component'] == 'Mobile App'
                filtered_df = df[component_filter]
            else:
                filtered_df = df

            # Basic dataset metrics
            analysis_results["metrics"] = {
                "total_rows": len(filtered_df),
                "total_columns": len(filtered_df.columns),
                "complete_records": filtered_df.dropna().shape[0],
                "matching_year_records": 0,  # Will be updated if year filter applies
                "unique_values": {
                    col: int(filtered_df[col].nunique()) 
                    for col in filtered_df.columns
                }
            }

            # Date analysis and time-based filtering
            date_cols = ['Defect Log Date', 'Defect Resolution Date']
            date_analysis = {}
            year_counts = {}
            
            for col in date_cols:
                if col in filtered_df.columns:
                    dates = pd.to_datetime(filtered_df[col], errors='coerce')
                    if not dates.empty:
                        min_date = dates.min()
                        max_date = dates.max()
                        
                        # Basic date info
                        date_info = {
                            "min_date": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
                            "max_date": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None,
                            "range_days": int((max_date - min_date).days) if pd.notna(max_date) and pd.notna(min_date) else 0
                        }
                        
                        # Year distribution
                        year_dist = dates.dt.year.value_counts()
                        date_info["year_distribution"] = {str(k): int(v) for k, v in year_dist.items()}
                        
                        # Update matching year records if query includes year
                        if query_year and str(query_year) in date_info["year_distribution"]:
                            analysis_results["metrics"]["matching_year_records"] = date_info["year_distribution"][str(query_year)]
                        
                        date_analysis[col] = date_info
                        year_counts[col] = {str(k): int(v) for k, v in year_dist.items()}
                        date_analysis[col] = {
                            "min_date": dates.min().strftime("%Y-%m-%d"),
                            "max_date": dates.max().strftime("%Y-%m-%d"),
                            "range_days": int((dates.max() - dates.min()).days),
                            "year_distribution": dates.dt.year.value_counts().to_dict()
                        }
                        # Count defects by year
                        year_counts[col] = dates.dt.year.value_counts().to_dict()

            analysis_results["summary"]["dates"] = date_analysis
            analysis_results["summary"]["year_counts"] = year_counts

            # Categorical analysis with counts
            important_cats = ['Component', 'Prict', 'Sev', 'Containment Phase', 'Root Cause']
            cat_analysis = {}
            
            for col in important_cats:
                if col in filtered_df.columns:
                    counts = filtered_df[col].value_counts()
                    cat_analysis[col] = {
                        "counts": counts.to_dict(),
                        "total": len(counts),
                        "top_5": counts.head().to_dict()
                    }

            analysis_results["summary"]["categorical"] = cat_analysis

            # Sample data (relevant to query)
            if 'mobile app' in query_lower:
                analysis_results["sample_data"] = filtered_df.head(5).to_dict('records')
            else:
                analysis_results["sample_data"] = df.head(5).to_dict('records')

            return analysis_results

        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            return {"error": str(e)}

    def _generate_analytical_context(self, df: pd.DataFrame, analysis_results: Dict, query: str) -> str:
        """Generate structured context from analysis results for LLM"""
        context_parts = []

        metrics = analysis_results.get('metrics', {})
        summary = analysis_results.get('summary', {})

        # Dataset overview
        context_parts.append(f"Dataset Overview:")
        context_parts.append(f"- Total records analyzed: {metrics.get('total_rows', 0)}")
        context_parts.append(f"- Complete records: {metrics.get('complete_records', 0)}")
        if metrics.get('matching_year_records', 0) > 0:
            context_parts.append(f"- Matching year records: {metrics['matching_year_records']}")

        # Date ranges and year distribution
        if 'dates' in summary:
            context_parts.append("\nTemporal Analysis:")
            for col, date_info in summary['dates'].items():
                context_parts.append(f"\n{col}:")
                context_parts.append(f"- Date range: {date_info['min_date']} to {date_info['max_date']}")
                context_parts.append("- Year distribution:")
                for year, count in date_info.get('year_distribution', {}).items():
                    context_parts.append(f"  * {year}: {count} records")

        # Category distributions
        if 'categorical' in summary:
            context_parts.append("\nCategory Analysis:")
            for col, cat_info in summary['categorical'].items():
                if col == 'Component' or 'mobile app' in query.lower():
                    context_parts.append(f"\n{col}:")
                    for category, count in cat_info['counts'].items():
                        if 'mobile app' in query.lower() and 'mobile app' in category.lower():
                            context_parts.append(f"- {category}: {count} records")
                        elif not 'mobile app' in query.lower():
                            context_parts.append(f"- {category}: {count} records")

        return "\n".join(context_parts)

    def _generate_analytical_response(self, query: str, context: str) -> str:
        """Generate LLM response for analytical mode"""
        try:
            analytical_prompt = f"""
            You are a data analyst assistant. Based on the following data analysis and user query,
            provide a clear, concise response that directly addresses the query using the available data insights.

            Data Analysis Context:
            {context}

            User Query: {query}

            Guidelines for your response:
            1. For time-based queries:
               - First check if we have data for the specific time period
               - If we have partial data for the period, specify exactly what we have
               - Provide the exact count for the available period
            2. For component-specific queries:
               - Give the exact count of incidents/defects
               - Specify the time period the count represents
            3. Always include specific numbers and dates when available
            4. If data is completely unavailable for the requested period, say so directly
            
            Format your response like this:
            "Found [X] matching records in [specific time period]. [Optional: Brief breakdown if relevant]"
            OR
            "No data available for [requested period]. Available data covers [actual period] with [X] matching records."

            Response:
            """

            response = self.llm.predict(analytical_prompt)
            return response

        except Exception as e:
            logger.error(f"Error generating analytical response: {e}")
            return f"Error generating analysis response: {str(e)}"

    def analyze_data_directly(
        self, query: str, excel_file_path: str = None, sheet_name: str = None
    ) -> Dict[str, Any]:
        """
        Perform direct data analysis using Pandas instead of document retrieval.
        This bypasses the RAG pipeline and provides structured data insights.
        Uses the system's already loaded data by default.
        """

        logger.info(f"Starting analytical thinking mode for query: {query[:100]}...")

        try:
            # Use the system's already loaded data instead of loading new data
            if self.data is None or self.data.empty:
                raise ValueError(
                    "No data available in the system. Please initialize the system first with data."
                )

            # Use the system's current data
            df = self.data.copy()

            # Remove the combined_text column for analysis as it's just a concatenation
            if "combined_text" in df.columns:
                df = df.drop("combined_text", axis=1)

            logger.info(
                f"Using system's loaded data: {len(df)} rows, {len(df.columns)} columns"
            )

            # Analyze the query to determine what kind of analysis to perform
            analysis_results = self._perform_data_analysis(df, query)

            # Generate structured context for LLM
            structured_context = self._generate_analytical_context(
                df, analysis_results, query
            )

            # Generate LLM response using analytical context
            if self.llm:
                analytical_response = self._generate_analytical_response(
                    query, structured_context
                )
            else:
                analytical_response = "LLM not available for response generation."

            return {
                "response": analytical_response,
                "analytical_results": analysis_results,
                "data_summary": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "columns": list(df.columns),
                    "source": "System loaded data",
                    "file_path": self.excel_file_path,
                },
                "query": query,
                "mode": "analytical_thinking",
            }

        except Exception as e:

            logger.error(f"Error in analytical thinking mode: {e}", exc_info=True)

            return {
                "response": f"Error in analytical mode: {str(e)}. Please ensure the system is properly initialized with data.",
                "analytical_results": {"error": str(e)},
                "query": query,
                "mode": "analytical_thinking",
            }
