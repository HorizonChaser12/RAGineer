import logging
import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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

 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',

 filename='azure_rag_system_api.log',

 filemode='a'

)

logger = logging.getLogger(__name__)

load_dotenv()

# Load Azure OpenAI environment variables

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_EMBEDDINGS")

AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")

if not AZURE_API_KEY or not AZURE_ENDPOINT:

 logger.warning("Azure OpenAI credentials not found in .env file. Ensure AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set.")

# --- Utility function for JSON serialization ---

def make_serializable(obj: Any) -> Any:

 """

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

 def __init__(self, excel_file_path: str, temperature: float = 0.7, concise_prompt: bool = False,

     index_file: str = "Azure_Implementation/faiss_index_azure.bin",

     use_sentence_transformers: bool = True, use_reranker: bool = True,

     sentence_transformer_model: str = "all-MiniLM-L6-v2",

     reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):

  self.excel_file_path = excel_file_path

  self.concise_prompt = concise_prompt

  self.index_file = index_file

  self.chunk_to_original_doc_mapping: List[int] = []

  self.use_sentence_transformers = use_sentence_transformers

  self.use_reranker = use_reranker

  # Initialize sentence transformer and reranker models

  if self.use_sentence_transformers:

   logger.info(f"Loading sentence transformer model: {sentence_transformer_model}")

   try:

    self.sentence_transformer = SentenceTransformer(sentence_transformer_model)

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

    self.reranker = CrossEncoder(reranker_model)

    logger.info("Reranker model loaded successfully")

   except Exception as e:

    logger.error(f"Failed to load reranker: {e}")

    self.reranker = None

    self.use_reranker = False

  else:

   self.reranker = None

  # Initialize Azure OpenAI models

  logger.info(f"Initializing Azure OpenAI models")

  try:

   self.embedding_model = AzureOpenAIEmbeddings(

    azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,

    openai_api_version=AZURE_EMBEDDING_API_VERSION,

    azure_endpoint=AZURE_EMBEDDING_ENDPOINT,

    api_key=AZURE_API_KEY

   )

   self.llm = AzureChatOpenAI(

    azure_deployment=AZURE_LLM_DEPLOYMENT,

    openai_api_version=AZURE_API_VERSION,

    azure_endpoint=AZURE_ENDPOINT,

    api_key=AZURE_API_KEY,

    temperature=temperature

   )

   logger.info("Azure OpenAI Models initialized successfully.")

  except Exception as e:

   logger.error(f"Fatal: Failed to initialize Azure OpenAI models: {e}.")

   self.embedding_model = None

   self.llm = None

  self.data = None

  self.metadata: Optional[List[Dict[Any, Any]]] = None

  self.index = None

  self.st_embeddings = None # Store sentence transformer embeddings

  # Use 1536 for Azure OpenAI Ada-002 embeddings, but will be updated based on actual model

  self.dimension = 1536

  self.column_info: Dict[str, Dict[str, Any]] = {}

  if (self.embedding_model and self.llm) or (self.sentence_transformer):

   self._load_data()

   try:

    logger.info(f"Attempting to load FAISS index from {self.index_file}...")

    self._load_index(self.index_file)

    if not self.chunk_to_original_doc_mapping and self.data is not None and 'combined_text' in self.data.columns:

     logger.info("Re-populating chunk_to_original_doc_mapping after loading index.")

     self._populate_chunk_mapping_from_data()

   except Exception as e:

    logger.warning(f"Could not load persisted index from {self.index_file} (Reason: {e}). Building new index...")

    self._build_index()

  else:

   logger.error("Skipping data loading and index building due to model initialization failure.")

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

    input_variables=["dataset_overview", "retrieved_documents_context", "pattern_analysis_summary", "query"],

    template=self.response_template,

    validate_template=True

   )

   self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

   logger.info("LLMChain initialized with comprehensive prompt.")

  else:

   self.chain = None

   logger.error("LLMChain could not be initialized because LLM is not available.")

 def _generate_dataset_overview_summary(self) -> str:

  """Generates a textual summary of the dataset's structure."""

  if self.data is None or self.metadata is None:

   return "Dataset information is currently unavailable."

  num_records = len(self.metadata)

  summary_parts = [f"The dataset contains {num_records} records (e.g., rows or entries)."]

  if not self.column_info:

   summary_parts.append("Column details are not analyzed.")

   return "\n".join(summary_parts)

  summary_parts.append("It has the following columns:")

  for col_name, info in self.column_info.items():

   col_desc = f"- '{col_name}': Type: {info.get('data_type', 'N/A')}"

   if 'semantic_type' in info:

    col_desc += f", Semantic Role: {info.get('semantic_type')}"

   if 'categories' in info and isinstance(info['categories'], list) and info['categories']:

    preview_cats = info['categories'][:3]

    etc_cats = "..." if len(info['categories']) > 3 else ""

    col_desc += f" (e.g., {', '.join(map(str, preview_cats))}{etc_cats})"

   summary_parts.append(col_desc)

  return "\n".join(summary_parts)

 def _load_data(self):

  logger.info(f"Loading data from {self.excel_file_path}...")

  if not os.path.exists(self.excel_file_path):

   logger.error(f"Excel file not found: {self.excel_file_path}")

   self.data = pd.DataFrame({'Error': [f'File not found: {self.excel_file_path}']})

   self._prepare_data()

   self.metadata = self.data.to_dict(orient='records')

   logger.warning("Proceeding with dummy data due to missing Excel file.")

   return

  try:

   excel_data = pd.read_excel(self.excel_file_path, sheet_name=None)

   self.data = None

   for sheet_name, df in excel_data.items():

    if not df.empty:

     self.data = df

     logger.info(f"Using sheet '{sheet_name}' ({len(df)}x{len(df.columns)})")

     break

   if self.data is None:

    logger.error("No non-empty sheets in Excel. Creating dummy data.")

    self.data = pd.DataFrame({'Error': ['No non-empty sheets in Excel.']})

   self._prepare_data()

   self.metadata = self.data.to_dict(orient='records')

   logger.info(f"Loaded {len(self.data)} records.")

  except Exception as e:

   logger.error(f"Error loading data: {e}", exc_info=True)

   self.data = pd.DataFrame({'Error': [f'Error loading data: {str(e)}']})

   self._prepare_data()

   self.metadata = self.data.to_dict(orient='records')

 def _prepare_data(self):

  if self.data is None:

   logger.error("Cannot prepare data: self.data is None.")

   return

  self.data = self.data.fillna('')

  self.data = self.data.dropna(how='all').dropna(axis=1, how='all')

  self._analyze_columns()

  self.data['combined_text'] = self.data.apply(

   lambda row: ' '.join(f"{col}: {val}" for col, val in row.items() if str(val).strip() != '' and col != 'combined_text'),

   axis=1

  )

  logger.info("Data preparation complete. 'combined_text' created.")

 def _analyze_columns(self):

  if self.data is None: return

  logger.info("Analyzing data columns...")

  self.column_info = {}

  for col in self.data.columns:

   if col == 'combined_text': continue

   col_data = self.data[col].astype(str)

   original_col_data = self.data[col]

   data_type = 'text'

   if pd.api.types.is_numeric_dtype(original_col_data.infer_objects()):

    data_type = 'numeric'

   elif self._is_date_column(original_col_data):

    data_type = 'date'

   empty_count = (original_col_data.isna()).sum() + (original_col_data.astype(str) == '').sum()

   sparsity = empty_count / max(1, len(original_col_data))

   unique_values = original_col_data.nunique(dropna=False)

   value_diversity = unique_values / max(1, len(original_col_data))

   self.column_info[col] = {

    'data_type': data_type, 'sparsity': sparsity,

    'value_diversity': value_diversity, 'unique_values_count': unique_values

   }

   if data_type == 'text':

    avg_len = col_data.str.len().mean() if not col_data.empty else 0

    if value_diversity > 0.8 and unique_values > 0.8 * len(original_col_data):

     self.column_info[col]['semantic_type'] = 'identifier' if avg_len < 50 else 'description'

    elif value_diversity < 0.2 and unique_values < 20:

     self.column_info[col]['semantic_type'] = 'category'

     if unique_values > 0:

      self.column_info[col]['categories'] = original_col_data.dropna().unique().tolist() if unique_values < 20 else 'Too many to list'

    else:

     self.column_info[col]['semantic_type'] = 'general_text'

   elif data_type == 'date':

    self.column_info[col]['semantic_type'] = 'date'

  logger.info(f"Column analysis complete: {self.column_info}")

 def _is_date_column(self, series: pd.Series) -> bool:

  if series.empty: return False

  try:

   non_null_series = series.dropna()

   if non_null_series.empty: return False

   sample_size = min(len(non_null_series), 20)

   sample = non_null_series.sample(sample_size)

   converted_sample = pd.to_datetime(sample, errors='coerce')

   success_rate = converted_sample.notna().mean()

   return success_rate > 0.7

  except Exception as e:

   logger.debug(f"Date column check error: {e}")

   return False

 def _populate_chunk_mapping_from_data(self):

  """Recreate the chunk mapping from data if index exists but mapping was lost"""

  if self.data is None or 'combined_text' not in self.data.columns:

   logger.warning("Cannot populate chunk mapping: data or combined_text column missing")

   return

  self.chunk_to_original_doc_mapping = list(range(len(self.data)))

  logger.info(f"Populated chunk mapping with {len(self.chunk_to_original_doc_mapping)} entries")

 def _build_index(self):

  """Build FAISS index and optionally sentence transformer embeddings"""

  if self.data is None or 'combined_text' not in self.data.columns:

   logger.error("Cannot build index: data or combined_text column missing")

   return

  logger.info("Building indices from Excel data...")

  documents = self.data['combined_text'].tolist()

  self.chunk_to_original_doc_mapping = list(range(len(documents)))

  # Build sentence transformer embeddings if enabled

  if self.use_sentence_transformers and self.sentence_transformer:

   try:

    logger.info("Generating sentence transformer embeddings...")

    self.st_embeddings = self.sentence_transformer.encode(documents, convert_to_tensor=False)

    logger.info(f"Generated {len(self.st_embeddings)} sentence transformer embeddings")

   except Exception as e:

    logger.error(f"Failed to generate sentence transformer embeddings: {e}")

    self.st_embeddings = None

  # Build Azure OpenAI FAISS index if available

  if self.embedding_model:

   try:

    logger.info("Generating Azure OpenAI embeddings...")

    embeddings = self.embedding_model.embed_documents(documents)

    self.dimension = len(embeddings[0])

    logger.info(f"Creating FAISS index with embedding dimension: {self.dimension}")

    self.index = faiss.IndexFlatL2(self.dimension)

    faiss.normalize_L2(np.array(embeddings, dtype=np.float32))

    self.index.add(np.array(embeddings, dtype=np.float32))

    logger.info(f"FAISS index built successfully with {len(documents)} vectors")

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

  logger.info(f"Successfully loaded FAISS index from {index_path}. N_vectors: {self.index.ntotal}, Dimension: {self.dimension}")

 def _sentence_transformer_retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:

  """Retrieve using sentence transformers"""

  if not self.use_sentence_transformers or self.sentence_transformer is None or self.st_embeddings is None:

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

    if 'combined_text' in doc_content:

     del doc_content['combined_text']

    retrieved_docs.append({

     "id": doc_idx,

     "content": doc_content,

     "similarity": similarity_score,

     "retrieval_method": "sentence_transformer"

    })

   logger.info(f"Sentence transformer retrieved {len(retrieved_docs)} documents.")

   return retrieved_docs

  except Exception as e:

   logger.error(f"Error during sentence transformer retrieval: {e}", exc_info=True)

   return []

 def _azure_openai_retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:

  """Retrieve using Azure OpenAI embeddings and FAISS"""

  if self.index is None or self.embedding_model is None:

   return []

  logger.info(f"Performing Azure OpenAI retrieval for: {query[:100]}...")

  try:

   query_embedding = self.embedding_model.embed_query(query)

   query_embedding_np = np.array([query_embedding], dtype=np.float32)

   faiss.normalize_L2(query_embedding_np)

   distances, indices = self.index.search(query_embedding_np, min(k, self.index.ntotal))

   retrieved_docs = []

   for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):

    if idx >= len(self.chunk_to_original_doc_mapping):

     continue

    doc_idx = self.chunk_to_original_doc_mapping[idx]

    if doc_idx >= len(self.metadata):

     continue

    similarity = 1.0 - min(1.0, float(distance) / 2.0)

    doc_content = self.metadata[doc_idx].copy()

    if 'combined_text' in doc_content:

     del doc_content['combined_text']

    retrieved_docs.append({

     "id": doc_idx,

     "content": doc_content,

     "similarity": similarity,

     "retrieval_method": "azure_openai"

    })

   logger.info(f"Azure OpenAI retrieved {len(retrieved_docs)} documents.")

   return retrieved_docs

  except Exception as e:

   logger.error(f"Error during Azure OpenAI retrieval: {e}", exc_info=True)

   return []

 def _rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:

  """Re-rank documents using cross-encoder"""

  if not self.use_reranker or self.reranker is None or not documents:

   return documents

  logger.info(f"Re-ranking {len(documents)} documents...")

  try:

   # Prepare query-document pairs for reranking

   query_doc_pairs = []

   for doc in documents:

    # Use combined_text if available, otherwise concatenate content

    if 'combined_text' in self.metadata[doc['id']]:

     doc_text = self.metadata[doc['id']]['combined_text']

    else:

     doc_text = ' '.join(f"{k}: {v}" for k, v in doc['content'].items() if str(v).strip())

    query_doc_pairs.append([query, doc_text])

   # Get reranking scores

   rerank_scores = self.reranker.predict(query_doc_pairs)

   # Add rerank scores to documents and sort

   for i, doc in enumerate(documents):

    doc['rerank_score'] = float(rerank_scores[i])

   # Sort by rerank score (higher is better for cross-encoder)

   reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

   # Limit to top_k if specified

   if top_k:

    reranked_docs = reranked_docs[:top_k]

   logger.info(f"Re-ranking complete. Top document rerank score: {reranked_docs[0]['rerank_score']:.4f}")

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

   st_docs = self._sentence_transformer_retrieve(query, k * 2) # Get more for diversity

   all_retrieved_docs.extend(st_docs)

  # Method 2: Azure OpenAI retrieval

  azure_docs = self._azure_openai_retrieve(query, k * 2)

  all_retrieved_docs.extend(azure_docs)

  # Remove duplicates based on document ID

  seen_ids = set()

  unique_docs = []

  for doc in all_retrieved_docs:

   if doc['id'] not in seen_ids:

    unique_docs.append(doc)

    seen_ids.add(doc['id'])

  logger.info(f"Combined retrieval found {len(unique_docs)} unique documents")

  # Re-rank if enabled

  if self.use_reranker:

   final_docs = self._rerank_documents(query, unique_docs, k)

  else:

   # Sort by similarity and take top k

   final_docs = sorted(unique_docs, key=lambda x: x.get('similarity', 0), reverse=True)[:k]

  logger.info(f"Final retrieval returned {len(final_docs)} documents")

  return final_docs

 def format_retrieved_document_for_llm(self, doc: Dict) -> str:

  """Format a retrieved document for inclusion in the LLM context"""

  formatted_content = [f"DOCUMENT ID: {doc['id']}"]

  # Add retrieval method and scores

  if 'retrieval_method' in doc:

   formatted_content.append(f"Retrieval Method: {doc['retrieval_method']}")

  if 'similarity' in doc:

   formatted_content.append(f"Similarity Score: {doc['similarity']:.4f}")

  if 'rerank_score' in doc:

   formatted_content.append(f"Rerank Score: {doc['rerank_score']:.4f}")

  if 'content' not in doc or not doc['content']:

   return "\n".join(formatted_content) + "\nNo content available for this document."

  for key, value in doc["content"].items():

   value_str = str(value) if value is not None else ""

   if value_str.strip():

    formatted_key = key.replace('_', ' ').title()

    formatted_content.append(f"{formatted_key}: {value_str}")

  return "\n".join(formatted_content)

 def analyze_patterns(self, retrieved_docs: List[Dict]) -> Dict[str, Any]:

  if not retrieved_docs: return {"count": 0, "patterns": {}, "date_range": None}

  analysis = {"count": len(retrieved_docs), "patterns": {}, "date_range": None}

  for col_name, info in self.column_info.items():

   if info.get('semantic_type') == 'category' or (info.get('data_type') == 'text' and info.get('value_diversity', 1.0) < 0.5):

    value_counts = {}

    for doc in retrieved_docs:

     value = doc["content"].get(col_name)

     if value is not None and str(value).strip():

      value_str = str(value)

      value_counts[value_str] = value_counts.get(value_str, 0) + 1

    if value_counts: analysis["patterns"][col_name] = value_counts

   if info.get('data_type') == 'date':

    dates = []

    for doc in retrieved_docs:

     date_val = doc["content"].get(col_name)

     if date_val:

      try: dt = pd.to_datetime(date_val, errors='coerce')

      except: dt = None

      if pd.notna(dt): dates.append(dt)

    if dates:

     min_date, max_date = min(dates), max(dates)

     if analysis["date_range"] is None:

      analysis["date_range"] = {

       "column": col_name, "min_date": min_date.strftime("%Y-%m-%d"),

       "max_date": max_date.strftime("%Y-%m-%d"), "span_days": (max_date - min_date).days

      }

  logger.info(f"Pattern analysis complete: {analysis}")

  return analysis

 def generate_response(self, query: str, k: int = 3) -> Dict[str, Any]:

  if not self.chain:

   logger.error("Cannot generate response: LLM chain not initialized.")

   return {"response": "System error: Unable to process request.", "retrieved_docs": [], "pattern_analysis": {"count": 0}}

  logger.info(f"Generating enhanced response for query: {query[:100]}..., k={k}")

  dataset_overview_summary = self._generate_dataset_overview_summary()

  retrieved_docs = self.retrieve(query, k)

  if retrieved_docs:

   retrieved_documents_llm_context = "\n\n===\n\n".join([self.format_retrieved_document_for_llm(doc) for doc in retrieved_docs])

  else:

   retrieved_documents_llm_context = "No specific documents were found to be highly relevant to this query."

   logger.warning("No relevant documents found for the query to pass to LLM.")

  pattern_analysis = self.analyze_patterns(retrieved_docs)

  pattern_analysis_llm_summary_parts = ["Summary of Patterns Found in Retrieved Documents:"]

  if pattern_analysis["count"] > 0:

   pattern_analysis_llm_summary_parts.append(f"- Number of similar records found: {pattern_analysis['count']}")

   if pattern_analysis.get("date_range"):

    dr = pattern_analysis["date_range"]

    pattern_analysis_llm_summary_parts.append(f"- These records span from {dr['min_date']} to {dr['max_date']} ({dr['span_days']} days) in the '{dr['column']}' field")

   if pattern_analysis.get("patterns"):

    pattern_analysis_llm_summary_parts.append("- Common patterns identified:")

    for field, value_counts in pattern_analysis["patterns"].items():

     top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:3]

     field_display = field.replace('_', ' ').title()

     values_display = ", ".join([f"{val} ({count}x)" for val, count in top_values])

     pattern_analysis_llm_summary_parts.append(f" * {field_display}: {values_display}")

  else:

   pattern_analysis_llm_summary_parts.append("- No specific patterns identified in the retrieved documents.")

  pattern_analysis_llm_summary = "\n".join(pattern_analysis_llm_summary_parts)

  try:

   logger.info("Calling LLM chain to generate response...")

   response = self.chain.run(

    dataset_overview=dataset_overview_summary,

    retrieved_documents_context=retrieved_documents_llm_context,

    pattern_analysis_summary=pattern_analysis_llm_summary,

    query=query

   )

   logger.info(f"LLM response generated successfully. Length: {len(response)} characters")

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

   "query": query

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

    "azure_openai": self.embedding_model is not None and self.llm is not None

   }

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

 def analyze_data_directly(self, query: str, excel_file_path: str = None, sheet_name: str = None) -> Dict[str, Any]:

  """

  Perform direct data analysis using Pandas instead of document retrieval.

  This bypasses the RAG pipeline and provides structured data insights.

  Uses the system's already loaded data by default.

  """

  logger.info(f"Starting analytical thinking mode for query: {query[:100]}...")

  try:

   # Use the system's already loaded data instead of loading new data

   if self.data is None or self.data.empty:

    raise ValueError("No data available in the system. Please initialize the system first with data.")

   # Use the system's current data

   df = self.data.copy()

   # Remove the combined_text column for analysis as it's just a concatenation

   if 'combined_text' in df.columns:

    df = df.drop('combined_text', axis=1)

   logger.info(f"Using system's loaded data: {len(df)} rows, {len(df.columns)} columns")

   # Analyze the query to determine what kind of analysis to perform

   analysis_results = self._perform_data_analysis(df, query)

   # Generate structured context for LLM

   structured_context = self._generate_analytical_context(df, analysis_results, query)

   # Generate LLM response using analytical context

   if self.llm:

    analytical_response = self._generate_analytical_response(query, structured_context)

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

     "file_path": self.excel_file_path

    },

    "query": query,

    "mode": "analytical_thinking"

   }

  except Exception as e:

   logger.error(f"Error in analytical thinking mode: {e}", exc_info=True)

   return {

    "response": f"Error in analytical mode: {str(e)}. Please ensure the system is properly initialized with data.",

    "analytical_results": {"error": str(e)},

    "query": query,

    "mode": "analytical_thinking"

   }

# FastAPI Application

app = FastAPI(

 title="Enhanced Azure RAG System API",

 description="Advanced RAG system with multiple retrieval methods and re-ranking",

 version="2.0.0"

)

app.add_middleware(

 CORSMiddleware,

 allow_origins=["*"],

 allow_credentials=True,

 allow_methods=["*"],

 allow_headers=["*"],

)

# Global variable to hold the RAG system instance

rag_system: Optional[EnhancedAdaptiveRAGSystem] = None

# Pydantic models for API requests

class QueryRequest(BaseModel):

 query: str

 k: Optional[int] = 3

 temperature: Optional[float] = None

 analytical_mode: Optional[bool] = False

 excel_file_path: Optional[str] = None

 sheet_name: Optional[str] = None

class InitializeRequest(BaseModel):

 excel_file_path: str

 temperature: Optional[float] = 0.7

 concise_prompt: Optional[bool] = False

 index_file: Optional[str] = "Azure_Implementation/faiss_index_azure.bin"

 use_sentence_transformers: Optional[bool] = True

 use_reranker: Optional[bool] = True

 sentence_transformer_model: Optional[str] = "all-MiniLM-L6-v2"

 reranker_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class RebuildIndexRequest(BaseModel):

 force_rebuild: Optional[bool] = False

# Exception handler for better error responses

@app.exception_handler(HTTPException)

async def http_exception_handler(request: Request, exc: HTTPException):

 return JSONResponse(

  status_code=exc.status_code,

  content={"detail": exc.detail, "status_code": exc.status_code}

 )

@app.exception_handler(Exception)

async def general_exception_handler(request: Request, exc: Exception):

 logger.error(f"Unhandled exception: {exc}", exc_info=True)

 return JSONResponse(

  status_code=500,

  content={"detail": "Internal server error", "error": str(exc)}

 )

@app.post("/initialize")

async def initialize_system(request: InitializeRequest):

 """Initialize the RAG system with specified configuration"""

 global rag_system

 try:

  logger.info(f"Initializing RAG system with file: {request.excel_file_path}")

  rag_system = EnhancedAdaptiveRAGSystem(

   excel_file_path=request.excel_file_path,

   temperature=request.temperature,

   concise_prompt=request.concise_prompt,

   index_file=request.index_file,

   use_sentence_transformers=request.use_sentence_transformers,

   use_reranker=request.use_reranker,

   sentence_transformer_model=request.sentence_transformer_model,

   reranker_model=request.reranker_model

  )

  status = rag_system.get_system_status()

  logger.info("RAG system initialized successfully")

  return {

   "message": "RAG system initialized successfully",

   "status": status

  }

 except Exception as e:

  logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)

  raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@app.post("/query")

async def query_system(request: QueryRequest):

 """Query the RAG system with optional analytical mode"""

 global rag_system

 if rag_system is None:

  raise HTTPException(status_code=400, detail="RAG system not initialized. Please call /initialize first.")

 try:

  logger.info(f"Processing query: {request.query[:100]}... (Analytical mode: {request.analytical_mode})")

  # Update temperature if provided

  if request.temperature is not None and rag_system.llm:

   rag_system.llm.temperature = request.temperature

  # Choose processing mode

  if request.analytical_mode:

   # Use analytical thinking mode with system's data

   response = rag_system.analyze_data_directly(query=request.query)

   logger.info("Query processed in analytical thinking mode using system data")

  else:

   # Use traditional RAG mode

   response = rag_system.generate_response(request.query, request.k)

   logger.info("Query processed in traditional RAG mode")

  return response

 except Exception as e:

  logger.error(f"Error processing query: {e}", exc_info=True)

  raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/rebuild-index")

async def rebuild_index(request: RebuildIndexRequest):

 """Rebuild the FAISS index and sentence transformer embeddings"""

 global rag_system

 if rag_system is None:

  raise HTTPException(status_code=400, detail="RAG system not initialized. Please call /initialize first.")

 try:

  logger.info("Rebuilding indices...")

  rag_system._build_index()

  status = rag_system.get_system_status()

  logger.info("Indices rebuilt successfully")

  return {

   "message": "Indices rebuilt successfully",

   "status": status

  }

 except Exception as e:

  logger.error(f"Failed to rebuild indices: {e}", exc_info=True)

  raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")

@app.get("/status")

async def get_status():

 """Get system status information"""

 global rag_system

 if rag_system is None:

  return {

   "initialized": False,

   "message": "RAG system not initialized"

  }

 try:

  status = rag_system.get_system_status()

  status["initialized"] = True

  return status

 except Exception as e:

  logger.error(f"Error getting status: {e}", exc_info=True)

  raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.post("/retrieve")

async def retrieve_documents(request: QueryRequest):

 """Retrieve relevant documents without generating a response"""

 global rag_system

 if rag_system is None:

  raise HTTPException(status_code=400, detail="RAG system not initialized. Please call /initialize first.")

 # Analytical mode doesn't support document retrieval

 if request.analytical_mode:

  raise HTTPException(status_code=400, detail="Document retrieval not available in analytical mode. Use /query endpoint instead.")

 try:

  logger.info(f"Retrieving documents for query: {request.query[:100]}...")

  retrieved_docs = rag_system.retrieve(request.query, request.k)

  pattern_analysis = rag_system.analyze_patterns(retrieved_docs)

  return {

   "query": request.query,

   "retrieved_docs": make_serializable(retrieved_docs),

   "pattern_analysis": make_serializable(pattern_analysis),

   "count": len(retrieved_docs)

  }

 except Exception as e:

  logger.error(f"Error retrieving documents: {e}", exc_info=True)

  raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")

@app.get("/health")

async def health_check():

 """Simple health check endpoint"""

 return {

  "status": "healthy",

  "timestamp": datetime.datetime.now().isoformat(),

  "system_initialized": rag_system is not None

 }

@app.get("/")

async def root():

 """Root endpoint with API information"""

 return {

  "message": "Enhanced Azure RAG System API",

  "version": "2.0.0",

  "features": [

   "Azure OpenAI Integration",

   "Sentence Transformers",

   "Cross-Encoder Re-ranking",

   "FAISS Vector Search",

   "Pattern Analysis"

  ],

  "endpoints": {

   "POST /initialize": "Initialize the RAG system",

   "POST /query": "Query the system for responses",

   "POST /retrieve": "Retrieve relevant documents only",

   "POST /rebuild-index": "Rebuild search indices",

   "GET /status": "Get system status",

   "GET /health": "Health check"

  }

 }

if __name__ == "__main__":

 logger.info("Starting Enhanced Azure RAG System API server...")

 uvicorn.run(

  "rag_system:app",

  host="0.0.0.0",

  port=8000,

  log_level="info",

  access_log=True,

  reload=True

 )