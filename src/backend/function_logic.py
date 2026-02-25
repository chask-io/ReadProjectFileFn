"""
ReadProjectFileFn - Business Logic

3 modes:
1. List files in a project
2. Read a specific file's content (max 8000 chars)
3. Semantic search via RAG/Pinecone

Uses direct REST API calls to OpenAI and Pinecone to avoid numpy dependency.
"""

import json
import logging
import ssl
import urllib.request
import uuid as uuid_module
from typing import Dict, Any, List

from chask_foundation.backend.models import OrchestrationEvent
from chask_foundation.configs.global_config import (
    get_openai_api_key,
    get_pinecone_credentials,
    PINECONE_INDEX,
)
from api.files_requests import files_api_manager
from api.orchestrator_requests import orchestrator_api_manager

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_CONTENT_LENGTH = 8000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NAMESPACE_PREFIX = "project-files"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536


def _https_request(url: str, data: dict, headers: dict) -> dict:
    """Make an HTTPS POST request and return parsed JSON response."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _https_get(url: str, headers: dict) -> dict:
    """Make an HTTPS GET request and return parsed JSON response."""
    req = urllib.request.Request(url, headers=headers, method="GET")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


class FunctionBackend:
    """Backend for ReadProjectFileFn with 3 modes: list, read, RAG search."""

    def __init__(self, orchestration_event: OrchestrationEvent):
        self.orchestration_event = orchestration_event
        self._openai_api_key = None
        self._pinecone_api_key = None
        self._pinecone_host = None

    @property
    def openai_api_key(self) -> str:
        if self._openai_api_key is None:
            self._openai_api_key = get_openai_api_key()
        return self._openai_api_key

    @property
    def pinecone_api_key(self) -> str:
        if self._pinecone_api_key is None:
            creds = get_pinecone_credentials()
            self._pinecone_api_key = creds["api_key"]
        return self._pinecone_api_key

    @property
    def pinecone_host(self) -> str:
        if self._pinecone_host is None:
            self._pinecone_host = self._get_pinecone_host()
        return self._pinecone_host

    def _get_pinecone_host(self) -> str:
        """Discover the Pinecone index host via the control plane API."""
        resp = _https_get(
            f"https://api.pinecone.io/indexes/{PINECONE_INDEX}",
            {"Api-Key": self.pinecone_api_key, "Accept": "application/json"},
        )
        host = resp.get("host")
        if not host:
            raise Exception(f"Could not discover Pinecone host for index {PINECONE_INDEX}")
        logger.info(f"Pinecone host discovered: {host}")
        return host

    # ── OpenAI Embeddings (direct REST) ──────────────────────────────

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI REST API."""
        resp = _https_request(
            "https://api.openai.com/v1/embeddings",
            {"input": texts, "model": EMBEDDING_MODEL},
            {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json",
            },
        )
        data = resp.get("data", [])
        data.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    # ── Pinecone (direct REST) ───────────────────────────────────────

    def _pinecone_headers(self) -> dict:
        return {
            "Api-Key": self.pinecone_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _pinecone_upsert(self, vectors: List[dict], namespace: str) -> None:
        """Upsert vectors to Pinecone via REST API."""
        _https_request(
            f"https://{self.pinecone_host}/vectors/upsert",
            {"vectors": vectors, "namespace": namespace},
            self._pinecone_headers(),
        )

    def _pinecone_query(
        self, vector: List[float], top_k: int, namespace: str,
        filter: dict = None,
    ) -> List[dict]:
        """Query Pinecone via REST API."""
        payload = {
            "vector": vector,
            "topK": top_k,
            "namespace": namespace,
            "includeMetadata": True,
        }
        if filter:
            payload["filter"] = filter
        resp = _https_request(
            f"https://{self.pinecone_host}/query",
            payload,
            self._pinecone_headers(),
        )
        return resp.get("matches", [])

    def _pinecone_namespace_exists(self, namespace: str) -> bool:
        """Check if a Pinecone namespace exists via describe_index_stats."""
        try:
            resp = _https_request(
                f"https://{self.pinecone_host}/describe_index_stats",
                {},
                self._pinecone_headers(),
            )
            namespaces = resp.get("namespaces", {})
            return namespace in namespaces
        except Exception as e:
            logger.warning(f"Error checking namespace existence: {e}")
            return False

    # ── Business Logic ───────────────────────────────────────────────

    def process_request(self) -> str:
        tool_args = self._extract_tool_args()

        project_uuid = tool_args.get("project_uuid") or self._resolve_project_uuid()
        if not project_uuid:
            raise ValueError(
                "Could not determine project_uuid. Either pass it as a parameter "
                "or ensure the orchestration session is linked to a project."
            )
        self._validate_uuid(project_uuid, "project_uuid")

        file_uuid = tool_args.get("file_uuid")
        query = tool_args.get("query")
        force_reindex = tool_args.get("force_reindex", False)
        top_k = tool_args.get("top_k", 5)

        if query:
            return self._handle_rag_query(project_uuid, query, top_k, force_reindex)
        elif file_uuid:
            self._validate_uuid(file_uuid, "file_uuid")
            return self._handle_file_read(project_uuid, file_uuid, force_reindex)
        else:
            return self._handle_list_files(project_uuid)

    def _handle_list_files(self, project_uuid: str) -> str:
        files = self._get_project_files(project_uuid)
        file_list = [
            {
                "uuid": f.get("uuid"),
                "filename": f.get("filename"),
                "mime_type": f.get("mime_type"),
                "created_at": f.get("created_at"),
            }
            for f in files
        ]
        return json.dumps({"files": file_list, "total": len(file_list)})

    def _handle_file_read(
        self, project_uuid: str, file_uuid: str, force_reindex: bool = False
    ) -> str:
        files = self._get_project_files(project_uuid)
        target_file = next(
            (f for f in files if f.get("uuid") == file_uuid), None
        )

        if not target_file:
            raise ValueError(f"File {file_uuid} not found in project {project_uuid}")

        content = self._get_file_content(file_uuid)

        namespace = f"{NAMESPACE_PREFIX}-{project_uuid}"
        if force_reindex or not self._is_file_indexed(namespace, file_uuid):
            self._index_file(
                namespace, file_uuid, target_file.get("filename", ""), content
            )

        truncated = len(content) > MAX_CONTENT_LENGTH
        return json.dumps(
            {
                "file_uuid": file_uuid,
                "filename": target_file.get("filename"),
                "content": content[:MAX_CONTENT_LENGTH],
                "truncated": truncated,
                "total_length": len(content),
            }
        )

    def _handle_rag_query(
        self,
        project_uuid: str,
        query: str,
        top_k: int = 5,
        force_reindex: bool = False,
    ) -> str:
        namespace = f"{NAMESPACE_PREFIX}-{project_uuid}"

        if not self._pinecone_namespace_exists(namespace) or force_reindex:
            logger.info(f"Indexing all project files for namespace: {namespace}")
            self._index_all_project_files(project_uuid, namespace)

        embedding = self._get_embeddings([query])[0]

        matches = self._pinecone_query(
            vector=embedding, top_k=top_k, namespace=namespace
        )

        results = []
        for match in matches:
            metadata = match.get("metadata", {})
            results.append(
                {
                    "content": metadata.get("content", ""),
                    "filename": metadata.get("filename", ""),
                    "file_uuid": metadata.get("file_uuid", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": match.get("score", 0.0),
                }
            )

        return json.dumps({"query": query, "results": results, "total": len(results)})

    # ── API Helpers ──────────────────────────────────────────────────

    def _resolve_project_uuid(self) -> str:
        """Resolve project_uuid from the orchestration session."""
        session_uuid = self.orchestration_event.orchestration_session_uuid
        if not session_uuid:
            return None

        response = orchestrator_api_manager.call(
            "get_single_orchestration_session",
            orchestration_session_id=session_uuid,
            access_token=self.orchestration_event.access_token,
            organization_id=self.orchestration_event.organization.organization_id,
        )

        if response.get("status_code") not in (200, 201):
            logger.warning(f"Failed to get session details: {response.get('error')}")
            return None

        project_uuid = response.get("project_uuid")
        if project_uuid:
            logger.info(f"Resolved project_uuid {project_uuid} from session {session_uuid}")
        return project_uuid

    def _get_project_files(self, project_uuid: str) -> List[Dict[str, Any]]:
        response = files_api_manager.call(
            "get_files_for_project",
            project_uuid=project_uuid,
            access_token=self.orchestration_event.access_token,
            organization_id=self.orchestration_event.organization.organization_id,
        )

        if response.get("status_code") not in (200, 201):
            error_msg = response.get("error", "Unknown error")
            raise Exception(f"Failed to get project files: {error_msg}")

        return response.get("files", [])

    def _get_file_content(self, file_uuid: str) -> str:
        response = files_api_manager.call(
            "get_file_content",
            file_uuid=file_uuid,
            access_token=self.orchestration_event.access_token,
            organization_id=self.orchestration_event.organization.organization_id,
        )

        if response.get("status_code") not in (200, 201):
            error_msg = response.get("error", "Unknown error")
            raise Exception(f"Failed to get file content: {error_msg}")

        content = response.get("content")
        if content is not None:
            return content

        return json.dumps(response)

    # ── Indexing Helpers ─────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _index_file(
        self, namespace: str, file_uuid: str, filename: str, content: str
    ) -> None:
        chunks = self._chunk_text(content)
        if not chunks:
            return

        embeddings = self._get_embeddings(chunks)

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append(
                {
                    "id": f"{file_uuid}-chunk-{i}",
                    "values": embedding,
                    "metadata": {
                        "file_uuid": file_uuid,
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "content": chunk,
                    },
                }
            )

        self._pinecone_upsert(vectors=vectors, namespace=namespace)
        logger.info(f"Indexed {len(vectors)} chunks for file {filename} ({file_uuid})")

    def _index_all_project_files(self, project_uuid: str, namespace: str) -> None:
        files = self._get_project_files(project_uuid)
        for f in files:
            file_uuid = f.get("uuid", "")
            try:
                content = self._get_file_content(file_uuid)
                self._index_file(
                    namespace, file_uuid, f.get("filename", ""), content
                )
            except Exception as e:
                logger.error(f"Failed to index file {file_uuid}: {e}")

    def _is_file_indexed(self, namespace: str, file_uuid: str) -> bool:
        if not self._pinecone_namespace_exists(namespace):
            return False
        try:
            results = self._pinecone_query(
                vector=[0.0] * EMBEDDING_DIMS,
                top_k=1,
                namespace=namespace,
                filter={"file_uuid": {"$eq": file_uuid}},
            )
            return len(results) > 0
        except Exception as e:
            logger.warning(f"Error checking if file is indexed: {e}")
            return False

    @staticmethod
    def _validate_uuid(value: str, param_name: str) -> None:
        """Validate that a value is a proper UUID format."""
        try:
            uuid_module.UUID(value)
        except (ValueError, AttributeError):
            raise ValueError(
                f"{param_name} must be a valid UUID format, got: {value}"
            )

    def _extract_tool_args(self) -> Dict[str, Any]:
        extra_params = self.orchestration_event.extra_params or {}
        tool_calls = extra_params.get("tool_calls", [])
        if not tool_calls:
            logger.warning("No tool calls found in orchestration event")
            return {}
        tool_call = tool_calls[0]
        return tool_call.get("args", {})
