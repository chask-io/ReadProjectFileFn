"""
ReadProjectFileFn - Business Logic

3 modes:
1. List files in a project
2. Read a specific file's content (max 8000 chars)
3. Semantic search via RAG/Pinecone
"""

import json
import logging
import ssl
import urllib.request
from typing import Dict, Any, List

from chask_foundation.backend.models import OrchestrationEvent
from chask_foundation.vector_store.factory import get_vector_store
from chask_foundation.file_processing.services.openai_service import OpenAIService
from api.files_requests import files_api_manager

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_CONTENT_LENGTH = 8000
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NAMESPACE_PREFIX = "project-files"


class FunctionBackend:
    """Backend for ReadProjectFileFn with 3 modes: list, read, RAG search."""

    def __init__(self, orchestration_event: OrchestrationEvent):
        self.orchestration_event = orchestration_event
        self._openai_service = None
        self._vector_store = None

    @property
    def openai_service(self) -> OpenAIService:
        if self._openai_service is None:
            self._openai_service = OpenAIService()
        return self._openai_service

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = get_vector_store()
        return self._vector_store

    def process_request(self) -> str:
        tool_args = self._extract_tool_args()

        project_uuid = tool_args.get("project_uuid")
        if not project_uuid:
            raise ValueError("Missing required parameter: project_uuid")

        file_uuid = tool_args.get("file_uuid")
        query = tool_args.get("query")
        force_reindex = tool_args.get("force_reindex", False)
        top_k = tool_args.get("top_k", 5)

        if query:
            return self._handle_rag_query(project_uuid, query, top_k, force_reindex)
        elif file_uuid:
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
        target_file = None
        for f in files:
            if f.get("uuid") == file_uuid:
                target_file = f
                break

        if not target_file:
            raise ValueError(f"File {file_uuid} not found in project {project_uuid}")

        presigned_url = target_file.get("presigned_url")
        if not presigned_url:
            raise ValueError(f"No presigned URL available for file {file_uuid}")

        content = self._download_file_content(presigned_url)

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

        if not self.vector_store.namespace_exists(namespace) or force_reindex:
            logger.info(f"Indexing all project files for namespace: {namespace}")
            self._index_all_project_files(project_uuid, namespace)

        embedding = self.openai_service.get_embeddings([query])[0]

        matches = self.vector_store.query_vectors(
            vector=embedding, top_k=top_k, namespace=namespace
        )

        results = []
        for match in matches:
            if hasattr(match, "metadata"):
                metadata = match.metadata
                score = match.score
            else:
                metadata = match.get("metadata", {})
                score = match.get("score", 0.0)

            results.append(
                {
                    "content": metadata.get("content", ""),
                    "filename": metadata.get("filename", ""),
                    "file_uuid": metadata.get("file_uuid", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": score,
                }
            )

        return json.dumps({"query": query, "results": results, "total": len(results)})

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

    def _download_file_content(self, presigned_url: str) -> str:
        ssl_context = ssl.create_default_context()
        req = urllib.request.Request(presigned_url)
        with urllib.request.urlopen(req, context=ssl_context, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")

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

        embeddings = self.openai_service.get_embeddings(chunks)

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

        self.vector_store.upsert_vectors(vectors=vectors, namespace=namespace)
        logger.info(f"Indexed {len(vectors)} chunks for file {filename} ({file_uuid})")

    def _index_all_project_files(self, project_uuid: str, namespace: str) -> None:
        files = self._get_project_files(project_uuid)
        for f in files:
            presigned_url = f.get("presigned_url")
            if not presigned_url:
                logger.warning(f"Skipping file {f.get('uuid')} - no presigned URL")
                continue
            try:
                content = self._download_file_content(presigned_url)
                self._index_file(
                    namespace, f.get("uuid", ""), f.get("filename", ""), content
                )
            except Exception as e:
                logger.error(f"Failed to index file {f.get('uuid')}: {e}")

    def _is_file_indexed(self, namespace: str, file_uuid: str) -> bool:
        if not self.vector_store.namespace_exists(namespace):
            return False
        try:
            dummy_vector = [0.0] * 1536
            results = self.vector_store.query_vectors(
                vector=dummy_vector,
                top_k=1,
                namespace=namespace,
                filter={"file_uuid": {"$eq": file_uuid}},
            )
            return len(results) > 0
        except Exception as e:
            logger.warning(f"Error checking if file is indexed: {e}")
            return False

    def _extract_tool_args(self) -> Dict[str, Any]:
        extra_params = self.orchestration_event.extra_params or {}
        tool_calls = extra_params.get("tool_calls", [])
        if not tool_calls:
            logger.warning("No tool calls found in orchestration event")
            return {}
        tool_call = tool_calls[0]
        return tool_call.get("args", {})
