# Adapting the platform to real data and services

Every external dependency sits behind a small interface. This page says exactly what to change,
what is already done, and what you must re-measure afterwards. Sections are ordered from "config
only" to "write a class".

## 0. After ANY change below: re-measure

```powershell
python scripts/calibrate_gate.py            # relevance threshold depends on corpus + embedding model
python scripts/run_eval.py --no-llm --show-failures
python scripts/run_eval.py --gate           # full run, compared with eval/baseline.json
```

The numbers in this repository belong to the GitLab sample. They say nothing about your data until
you replace `eval/golden.yaml` with questions people actually ask (`scripts/build_golden.py`
drafts candidates; a person must review them).

## 1. Documents from an S3 / MinIO bucket (config only)

```yaml
# config/company.yaml
source:
  type: s3
  s3: {name: company-bucket, bucket: support-kb, prefix: incoming/, default_department: support}
```

Upload files under the prefix (md, html, docx, pdf, txt). For AWS S3 instead of MinIO, in `.env`:
`S3_ENDPOINT_URL=` (empty), real `S3_ACCESS_KEY` / `S3_SECRET_KEY`, `S3_REGION`, `S3_BUCKET`.
**Limitation:** the source connector and the raw landing store share one set of S3 credentials
(`core/storage.py`). Reading from a different account than you write to needs a second client.

## 2. The company's own files and permissions

Today `ingestion/metadata.py` invents `department` / `access_level` / `product` from the path.
For real data, replace `assign_metadata()` so the tags come from the source of truth (folder
rules, front matter, a mapping table, or the source system's labels), then adjust:

- `metadata:` taxonomy and `roles:` in `config/company.yaml` (who may see which tags);
- users: `scripts/seed_users.py` now, SSO later (see `docs/SECURITY.md`).

Nothing in retrieval, caching or evaluation changes: they only read the tags and the role config.

## 3. SharePoint (write one connector, plus one permission change)

Not implemented. The design, so it can be built and reviewed:

**Connector** `connectors/sharepoint.py`, a `SourceConnector` (see `connectors/base.py`):

| Method | Microsoft Graph |
|---|---|
| `list_documents()` | `GET /drives/{drive-id}/root/delta` (follow `@odata.nextLink`); keep files, skip folders; `path` = the item's path under the drive root; `etag` = the item's `eTag`/`cTag`; `last_modified` = `lastModifiedDateTime` |
| `fetch(path)` | `GET /drives/{drive-id}/root:/{path}:/content` |
| `list_changes(since)` | the same delta call with the saved `deltaLink`; items with a `deleted` facet are removals |

Authentication: an Entra ID app registration using client credentials with the narrowest scope
that works (`Sites.Selected` granted to one site, not `Sites.Read.All`). Token handling and 429
`Retry-After` back-off belong in the connector. Everything downstream (landing by content hash,
parsing, chunking, the deletion safety checks, Airflow) is unchanged: the sync already treats a
failed or partial listing as "delete nothing". Keep `doc_id` stable across renames by using the
drive item id as the path key if renames matter to you.

**Permissions** are the real work. SharePoint ACLs are per user/group, not three levels:

1. At ingestion, read each file's permissions (`GET .../items/{id}/permissions`) and store the
   allowed principal ids (Entra group ids) in an `allowed_groups` payload field on every chunk
   (and in the `documents` / `chunks` tables).
2. At query time, resolve the caller's groups (token claim or Graph) and filter the Qdrant search
   with `MatchAny(allowed_groups, caller_groups)` on **every prefetch**, as `core/vectorstore.py`
   already does for `access_level` / `department`.
3. Put the caller's group set into the answer-cache key (today: the role's permission fingerprint).
4. Re-run the leak checks in `eval/answer_eval.py` using the same ACL data as ground truth.
5. Plan for ACL changes without a content change: Graph delta reports permission changes, but you
   must re-tag the chunks (a payload update, no re-embedding).

Expect to verify this against a real tenant; mock tests alone prove the plumbing, not the behaviour
of your tenant's sharing links, inherited permissions or sensitivity labels.

## 4. A different language model

Implement the same two things as `OllamaClient` (`serving/llm.py`): a `stream(messages, max_tokens)`
generator that yields text pieces and then one `Usage`, and (optionally) `warm_up()`. Illustrative
sketch for an OpenAI-compatible server (not tested here):

```python
class OpenAICompatClient:
    def __init__(self, base_url, api_key, model):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model

    def stream(self, messages, max_tokens):
        import json, requests
        r = requests.post(f"{self.base_url}/v1/chat/completions", stream=True, timeout=(5, 120),
                          headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}})
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
        for line in r.iter_lines():
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                yield Usage(event["usage"]["prompt_tokens"], event["usage"]["completion_tokens"])
            for choice in event.get("choices", []):
                if choice["delta"].get("content"):
                    yield choice["delta"]["content"]
```

Register it in `build_llm()` under a new `LLM_PROVIDER` value. The answer-cache key already includes
the model name and prompt fingerprint, so a swap cannot serve old cached answers. A stronger
model also lets you raise `context_chunks` / `max_answer_tokens` and add an LLM-judged faithfulness
metric to `eval/` (the current checks are deliberately judge-free because a 3B judge is unreliable).
**Data leaves your network with a hosted model**: check that this is allowed for the documents.

## 5. A different embedding model or vector store

- Embedding model: change `EMBEDDING_MODEL`, run `python scripts/run_indexing.py --recreate`, then
  section 0. Vector size is checked on startup, so a mismatch fails loudly.
- Vector store: implement the methods `QdrantIndex` exposes (`ensure`, `upsert`, `delete_stale`,
  `delete_document`, `count`, `healthy`, `dense_search`, `sparse_search`, `hybrid_search`) with the
  permission filter applied inside each search. Retrieval, indexing and Airflow use only those.

## 6. Data licences

The sample corpus is the GitLab handbook (CC BY-SA 4.0) and GitLab docs. The repository does not
contain that content (it is cloned at run time into `~/.cache/rag-sources`), but answers shown to
users quote it: keep the attribution (the source links on every citation) and note ShareAlike if
you redistribute extracts. For company data, check the licence/confidentiality terms before
indexing, and especially before sending any text to a hosted model.
