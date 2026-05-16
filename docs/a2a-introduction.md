# A2A Protocol Introduction

> **Background document** — Use this alongside the codebase to walk through the A2A protocol with live code examples.
>
> 📐 **Spec reference:** [A2A Protocol Specification (1.0)](https://a2a-protocol.org/latest/specification/) — released under the Linux Foundation, contributed by Google.

---

## Table of Contents

1. [What is A2A?](#1-what-is-a2a)
2. [Why A2A Exists](#2-why-a2a-exists)
3. [A2A vs MCP](#3-a2a-vs-mcp)
4. [Core Concepts](#4-core-concepts)
5. [Protocol Architecture (3 Layers)](#5-protocol-architecture-3-layers)
6. [The Agent Card — Discovery](#6-the-agent-card--discovery)
7. [Messages, Parts & Artifacts](#7-messages-parts--artifacts)
8. [Tasks & Lifecycle](#8-tasks--lifecycle)
9. [Operations (JSON-RPC Methods)](#9-operations-json-rpc-methods)
10. [Interaction Patterns](#10-interaction-patterns)
11. [Security & Authentication](#11-security--authentication)
12. [Our Implementation — Code Walkthrough](#12-our-implementation--code-walkthrough)
13. [Gap Analysis — What We Don't Implement](#13-gap-analysis--what-we-dont-implement)
14. [Resources](#14-resources)

---

## 1. What is A2A?

**Agent-to-Agent (A2A)** is an open protocol for agent-to-agent communication. It enables AI agents — built on different frameworks, by different teams, running on separate servers — to discover each other, negotiate interaction modalities, and collaborate on tasks.

Key properties:
- **Open standard** — Linux Foundation project, Apache 2.0 license
- **Framework-agnostic** — Works with any agent framework (LangChain, ADK, LlamaIndex, custom, etc.)
- **Opaque execution** — Agents collaborate without exposing internal state, memory, or tools
- **22k+ GitHub stars**, 141+ contributors, SDKs in Python, Go, JS, Java, .NET

```
┌──────────────────┐       A2A Protocol        ┌──────────────────┐
│  Agent A          │  ◄───────────────────►   │  Agent B          │
│  (any framework)  │   JSON-RPC / gRPC /      │  (any framework)  │
│                   │   HTTP+JSON over HTTPS   │                   │
└──────────────────┘                           └──────────────────┘
```

---

## 2. Why A2A Exists

| Problem | A2A Solution |
|---------|-------------|
| Agents are siloed in framework-specific ecosystems | Standardized wire protocol across all frameworks |
| No way to discover what an agent can do | **Agent Cards** — machine-readable capability manifests |
| Agents need to share internal state to collaborate | **Opaque execution** — only Messages and Artifacts are exchanged |
| Long-running tasks have no standard lifecycle | **Task model** with defined states, streaming, and push notifications |
| No standard for multi-turn conversations | **Context IDs** and **Task IDs** for conversation continuity |

---

## 3. A2A vs MCP

| | **MCP** (Model Context Protocol) | **A2A** (Agent-to-Agent) |
|---|---|---|
| **Purpose** | Agent ↔ Tool/Resource connectivity | Agent ↔ Agent collaboration |
| **Analogy** | "How an agent *uses* a tool" | "How agents *talk* to each other" |
| **Visibility** | Client sees tool internals (parameters, schema) | Agents are opaque black boxes |
| **State** | Stateless function calls | Stateful tasks with lifecycle |
| **Initiated by** | The agent (calling a tool) | Either agent (peer communication) |

**They're complementary:** An A2A client agent requests another A2A server agent to do something. That server agent internally uses MCP to call tools/APIs. The client never sees the tools.

```
┌─────────┐  A2A   ┌─────────┐  MCP   ┌───────────┐
│ Agent A  │ ────► │ Agent B  │ ────► │ Tool/API   │
│ (client) │       │ (server) │       │ (resource) │
└─────────┘       └─────────┘       └───────────┘
```

---

## 4. Core Concepts

### 4.1 Roles

| Role | Description |
|------|-------------|
| **A2A Client** | Initiates requests (sends messages) |
| **A2A Server** | Receives and processes requests (the remote agent) |

An agent can be both client AND server simultaneously.

### 4.2 The Six Key Objects

| Object | Purpose |
|--------|---------|
| **Agent Card** | Self-describing manifest (name, skills, supported interfaces, auth requirements) |
| **Message** | A communication turn — has a `role` (user/agent) and contains **Parts** |
| **Part** | Smallest content unit; in A2A 1.0 this is a protobuf oneof with `text`, `file`/`raw`/`url`, or `data` content |
| **Task** | Unit of work with lifecycle states and history |
| **Artifact** | Output produced by an agent (documents, results, structured data) |
| **Extension** | Optional add-on functionality beyond the core spec |

### 4.3 Identifiers

| ID | Scope | Purpose |
|----|-------|---------|
| `messageId` | Per message | Unique message identifier, enables idempotency |
| `taskId` | Per task | Tracks a unit of work through its lifecycle |
| `contextId` | Across tasks | Groups related tasks/messages in a conversation |

---

## 5. Protocol Architecture (3 Layers)

The spec separates **what** you say from **how** you say it. Think of it like human communication: the *content* of a conversation (data model) and the *actions* you can take (operations) are the same whether you communicate via phone, email, or face-to-face (bindings). This separation means you can switch transport mechanisms without changing your agent logic.

### Layer 1 — Canonical Data Model (the "nouns")

These are the **data structures** that every A2A implementation must understand. They're defined using Protocol Buffers (Google's language-neutral schema format) so they can be auto-generated for any programming language.

```
Task, Message, Part (text, file, or data content),
Artifact, AgentCard, TaskStatus, Extension, ...
```

> **Analogy:** These are like the "vocabulary" of A2A — the words everyone agrees on.

### Layer 2 — Abstract Operations (the "verbs")

These define **what you can do** — the actions/capabilities — independent of any specific transport:

```
SendMessage          — "Here's a message, please process it"
SendStreamingMessage — "Same, but stream me updates in real-time"
GetTask              — "What's the status of task X?"
ListTasks            — "Show me all tasks matching these filters"
CancelTask           — "Stop working on task X"
SubscribeToTask      — "Stream me updates for an existing task"
PushNotification CRUD — "Set up webhooks for async updates"
GetExtendedAgentCard — "Show me your full capabilities (authenticated)"
```

> **Analogy:** These are the "grammar rules" — the actions you can express using the vocabulary.

### Layer 3 — Protocol Bindings (the "delivery method")

These define **how** the data and operations travel over the wire. The spec provides three standard bindings:

| Binding | Transport | Format | Streaming | When to use |
|---------|-----------|--------|-----------|-------------|
| **JSON-RPC 2.0** | HTTP(S) | JSON | SSE (Server-Sent Events) | Most common, simple, our choice ← |
| **gRPC** | HTTP/2 | Protocol Buffers (binary) | Native gRPC streaming | High performance, internal services |
| **HTTP+JSON/REST** | HTTP(S) | JSON | SSE | REST-familiar teams, standard HTTP verbs |

**JSON-RPC** wraps every request in a standard envelope with `method`, `params`, `id`:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"text": "Hello"}],
      "messageId": "msg-uuid"
    }
  }
}
```

**gRPC** would send the exact same `SendMessage` operation with the same `Message` data, but encoded as binary Protocol Buffers over HTTP/2 — more efficient but less human-readable.

**HTTP+JSON/REST** would express the same as `POST /message:send` with a plain JSON body — familiar to anyone who's built REST APIs.

### How the layers stack

```
Layer 3:  Protocol Bindings — HOW it travels
          ├── JSON-RPC 2.0 over HTTP(S)     ← we use this one
          ├── gRPC over HTTP/2
          └── HTTP+JSON/REST
                    │
                    │  each binding implements ▼
                    │
Layer 2:  Abstract Operations — WHAT you can do
          SendMessage, GetTask, CancelTask, ListTasks, ...
                    │
                    │  operations act on ▼
                    │
Layer 1:  Canonical Data Model — THE DATA itself
          Task, Message, Part, Artifact, AgentCard, ...
```

### Why this matters

- **Interoperability:** Your agent uses JSON-RPC, another team's agent uses gRPC — they can still talk because the data model and operations are identical. Only a thin translation layer is needed at the binding level.
- **Future-proofing:** If a new transport emerges (e.g., WebSocket binding), it can be added as Layer 3 without touching Layers 1 or 2.
- **SDK simplicity:** The Python SDK handles the binding layer for you — you write `AgentExecutor.execute()` logic against the data model (Layer 1) and the SDK takes care of JSON-RPC serialization (Layer 3).

---

## 6. The Agent Card — Discovery

Every A2A server MUST publish an **Agent Card** — a JSON document describing the agent.

### Spec-defined fields (A2A 1.0):

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✅ | Human-readable agent name |
| `description` | ✅ | What the agent does |
| `supportedInterfaces` | ✅ | List of `{url, protocolBinding, protocolVersion}` |
| `version` | ✅ | Agent version |
| `capabilities` | ✅ | `{streaming, pushNotifications, extensions, extendedAgentCard}` |
| `defaultInputModes` | ✅ | Accepted media types (e.g. `["text/plain"]`) |
| `defaultOutputModes` | ✅ | Output media types |
| `skills` | ✅ | Array of `AgentSkill` (id, name, description, tags, examples) |
| `securitySchemes` | ❌ | Auth methods (OAuth2, API key, etc.) |
| `securityRequirements` | ❌ | Which security schemes are required |
| `provider` | ❌ | Organization info |
| `signatures` | ❌ | JWS signatures for integrity verification |
| `iconUrl` | ❌ | Agent icon URL |

### Discovery mechanisms:
1. **Well-Known URI:** `https://{domain}/.well-known/agent-card.json`
2. **Registry/Catalog:** Curated agent directories
3. **Direct configuration:** Pre-configured URLs

### 📂 Our code:
→ **`src/track_1_agent_under_test/server.py`** — `prepare_agent_card()` function creates our AgentCard

---

## 7. Messages, Parts & Artifacts

### Messages

A **Message** is a single communication turn. It has:
- `role`: `"user"` (from client) or `"agent"` (from server)
- `parts`: Array of content Parts
- `messageId`: Unique identifier
- `contextId`: Optional conversation context
- `taskId`: Optional task association
- `referenceTaskIds`: Optional references to related tasks
- `extensions`: Optional extension URIs
- `metadata`: Optional key-value pairs

### Parts — the content containers

| Part Type | Contains | Use Case |
|-----------|----------|----------|
| **Text part** | `text` (string) | Natural language, prompts, responses |
| **File part** | `url` or `raw` bytes + `mediaType` | Images, PDFs, audio |
| **Data part** | `data` (JSON object) + `mediaType` | Structured data, tool calls, forms |

A single message can contain **multiple Parts** of different types.

In this repository's A2A SDK 1.0 code, these are concrete protobuf `Part`
objects. Build them with `new_text_part(...)` and `new_data_part(...)`, and
parse them with `part.WhichOneof("content")`.

### Artifacts vs Messages

| | Messages | Artifacts |
|---|---|---|
| **Purpose** | Communication (back-and-forth) | Task outputs/results |
| **Direction** | Client ↔ Server | Server → Client (output only) |
| **Persistence** | May not persist in history | Attached to Task |
| **Usage** | Initiation, clarification, status | Documents, reports, generated data |

> **Spec rule:** Messages SHOULD NOT deliver task outputs. Results SHOULD use Artifacts.

### 📂 Our code:
→ **`src/agentbeats/client.py`** — `create_message()`, `create_message_with_parts()`, `merge_parts()`
→ **`src/track_1_agent_under_test/car_bench_agent.py`** — Building response Parts (text + data)
→ **`src/evaluator/car_bench_evaluator.py`** — Adds Artifacts with evaluation results

---

## 8. Tasks & Lifecycle

A **Task** is the fundamental unit of work. It progresses through defined states:

```
                          ┌──────────────┐
                    ┌────►│  completed   │ (terminal)
                    │     └──────────────┘
┌──────────┐    ┌───┴──────┐    ┌──────────────┐
│ submitted│───►│ working  │───►│   failed     │ (terminal)
└──────────┘    └───┬──────┘    └──────────────┘
                    │     ┌──────────────┐
                    ├────►│  canceled    │ (terminal)
                    │     └──────────────┘
                    │     ┌──────────────┐
                    ├────►│  rejected    │ (terminal)
                    │     └──────────────┘
                    │     ┌──────────────────┐
                    ├────►│ input-required   │ (interrupted → can resume)
                    │     └──────────────────┘
                    │     ┌──────────────────┐
                    └────►│ auth-required    │ (interrupted → can resume)
                          └──────────────────┘
```

### Intermediate Progress Updates (while `working`)

A task can stay in the `working` state and emit **multiple status updates** with progress information. Each update is delivered as a `TaskStatusUpdateEvent` via streaming/push notifications:

```
Client                                   Server
  │                                        │
  │◄── StatusUpdate{working,               │
  │     msg: "Starting evaluation..."}─────│
  │                                        │
  │◄── StatusUpdate{working,               │  ← intermediate update
  │     msg: "Turn 2/5 complete",          │
  │     metadata: {progress: 40}} ─────────│
  │                                        │
  │◄── StatusUpdate{working,               │  ← another intermediate update
  │     msg: "Processing tool results",    │
  │     metadata: {progress: 80}} ─────────│
  │                                        │
  │◄── ArtifactUpdate{result data} ────────│
  │◄── StatusUpdate{completed} ────────────│
```

The `metadata` field on each event is a `dict[str, Any]` — use it for custom fields like progress percentage, current step, tokens used, etc. Both `TaskStatusUpdateEvent` and `TaskArtifactUpdateEvent` support metadata.

> **Note:** Intermediate updates are only visible to clients using **streaming** or **push notifications**. Non-streaming `SendMessage` waits for the final result.

### The `TaskUpdater` — SDK helper

The Python SDK provides `TaskUpdater` to simplify emitting these events from your `AgentExecutor`:

| Method | Transitions to | Terminal? |
|--------|---------------|-----------|
| `submit(msg?)` | `submitted` | No |
| `start_work(msg?)` | `working` | No |
| `update_status(state, msg?, metadata?)` | any state | Depends |
| `requires_input(msg?)` | `input_required` | Interrupted |
| `requires_auth(msg?)` | `auth_required` | Interrupted |
| `add_artifact(parts, name?, metadata?)` | — | No (emits `TaskArtifactUpdateEvent`) |
| `complete(msg?)` | `completed` | ✅ Yes |
| `failed(msg?)` | `failed` | ✅ Yes |
| `cancel(msg?)` | `canceled` | ✅ Yes |
| `reject(msg?)` | `rejected` | ✅ Yes |

`update_status()` can be called **multiple times** with `TaskState.working` to send intermediate progress — each call emits a separate `TaskStatusUpdateEvent`.

### Task Update Delivery — 3 mechanisms:

| Mechanism | How it works | Best for |
|-----------|-------------|----------|
| **Polling** | Client calls `GetTask` periodically | Simple integrations |
| **Streaming** | SSE connection for real-time events | Interactive apps, dashboards |
| **Push Notifications** | Webhook POST to client endpoint | Long-running tasks, server-to-server |

### 📂 Our code:
→ **`src/agentbeats/evaluator_executor.py`** — Uses `TaskUpdater` to transition task states (`working` → `complete`/`failed`)
→ **`src/agentbeats/client_cli.py`** — Consumes streaming events (`TaskStatusUpdateEvent`, `TaskArtifactUpdateEvent`)

---

## 9. Operations (JSON-RPC Methods)

The spec defines **12 operations**. Here's the full list with what we implement:

| # | Operation | JSON-RPC Method | Description | We implement? |
|---|-----------|----------------|-------------|:---:|
| 1 | **Send Message** | `SendMessage` | Send a message, get Task or Message back | ✅ |
| 2 | **Stream Message** | `SendStreamingMessage` | Send with SSE real-time updates | ✅ |
| 3 | **Get Task** | `GetTask` | Retrieve current task state | via SDK |
| 4 | **List Tasks** | `ListTasks` | List/filter tasks with pagination | ❌ |
| 5 | **Cancel Task** | `CancelTask` | Cancel an in-progress task | ❌ (raises error) |
| 6 | **Subscribe to Task** | `SubscribeToTask` | SSE subscription for existing task | ❌ |
| 7 | **Create Push Notification Config** | `CreateTaskPushNotificationConfig` | Register webhook | ❌ |
| 8 | **Get Push Notification Config** | `GetTaskPushNotificationConfig` | Retrieve webhook config | ❌ |
| 9 | **List Push Notification Configs** | `ListTaskPushNotificationConfigs` | List webhooks | ❌ |
| 10 | **Delete Push Notification Config** | `DeleteTaskPushNotificationConfig` | Remove webhook | ❌ |
| 11 | **Get Extended Agent Card** | `GetExtendedAgentCard` | Auth'd card with more details | ❌ |
| 12 | **Get Agent Card** | (well-known URI) | Fetch the public agent card | ✅ (via SDK) |

### 📂 Our code:
→ **`src/agentbeats/client.py`** — `send_message()` uses the SDK's `client.send_message()` (operations 1+2)
→ **`src/agentbeats/sync_client.py`** — `send_message_with_parts_sync()` makes raw JSON-RPC calls (operation 1)
→ **`src/agentbeats/evaluator_executor.py`** — `cancel()` raises `UnsupportedOperationError`

---

## 10. Interaction Patterns

### 10.1 Single-Turn (Fire and Forget)

```
Client                              Server
  │                                    │
  │─── SendMessage("What's 2+2?") ───►│
  │                                    │
  │◄── Task{completed, artifacts} ─────│
  │                                    │
```

### 10.2 Multi-Turn (input-required)

```
Client                              Server
  │                                    │
  │─── SendMessage("Book flight") ────►│
  │                                    │
  │◄── Task{input-required:            │
  │     "Where from and to?"} ─────────│
  │                                    │
  │─── SendMessage(taskId=...,         │
  │     "SF to NYC") ────────────────►│
  │                                    │
  │◄── Task{completed} ───────────────│
```

### 10.3 Streaming

```
Client                              Server
  │                                    │
  │─── SendStreamingMessage ──────────►│
  │                                    │
  │◄── SSE: Task{working} ────────────│
  │◄── SSE: ArtifactUpdate(chunk1) ───│
  │◄── SSE: ArtifactUpdate(chunk2) ───│
  │◄── SSE: StatusUpdate{completed} ──│
  │                                    │
```

### 10.4 Context Continuity

- Same `contextId` across multiple tasks = shared conversation context
- Same `taskId` = continuing/refining the same task
- `referenceTaskIds` = explicitly linking related tasks

### 📂 Our code:
→ **`src/agentbeats/tool_provider.py`** — `ToolProvider` maintains `_context_ids` per agent URL for multi-turn
→ **`src/agentbeats/client_cli.py`** — Uses streaming with event consumer callback

---

## 11. Security & Authentication

The spec treats agents as **standard enterprise applications**, relying on established web security:

| Feature | Spec Requirement |
|---------|-----------------|
| **Transport** | MUST use HTTPS/TLS in production |
| **Auth schemes** | Declared via `securitySchemes` in AgentCard (OAuth2, API key, OpenID Connect, mTLS) |
| **Per-request auth** | Client includes credentials in every request |
| **In-task auth** | Agent transitions to `auth-required` state |
| **Data access** | Agent MUST scope data access per authenticated client |
| **Agent Card signing** | Optional JWS signatures for integrity verification |
| **Extended Agent Card** | Auth'd endpoint reveals additional capabilities |

### Versioning

- Clients MUST send `A2A-Version` header (e.g. `A2A-Version: 1.0`)
- Servers MUST process using the requested version's semantics
- Missing version header is interpreted as `0.3`

---

## 12. Our Implementation — Code Walkthrough

### Architecture

```
┌──────────────────────────────┐
│        A2A CLI        │  ← src/agentbeats/client_cli.py
│  (A2A Client + Orchestrator) │  ← src/agentbeats/run_scenario.py
└──────────┬───────────────────┘
           │ A2A (SendMessage + Streaming)
           ▼
┌──────────────────────────────┐
│      Evaluator (Server)    │  ← src/evaluator/
│   CAR-bench Evaluator        │
│   [EvaluatorExecutor extends     │
│    AgentExecutor]            │
└──────────┬───────────────────┘
           │ A2A (SendMessage, multi-turn via context_id)
           ▼
┌──────────────────────────────┐
│     Agent Under Test (Server)    │  ← src/track_1_agent_under_test/
│   Agent Under Test           │
│   [CARBenchAgentExecutor     │
│    extends AgentExecutor]    │
└──────────────────────────────┘
```

### Key files to show:

| File | What to show | A2A Concepts |
|------|-------------|--------------|
| `src/track_1_agent_under_test/server.py` | `prepare_agent_card()`, server setup | Agent Card, Starlette route setup |
| `src/track_1_agent_under_test/car_bench_agent.py` | `execute()` method — parsing Parts, building response | Messages, Parts (Text+Data), multi-turn |
| `src/agentbeats/client.py` | `send_message()`, `send_message_with_parts()` | Client → Server communication, SDK usage |
| `src/agentbeats/sync_client.py` | `send_message_with_parts_sync()` | Raw JSON-RPC request construction |
| `src/agentbeats/evaluator_executor.py` | `EvaluatorExecutor.execute()` | Task lifecycle, TaskUpdater |
| `src/agentbeats/client_cli.py` | `event_consumer()` callback | Streaming events (SSE) |
| `src/agentbeats/tool_provider.py` | `ToolProvider._context_ids` | Multi-turn context management |

### What the A2A Python SDK gives us:

| SDK Component | What it does | Where we use it |
|---------------|-------------|-----------------|
| `create_jsonrpc_routes` | Creates the A2A JSON-RPC POST route | `server.py` |
| `create_agent_card_routes` | Serves `/.well-known/agent-card.json` | `server.py` |
| `Starlette` | Combines A2A routes into the ASGI app | `server.py` |
| `DefaultRequestHandler` | Dispatches to `AgentExecutor` | `server.py` |
| `AgentExecutor` | Interface: `execute()` + `cancel()` | `car_bench_agent.py`, `evaluator_executor.py` |
| `InMemoryTaskStore` | Task persistence (in-memory) | `server.py` |
| `A2ACardResolver` | Fetches Agent Card from server | `client.py` |
| `ClientFactory` / `ClientConfig` | Creates A2A client with streaming support | `client.py` |
| `TaskUpdater` | Helper to emit task status updates | `evaluator_executor.py` |
| Types (`Message`, `Part`, `Task`, etc.) | Data model objects | Everywhere |

---

## 13. Gap Analysis — What We Don't Implement

### 13.1 Features NOT implemented

| A2A Feature | Spec Section | Status in Our Code | Impact |
|-------------|-------------|-------------------|--------|
| **Streaming responses (server-side)** | §3.1.2 | ❌ Not implemented — our agents return complete responses, no incremental streaming | Low — tasks are short enough |
| **`SubscribeToTask`** | §3.1.6 | ❌ Not implemented | Low — we use streaming on send |
| **`ListTasks`** | §3.1.4 | ❌ Not implemented | Low — single-task evaluation |
| **`CancelTask`** | §3.1.5 | ❌ Raises `UnsupportedOperationError` | Medium — can't cancel long evals |
| **Push Notifications** | §3.1.7–3.1.10 | ❌ Not implemented — `capabilities.pushNotifications` not set | Low — not needed for our use case |
| **Extended Agent Card** | §3.1.11 | ❌ Not implemented — `capabilities.extendedAgentCard` not set | Low |
| **`input-required` state** | §4.1.3 | ❌ Not used — we never ask client for mid-task input | Medium — could improve multi-turn |
| **`auth-required` state** | §4.1.3 | ❌ Not used | Low |
| **`rejected` state** | §4.1.3 | ❌ Not used | Low |
| **File part** | §4.1.6 | ❌ We only use text and data Parts, no file exchange | Low |
| **Extensions** | §4.6 | ❌ Not implemented | Low |
| **Agent Card Signing** | §8.4 | ❌ No JWS signatures on our Agent Card | Medium — no integrity verification |

### 13.2 Partially implemented

| A2A Feature | Spec Section | What We Do | What the Spec Says |
|-------------|-------------|-----------|-------------------|
| **`messageId`** | §4.1.4 | We generate UUIDs but don't use for idempotency | Servers MAY use `messageId` for dedup |
| **Error handling** | §3.3.2 | Basic errors via SDK | Spec defines 9 specific A2A error types with error codes |
| **Artifacts** | §3.7 | The evaluator emits benchmark result Artifacts ✅; ordinary agent turns are Messages | Task outputs should use Artifacts; conversational turns stay in Messages |
| **Security** | §7 | No auth — agents run locally or in Docker network | Production MUST use HTTPS + auth schemes |
| **Multi-turn** | §3.4 | Uses `contextId` for conversation continuity ✅ | Also supports `taskId` references and `referenceTaskIds` |
| **v0.3 compatibility** | Appendix A | JSON-RPC routes enable the SDK's v0.3 compatibility adapter | New integrations should use A2A 1.0 shapes first |

### 13.3 Summary

```
Implemented for CAR-bench: SendMessage, SendStreamingMessage, Agent Cards,
well-known card route, supportedInterfaces, explicit capabilities, A2A-Version
headers, protobuf Part oneofs, message IDs, context continuity, and result
Artifacts.

Out of scope for this benchmark harness: auth schemes, push notifications,
extended cards, file exchange, agent-side input-required flows, and general task
management APIs beyond the message path.
```

**What matters:** For our use case (benchmarking agents locally / in Docker), the current implementation covers the essential A2A features — SendMessage, multi-turn with context IDs, streaming events, Agent Cards, and the Task lifecycle. The missing features are primarily enterprise/production concerns such as security, push notifications, extended cards, and file exchange.

**If moving to production, prioritize:**
1. 🔒 Security — HTTPS + auth schemes in Agent Card
2. ❌ Cancel support — implement `CancelTask` for long evaluations
3. 📡 Push notifications — for async/disconnected evaluations
4. 📄 Extended Agent Card — expose authenticated details when needed
5. 📁 File Parts — add only if benchmark artifacts need binary/file exchange

---

## 14. Resources

| Resource | URL |
|----------|-----|
| **A2A Specification (1.0)** | https://a2a-protocol.org/latest/specification/ |
| **A2A GitHub Repository** | https://github.com/a2aproject/A2A |
| **Python SDK** | `pip install a2a-sdk` — [GitHub](https://github.com/a2aproject/a2a-python) |
| **Key Concepts Guide** | https://a2a-protocol.org/latest/topics/key-concepts/ |
| **Life of a Task** | https://a2a-protocol.org/latest/topics/life-of-a-task/ |
| **A2A and MCP** | https://a2a-protocol.org/latest/topics/a2a-and-mcp/ |
| **Streaming & Async** | https://a2a-protocol.org/latest/topics/streaming-and-async/ |
| **Enterprise Features** | https://a2a-protocol.org/latest/topics/enterprise-ready/ |
| **Python Tutorial** | https://a2a-protocol.org/latest/tutorials/python/1-introduction/ |
| **DeepLearning.AI Course** | https://goo.gle/dlai-a2a |
| **Samples Repository** | https://github.com/a2aproject/a2a-samples |
