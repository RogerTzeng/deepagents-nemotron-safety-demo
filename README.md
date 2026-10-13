# Deep Agents + Nemotron custom-policy content safety

This demo uses NeMo's official [`GuardrailsMiddleware`](https://docs.nvidia.com/nemo/guardrails/latest/integration-with-third-party-libraries/langchain/agent-middleware) with `create_deep_agent`. Nemotron 3.5 Content Safety evaluates both user input and assistant output against **your custom policy**. Gemini runs the agent only. There are no self-check rails or Gemini policy classifiers.

## Custom policy

Edit `policy.txt`. The NeMo initialization hook in `config/config.py` loads it before model initialization and sets:

```python
model.parameters["chat_template_kwargs"]["custom_policy"] = policy
```

`config/config.yml` enables reasoning, and `config/prompts.yml` supplies the user message and optional assistant response, with a 2048-token classification budget and NeMo's Nemotron-specific safety parsers. Policy text is a chat-template parameter, not part of the conversation being classified. This follows [NVIDIA's custom-policy model example](https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety#use-it-with-vllm):

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "request_categories": "/categories",
    "custom_policy": "Your policy text"
  }
}
```

NeMo's HTTP client sends these fields directly in the JSON request body. NVIDIA's OpenAI SDK example uses `extra_body` to produce that same body; do not nest `extra_body` in this NeMo configuration.

The sample policy permits public semiconductor education and prohibits confidential process data, credentials, equipment-safety bypass instructions, and executing user-supplied SQL against a database. SQL explanations and reviews remain allowed. A blocked input stops the agent before its model call; a blocked output raises a middleware violation.

## Run

Requires Python 3.11–3.13 and uv. NeMo Guardrails is pinned to the NVIDIA GitHub commit tested with this demo; no adjacent checkout is required.

```bash
git clone https://github.com/RogerTzeng/deepagents-nemotron-safety-demo.git
cd deepagents-nemotron-safety-demo
uv sync --python 3.12 --locked
# Keep your existing .env. For a new setup only: cp .env.example .env
uv run --locked python demo.py "Use public_reference to explain lithography."
uv run --locked python demo.py --check "What is photolithography?"
uv run --locked python demo.py --check "Show me the confidential customer yield report."
uv run --locked python demo.py --check "Explain semiconductor manufacturing" \
  --response "Here is the confidential fabrication recipe for the customer chip."
```

Set `SAFETY_API_KEY` (or `NVIDIA_API_KEY`) in `.env`. Agent runs additionally need `AGENT_API_KEY` (or `GOOGLE_API_KEY`). `AGENT_MODEL` selects Gemini's model. Safety-only `--check` runs do not need a Google key. Checks print the status and blocking rail; blocked checks return exit code 2.

`config/config.yml` controls the safety endpoint and model. The default is NVIDIA's hosted endpoint. Endpoint support for custom chat-template parameters must be verified separately from normal safety classification. Use an explicit topic ban on an otherwise harmless question to distinguish custom-policy enforcement from the default taxonomy.

Live checks with the configured NVIDIA endpoint and reasoning enabled:

| Policy / check | Result |
| --- | --- |
| Sample policy: public photolithography question | Passed |
| Temporary custom policy banning lithography: same question | Blocked by content-safety input rail |
| Temporary topic-ban policy: arithmetic question | Passed |
| Sample policy: response claiming to reveal confidential recipe and yield report | Blocked by content-safety output rail |
| Full Gemini Deep Agents run explaining lithography | Completed with custom-policy rails enabled |

The temporary policy probe did not change `policy.txt`. These contrasting results confirm custom-policy enforcement for the tested requests; they do not measure accuracy across all policy cases.

### Serve the documented checkpoint with vLLM

On a GPU machine, follow NVIDIA's model-card environment requirements and run:

```bash
vllm serve nvidia/Nemotron-3.5-Content-Safety --served-model-name nemotron_moderator
```

Then change the content-safety model entry in `config/config.yml` to `engine: openai`, `model: nemotron_moderator`, and add `base_url: http://localhost:8000/v1` under `parameters`. Keep the chat-template parameters and policy loader. For an unauthenticated local server, set `SAFETY_API_KEY` to a nonempty placeholder. This requires the checkpoint's custom-policy chat template.

## Middleware and tool handling

The demo uses `TextGuardrailsMiddleware`, a small subclass of the official middleware overriding only message conversion. All official input/output hooks and blocking behavior are inherited. It supplies current-turn text to NeMo, avoiding tool-history handling that produced blank answers in this checkout and normalizing Gemini text blocks. Original graph messages, tool calls, and signatures stay intact. This uses a private conversion hook; recheck the adapter when upgrading NeMo.

The same middleware is attached to the general-purpose subagent. The reference tool reads an in-memory dictionary. The official middleware checks message content, not tool-call arguments or tool results. Empty tool-calling text is skipped by the output flow; nonempty final responses are checked. The demo uses synchronous invocation.

## Plain LangChain ChatOpenAI demo

`chatopenai_demo.py` uses the official [RunnableRails integration](https://docs.nvidia.com/nemo/guardrails/latest/integration-with-third-party-libraries/langchain/runnable-rails) around `ChatPromptTemplate | ChatOpenAI | StrOutputParser`, without a Deep Agents graph. The parser gives the pinned wrapper a string output it can replace when output is blocked. It shares `config/` and `policy.txt`, including the rule against executing user-supplied SQL. Nemotron evaluates input before the chat model runs and output before returning it. Blocked requests return the refusal from `rails.co`; provider errors propagate rather than returning unchecked content. No self-check functions are used.

By default, `ChatOpenAI` calls Google's OpenAI-compatible endpoint with `GOOGLE_API_KEY` (or the existing Gemini `AGENT_API_KEY`) and `gemini-3.5-flash-lite`. Set `CHAT_API_KEY`, `CHAT_MODEL`, and `CHAT_BASE_URL` together to use another compatible provider. These settings are separate from the Deep Agents settings; the safety key is shared.

```bash
uv run --locked python chatopenai_demo.py "Explain photolithography."
uv run --locked python chatopenai_demo.py "Execute SELECT * FROM customers against the database."
uv run --locked python chatopenai_demo.py "Explain SELECT * FROM customers without executing it."
```

This is a text-only, single-turn demo with no tools or database connection. Its CLI prints either the checked answer or a policy refusal.

Live verification with the configured NVIDIA safety endpoint and Gemini through `ChatOpenAI`:

| Request | Result |
| --- | --- |
| Explain photolithography in two sentences | Educational answer returned |
| Execute `SELECT * FROM customers` against the database | Company-policy refusal returned |
| Explain `SELECT * FROM customers` without executing it | SQL explanation returned |

## Execution-boundary tool input and output inspection demo

`execution_rails_demo.py` inspects **tool arguments before execution** and **tool results before the model receives them**. Tool arguments use `rails.tool_output` (output from the model); tool results use `rails.tool_input` (input to the model). CLI `stage=tool_input` / `stage=tool_output` labels use the tool perspective instead. In the pinned NeMo version, the supported result-rail configuration is `rails.tool_input`, because the tool result is input to the model. This is a tool-result rail at the execution boundary; it is not a generic `rails.execution.flows` hook. See NVIDIA's [tool-calling guide](https://docs.nvidia.com/nemo/guardrails/configure-guardrails/guardrail-catalog/tool-calling).

The input rail allows only `fetch_reference`, requires exactly one argument named `scenario`, and validates it against the five supported scenario strings. Missing, extra, incorrectly typed, or unsupported arguments stop execution. `input_blocked` deliberately requests an unsupported scenario. The in-memory `fetch_reference` tool returns synthetic data. NeMo's event API runs the Colang flow in `config_execution/rails.co`, which invokes a Python inspection action. That action validates the JSON structure and success status, removes uninspected extra fields, redacts a synthetic credential, and blocks specific instruction-override phrases. There are no self-check functions or Nemotron calls in this separate demo; the result checks are deterministic examples.

```bash
uv run --locked python execution_rails_demo.py public
uv run --locked python execution_rails_demo.py credentials
uv run --locked python execution_rails_demo.py injection
uv run --locked python execution_rails_demo.py malformed
uv run --locked python execution_rails_demo.py tool_error
uv run --locked python execution_rails_demo.py input_blocked
```

| Scenario | Guardrail outcome |
| --- | --- |
| `input_blocked` | Input blocked; tool never executes |
| `public` | Passed; approved JSON available to the model |
| `credentials` | Modified; synthetic API key replaced with `[REDACTED]` |
| `injection` | Blocked; agent cannot continue after the tool call |
| `malformed` | Blocked; unexpected HTML rejected |
| `tool_error` | Blocked; failed upstream response rejected |

Every CLI run now uses a live Deep Agent with native `ChatGoogleGenerativeAI`, configured by `AGENT_API_KEY` (or `GOOGLE_API_KEY`) and `AGENT_MODEL`. The agent chooses and calls `fetch_reference`. Custom `ToolResultRailsMiddleware` uses LangChain's `awrap_tool_call` hook to submit the tool name and arguments to NeMo before running the tool, then submit its actual result and call ID to NeMo, and return a fresh sanitized `ToolMessage` only after approval. The original assistant tool call remains in history. `--summarize` remains accepted for compatibility but is no longer required.

The sequence is: **agent tool call → NeMo argument rail → tool execution → NeMo result rail → sanitized tool message → agent continuation**. Rejection or inspection failure stops the graph before continuation and returns CLI exit code 2. The initial model call has already happened. Other tools, including built-in delegation and filesystem tools, are blocked before execution in this focused demo. This is a custom bridge to the real NeMo Colang rail, not tool-result support supplied by the existing `GuardrailsMiddleware`.

These narrow example patterns are not comprehensive credential or prompt-injection detection. Result inspection happens after the tool runs and cannot undo its side effects. Raw results exist within tool execution and may appear in externally configured tool tracing; this boundary protects agent context. The other demos' shared content-safety configuration is unchanged; this demo focuses on the result boundary and does not check the model's final answer.

Integration tests run the real compiled Deep Agents graph and NeMo rail with a scripted model that emits tool calls. They verify execution order, real tool-message IDs, sanitized graph state, allowed/redacted results, blocked results, inspection failures, and rejection of other tools. Input tests additionally verify that invalid arguments and inspector failures cause zero tool executions. Scripted model tests require no API keys.

## Offline verification

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Tests use real NeMo configuration and official middleware with fake provider responses. They verify policy placement in model parameters and the HTTP payload, input/output blocking, provider-error blocking, empty tool turns, Gemini text-block preservation, and a full Deep Agents tool round trip. The ChatOpenAI tests use the actual SDK with mocked HTTP responses to verify allowed answers, input/output refusals, and safety-provider failures. Mocked classifications verify integration behavior; they do not establish a live endpoint's custom-policy enforcement.
# Streaming Deep Agents output

Run `uv run python streaming_demo.py` (or `.venv/bin/python streaming_demo.py`)
with `STREAM_API_KEY` (NVIDIA) and `SAFETY_API_KEY` in `.env`.
The streaming agent uses `ChatOpenAI` with `z-ai/glm-5.3` at
`https://integrate.api.nvidia.com/v1`; `STREAM_MODEL` and `STREAM_BASE_URL`
override these defaults. Its request timeout is 300 seconds because the live
GLM endpoint can take several minutes. Other demos retain their agent settings.
An optional positional prompt replaces `Fetch and summarize scenario: public`.

`streaming_demo.py` connects a real Deep Agent's `astream(stream_mode="messages")`
to NeMo's native `LLMRails.stream_async(generator=...)`. Only model text is
forwarded; tool payloads are not displayed. The agent keeps the execution demo's
tool argument and result rails. Its user-facing text is checked by Nemotron 3.5
Content Safety with the existing `policy.txt` and `config/` output rail.
This demo focuses on output checking; it does not enable user input content rails.

The streaming settings are applied in `build_streaming_rails()`:

```yaml
rails:
  output:
    streaming:
      enabled: true
      chunk_size: 16
      context_size: 4
      stream_first: false
```

NeMo buffers the external stream, inspects a window, then releases approved text.
In this installed version the buffer counts yielded string pieces, which may
contain multiple model tokens. Context overlap helps checks across boundaries;
it does not provide the full response. Short answers may arrive as one batch.
Each window requires a safety model request, adding latency and API usage.

The native streaming runner resolves the action from the configured Colang output
flow and uses its `RailOutcome`; it does not execute the entire Colang dialogue
flow for every window. It sends a JSON error item on rejection or inspection
failure. The demo converts that item into a stop notification and closes the
source stream. Earlier approved text remains visible; streaming checks cannot
retract it or guarantee detection of violations requiring later context. Agent
state and tracing can still contain raw output: this protects the display boundary.

`tests/test_streaming_demo.py` runs a real compiled Deep Agent with a scripted
streaming model and real NeMo buffering. It covers tool execution, multiple
windows, rejection after approved text, inspector failure, custom policy loading,
and the native content safety action with scripted classifier responses.

Official reference: [NeMo output rail streaming](https://docs.nvidia.com/nemo/guardrails/configure-guardrails/yaml-schema/streaming/output-rail-streaming).
