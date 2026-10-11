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

## Offline verification

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Tests use real NeMo configuration and official middleware with fake provider responses. They verify policy placement in model parameters and the HTTP payload, input/output blocking, provider-error blocking, empty tool turns, Gemini text-block preservation, and a full Deep Agents tool round trip. Mocked classifications verify integration behavior; they do not establish a live endpoint's custom-policy enforcement.
