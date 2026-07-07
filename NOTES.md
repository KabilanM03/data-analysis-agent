# Build notes

Personal notes from building this. Not polished documentation, just the
things that surprised me, the dead ends, and what I'd do next.

## Why smolagents and not LangChain

I started with LangChain because that was the framework I'd seen most in
job postings, but the agent layer felt heavy for what I actually needed:
a handful of pandas functions and a chat loop. The HF Agents Course was
recommending smolagents at the time so I switched, and the `CodeAgent`
ergonomics turned out to fit better. The model writes Python that calls
my `@tool` functions directly. No JSON schema dance, no agent executor
boilerplate. Easier to debug too because the tool call is just a
function invocation.

## The Kaggle SDK is a moving target

The first version of `fetch_tools.py` imported a class called
`KaggleApiClient` which doesn't exist. I had cribbed from a snippet I
hadn't actually run. The real entry point is `KaggleApi` from
`kaggle.api.kaggle_api_extended`. Lesson: run every import once
in a Python REPL before trusting it, especially when the suggestion came
from an autocomplete.

## Dataframe state caused me grief

Original design kept the active dataset in a module-level dict keyed by
name. Worked for one user, broke as soon as I imagined two Gradio
sessions in the same process. I rebuilt it around a `DataframeStore`
class held inside a `Session` dataclass, bound to a `ContextVar` for the
duration of `agent.run()`. The tools still call `get_active_df()` /
`set_active_df()` like before, they just resolve through the context
var. Tests get isolation via an autouse fixture that swaps in a fresh
store.

The thing I didn't appreciate until I tried it: `gr.State(SomeObject())`
passes the same instance to every browser tab. The fix is
`gr.State(value=None)` plus a `demo.load(factory_fn, outputs=session)`
call so each visitor gets a fresh `Session`.

## Charts in a chat box

Returning a markdown image link from a tool looked clean but Gradio
doesn't serve arbitrary file paths to the Chatbot component by default.
You either configure `allowed_paths` on `demo.launch()` or surface the
image as a separate message with `{"path": "..."}` content. I went with
both: the viz tool returns a `[CHART:<path>]` sentinel, the chat handler
strips it from the text response and appends the image as its own
assistant message.

## Pinning a dependency is not the same as testing against it

While preparing the 1.0 release I bumped the pin to Gradio 6 and assumed
the app still worked because the test suite was green. It wasn't: Gradio 6
removed the `type="messages"` argument on `Chatbot` (messages became the
default), so `app.py` crashed on the very first line of UI construction.
None of the 29 tests caught it because none of them ever built the UI;
they all test the tools layer. Two fixes: dropped the dead argument, and
added a test that actually calls `build_ui()` so the suite fails the next
time a Gradio upgrade breaks a constructor.

## What I haven't done

- HF Spaces deployment. The repo is wired for it (no hardcoded host,
  PLOTS_DIR is env-configurable) but I haven't pushed a Space yet.
- A Dockerfile. Would let me back up the "deployable" claim more
  cleanly.
- LangSmith / W&B tracing. Free tier would be enough; useful for
  showing observability awareness.
- Streaming. The chat blocks for 10-30 seconds with no feedback while
  the model thinks. smolagents supports step streaming, I just haven't
  wired it through Gradio.
- An eval harness beyond the small YAML fixture in `evals/`. Five
  golden questions is a starting point, not a benchmark.

## Things I'd do differently if starting again

1. Set up the eval harness first, so every change has a measurable
   answer to "did this regress anything".
2. Mock the network tools from day one. The mocked tests caught two
   regressions while I was refactoring state handling.
3. Pick a single LLM for the dev loop and only add fallbacks once the
   tool surface is stable. Routing across HF / Anthropic / Ollama too
   early hid model-specific weirdness.
