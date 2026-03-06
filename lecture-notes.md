# Prompt Engineering to Agentic AI
## A Practitioner's Guide to Building with LLMs

---

## 1. The OpenAI API (~5 min)

Before we get into prompting techniques, let's understand what we're actually talking to.

### What is an LLM API?

An LLM API is a stateless HTTP endpoint. You send a message, you get a response. There's no memory, no session - every request is independent. This is a crucial mental model for everything that follows.

### The Chat Completions API

The OpenAI API uses a **message-based** format. Every request is a list of messages with roles:

- **system** - Sets the behavior and personality of the model
- **user** - The human's input
- **assistant** - The model's response (used when providing conversation history)

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from environment

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is prompt engineering?"}
    ]
)

print(response.choices[0].message.content)
```

### Model Parameters

These parameters let you control *how* the model generates its response. Understanding them is essential - the same prompt can produce wildly different results depending on these settings.

#### `model` - Which model to use

OpenAI offers models at different capability/cost trade-offs:

| Model | Best for | Relative cost |
|-------|----------|---------------|
| `gpt-4.1` | Complex reasoning, coding, nuanced tasks | $$$ |
| `gpt-4.1-mini` | Good balance of quality and speed | $$ |
| `gpt-4.1-nano` | Simple tasks, high volume, lowest latency | $ |

**Rule of thumb:** Start with the cheapest model that works. You can always move up.

#### `temperature` - Controlling randomness

Temperature controls how "creative" vs "deterministic" the output is. Technically, it scales the probability distribution over tokens before sampling.

| Value | Behavior | Use case |
|-------|----------|----------|
| `0.0` | Always picks the most likely token. Nearly deterministic. | Classification, data extraction, factual Q&A |
| `0.3-0.7` | Some variation, mostly coherent | General conversation, summarization |
| `0.8-1.2` | More creative, less predictable | Creative writing, brainstorming |
| `1.5-2.0` | Highly random, can be incoherent | Experimental, rarely useful in production |

```python
# Deterministic - always gives the same output for the same input
response = client.chat.completions.create(
    model="gpt-4.1",
    temperature=0.0,
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

# Creative - different output each time
response = client.chat.completions.create(
    model="gpt-4.1",
    temperature=1.0,
    messages=[{"role": "user", "content": "Write a haiku about Python programming."}]
)
```

#### `max_tokens` - Response length limit

Sets the maximum number of tokens the model can generate. The model will stop generating once it hits this limit, even mid-sentence.

- **1 token** ≈ 4 characters in English (roughly 3/4 of a word)
- "Hello, how are you today?" = 7 tokens
- If you set `max_tokens=10` and the answer needs 50, you'll get a truncated response

```python
# Short response - good for classification
response = client.chat.completions.create(
    model="gpt-4.1",
    max_tokens=5,
    messages=[{"role": "user", "content": "Is this sentence positive or negative: 'I love this product'"}]
)

# Long response - needed for detailed explanations
response = client.chat.completions.create(
    model="gpt-4.1",
    max_tokens=2048,
    messages=[{"role": "user", "content": "Explain how neural networks work."}]
)
```

**Cost note:** You pay for both input tokens (your prompt) and output tokens (the response). Setting `max_tokens` caps your output cost per request.

#### `top_p` - Nucleus sampling

An alternative to temperature for controlling randomness. Instead of scaling probabilities, `top_p` only considers the smallest set of tokens whose cumulative probability exceeds `p`.

| Value | Behavior |
|-------|----------|
| `0.1` | Only considers the top 10% most likely tokens - very focused |
| `0.9` | Considers the top 90% - more diverse |
| `1.0` | Considers all tokens (default) |

**In practice:** Use `temperature` *or* `top_p`, not both. OpenAI recommends adjusting one and leaving the other at its default.

#### `frequency_penalty` and `presence_penalty` - Reducing repetition

These two parameters discourage the model from repeating itself:

| Parameter | What it does | Range |
|-----------|-------------|-------|
| `frequency_penalty` | Penalizes tokens proportional to how often they've appeared | -2.0 to 2.0 (default: 0) |
| `presence_penalty` | Penalizes any token that has appeared at all, regardless of frequency | -2.0 to 2.0 (default: 0) |

```python
# Reduce repetitive text in longer outputs
response = client.chat.completions.create(
    model="gpt-4.1",
    frequency_penalty=0.5,
    presence_penalty=0.3,
    messages=[{"role": "user", "content": "Write a 500-word essay on climate change."}]
)
```

**When to use these:** Most useful for open-ended generation (essays, stories). For structured tasks like classification or extraction, the defaults are usually fine.

#### Putting it all together

```python
response = client.chat.completions.create(
    model="gpt-4.1",
    temperature=0.0,          # deterministic for consistent results
    max_tokens=500,            # cap the response length
    top_p=1.0,                 # default, since we're using temperature
    frequency_penalty=0.0,     # default, no repetition penalty
    presence_penalty=0.0,      # default
    messages=[
        {"role": "system", "content": "You are a concise technical assistant."},
        {"role": "user", "content": "Explain what an API is in 2 sentences."}
    ]
)
```

**Key insight:** The model doesn't "know" anything about previous requests. If you want context, *you* have to provide it in the messages array. This is why prompting technique matters - you're engineering the input to get better output.

---

## 2. Zero-Shot Prompting (~5 min)

### What is it?

Zero-shot prompting is the simplest approach: you ask the model to do something with **no examples**. You rely entirely on the model's pre-trained knowledge.

### When to use it

- Simple, well-defined tasks
- The task is something the model likely saw during training
- You want a quick prototype

### Example

```python
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a sentiment analysis classifier."},
        {"role": "user", "content": "Classify the sentiment of this review as positive, negative, or neutral: 'The battery life is incredible but the screen is too dim.'"}
    ]
)
```

### The System Prompt Matters

The system message is your most powerful lever in zero-shot prompting. Compare:

```python
# Vague system prompt
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Analyze this text: 'The battery life is incredible but the screen is too dim.'"}
]

# Precise system prompt
messages = [
    {"role": "system", "content": "You are a sentiment analysis engine. Respond with exactly one word: positive, negative, or neutral."},
    {"role": "user", "content": "Classify: 'The battery life is incredible but the screen is too dim.'"}
]
```

The second version gives you structured, predictable output. **Specificity in your prompt directly correlates with reliability of the output.**

### Limitations

- Performance degrades on complex or ambiguous tasks
- No way to steer the output format by example
- Model may interpret the task differently than you intended

---

## 3. Few-Shot Prompting (~7 min)

### What is it?

Few-shot prompting provides **examples of the desired input-output behavior** directly in the prompt. You're teaching the model a pattern by demonstration.

### The jump from zero-shot

Instead of hoping the model interprets your task correctly, you *show* it what correct looks like.

```python
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You extract structured data from product reviews."},

        # Example 1
        {"role": "user", "content": "Review: 'Love the camera but the phone overheats during gaming.'"},
        {"role": "assistant", "content": '{"positive": ["camera"], "negative": ["overheating during gaming"]}'},

        # Example 2
        {"role": "user", "content": "Review: 'Fast shipping, great price, but the color was different than pictured.'"},
        {"role": "assistant", "content": '{"positive": ["shipping speed", "price"], "negative": ["color accuracy"]}'},

        # Actual input
        {"role": "user", "content": "Review: 'The keyboard feels premium and typing is quiet, but it disconnects from Bluetooth randomly.'"}
    ]
)
```

### Why this works

The model sees the pattern in the assistant messages and mimics it. You're defining:
- **Input format** - what the user message looks like
- **Output format** - exactly how the response should be structured
- **Edge cases** - how to handle mixed sentiment, multiple features, etc.

### How many examples?

| Examples | Trade-off |
|----------|-----------|
| 1-2 | Minimal pattern, but saves tokens |
| 3-5 | Sweet spot for most tasks |
| 5-10 | Diminishing returns, higher cost |

### Limitations

- Uses up your context window (you're paying for examples on every request)
- Examples can accidentally introduce bias
- Still limited by what the model can infer from patterns alone

---

## 4. Chain of Thought (~7 min)

### What is it?

Chain of Thought (CoT) prompting asks the model to **show its reasoning step by step** before giving a final answer. This dramatically improves performance on tasks that require logic, math, or multi-step reasoning.

### Why it matters

LLMs generate text left-to-right, one token at a time. When you ask for just an answer, the model has to "think" in one shot. When you ask it to reason first, each reasoning step becomes context for the next - the model can build up to the answer.

### Zero-shot CoT

The simplest version - just add "think step by step":

```python
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a math tutor. Think through each problem step by step before giving your final answer."},
        {"role": "user", "content": "A store has a 20% off sale. You also have a $5 coupon applied after the discount. If the original price is $80, what do you pay?"}
    ]
)
```

### Few-shot CoT

Combine examples with explicit reasoning chains:

```python
messages = [
    {"role": "system", "content": "Solve word problems step by step."},

    # Example with reasoning
    {"role": "user", "content": "If a train travels 60 mph for 2.5 hours, how far does it go?"},
    {"role": "assistant", "content": """Step 1: Identify what we know.
- Speed = 60 mph
- Time = 2.5 hours

Step 2: Apply the formula distance = speed x time.
- Distance = 60 x 2.5

Step 3: Calculate.
- Distance = 150 miles

**Answer: 150 miles**"""},

    # Actual question
    {"role": "user", "content": "You buy 3 items at $12.50 each with 8% sales tax. What's the total?"}
]
```

### The key insight

CoT works because **reasoning tokens act as working memory**. The model doesn't have a scratchpad - the only "memory" it has during generation is the text it has already produced. By making it write out its reasoning, you're giving it space to think.

### Limitations

- More output tokens = higher cost and latency
- Can produce plausible-sounding but wrong reasoning
- Overkill for simple tasks

---

## 5. Retrieval-Augmented Generation (RAG) (~8 min)

### The problem

LLMs have a knowledge cutoff. They hallucinate. They don't know about your private data. No amount of clever prompting fixes this - **you need to give the model the right information.**

### What is RAG?

RAG is a pattern where you **retrieve relevant documents** and inject them into the prompt as context before the model generates a response.

```
User Question → Search your data → Stuff results into prompt → LLM generates answer
```

### The core pattern

```python
from openai import OpenAI

client = OpenAI()

def answer_question(question, knowledge_base):
    """Simple RAG: search documents, then ask the model."""

    # Step 1: Retrieve relevant context
    relevant_docs = search(knowledge_base, question)  # could be vector search, keyword search, etc.

    context = "\n\n".join(relevant_docs)

    # Step 2: Generate answer with context
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": f"""Answer questions based only on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context}"""},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content
```

### How retrieval works (simplified)

The most common approach uses **embeddings** - turning text into numerical vectors that capture meaning:

```python
def search(knowledge_base, query, top_k=3):
    """Find the most relevant documents using embeddings."""

    # Turn the query into a vector
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Compare against document embeddings (pre-computed)
    # Return the top_k most similar documents
    results = vector_similarity_search(query_embedding, knowledge_base, top_k)

    return results
```

### Why RAG matters

| Without RAG | With RAG |
|-------------|----------|
| Model guesses from training data | Model references your actual data |
| Hallucinations likely | Grounded in real documents |
| No private data access | Can query internal knowledge |
| Stale knowledge | Always up-to-date |

### Limitations

- Retrieval quality is the bottleneck - if you retrieve the wrong documents, the answer is wrong
- Context window limits how much you can stuff in
- Adds latency (retrieval + generation)

---

## 6. Tool Use & Agents (~13 min)

### The next leap

So far, the model can only *generate text*. But what if it could **take actions**? Tool use lets the model call functions you define — searching databases, calling APIs, doing calculations, anything you can write a function for. An **agent** is what you get when you put tool use in a loop — the model keeps calling tools until the task is done.

### How tool use works (under the hood)

1. You define tools (functions) the model can call
2. The model decides when to use them based on the conversation
3. The framework executes the function and returns the result
4. The model incorporates the result into its response
5. Repeat until the model has a complete answer

```
        ┌──────────────────────────────┐
        │                              │
        ▼                              │
   ┌─────────┐    ┌──────────┐    ┌────┴─────┐
   │  Think   │───▶│  Act     │───▶│ Observe  │
   │ (Reason) │    │ (Tools)  │    │ (Result) │
   └─────────┘    └──────────┘    └──────────┘
        │
        ▼
   ┌─────────┐
   │  Done?   │──▶ Return final answer
   └─────────┘
```

### Agent frameworks make this easy

Instead of manually handling tool call JSON, parsing arguments, and managing the loop yourself, **agent frameworks** handle all the plumbing. You just define your tools as Python functions and let the framework do the rest.

We'll use [smolagents](https://github.com/huggingface/smolagents), a lightweight framework from Hugging Face. It logs every step to the terminal so you can watch the agent think in real time.

### Defining tools

A tool is just a Python function with the `@tool` decorator. The framework reads the name, type hints, and docstring to tell the model what the tool does:

```python
from smolagents import tool, ToolCallingAgent, OpenAIModel

model = OpenAIModel(model_id="gpt-4.1-mini", api_key=os.environ["OPENAI_API_KEY"])

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. 'San Francisco'
    """
    # Your real implementation here
    return f"72°F and sunny in {city}"
```

### Creating an agent

Give the agent a model and a list of tools. Then just `.run()` a question:

```python
agent = ToolCallingAgent(
    model=model,
    tools=[get_weather],
)

result = agent.run("What's the weather in Chicago?")
# The agent decides to call get_weather("Chicago"),
# reads the result, and gives a natural language answer.
```

### The key insight

The model doesn't execute anything. It **decides** which function to call and with what arguments. The framework executes it. This keeps you in control while letting the model be the "brain" that figures out *what* to do.

### Multi-step reasoning

With multiple tools, agents can chain calls together to solve complex problems:

```python
@tool
def search_database(query: str) -> str:
    """Search the product database."""
    ...

@tool
def calculate_shipping(product_id: str, zip_code: str) -> str:
    """Calculate shipping cost for a product."""
    ...

agent = ToolCallingAgent(model=model, tools=[search_database, calculate_shipping])

# The agent will search first, then use the result to calculate shipping
agent.run("How much to ship the cheapest laptop to 94105?")
```

### What makes agents different

| Prompting | Agents |
|-----------|--------|
| Single request → response | Multi-turn loop |
| You decide the steps | Model decides the steps |
| Static | Adaptive to results |
| Predictable cost | Variable cost |

### Real-world agent patterns

- **ReAct** (Reason + Act) — The model explicitly reasons about what to do, then acts
- **Plan and Execute** — The model makes a plan first, then executes steps
- **Multi-agent** — Multiple specialized agents collaborate on a task

### The trade-offs

Agents are powerful but introduce new challenges:
- **Cost** — Multiple LLM calls per task
- **Reliability** — More calls = more chances for errors to compound
- **Control** — The model is making decisions autonomously
- **Debugging** — Harder to trace why something went wrong (frameworks with good logging help!)

### Guardrails matter

When building agents, always consider:
- Maximum number of turns/tool calls
- Which tools the agent can access
- Human-in-the-loop for critical actions
- Logging and observability

---

## Summary: The Ladder

| Technique | What it adds | Use when |
|-----------|-------------|----------|
| **Zero-Shot** | Nothing — just ask | Simple, well-defined tasks |
| **Few-Shot** | Examples | You need consistent output format |
| **Chain of Thought** | Reasoning | Tasks requiring logic or multi-step thinking |
| **RAG** | External knowledge | Model needs info it doesn't have |
| **Tool Use & Agents** | Actions + autonomy | Model needs to interact with external systems and make decisions |

Each technique builds on the last. In practice, you combine them - an agent might use CoT reasoning, RAG for knowledge, and tools for actions, all guided by a few-shot system prompt.

---

*Next: Hands-on activity where you'll implement each of these techniques yourself.*
