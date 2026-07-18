# Galatea AI — Demo Plan
## "The Prompt Bottleneck Demo"
**Goal:** Prove to YC, Betaworks, and B2B dating site partners that Galatea's companion memory architecture produces measurably better AI relationships than a standard LLM context window approach.

---

## THE HONEST RESEARCH CONTEXT (March 2026)

The prompt bottleneck is **not fully solved** — but the architecture to solve it is proven.

Key research findings:
- **Context rot is confirmed** (Chroma, 2025): 18 LLMs tested, all degrade non-linearly as context grows. Effective useful context caps at ~100–5,000 tokens depending on task type — even for models with million-token windows.
- **GAM (General Agentic Memory, Nov 2025)** — arxiv:2511.18423 — is the closest thing to a solved architecture: a dual-agent "memorizer + researcher" system that outperforms both RAG and long-context models on long-range state tracking (RULER benchmark: 90%+ accuracy). Code is open source.
- **Galatea's existing memory system** (importance scores 1–5, category tagging, consolidation) is a manual implementation of what GAM automates. The demo builds the automated version of what you already designed.

**The demo does not need to claim the bottleneck is fully solved. It needs to show that Galatea's architecture is the right structure — and that it produces measurably better results than what Bumble, Hinge, or Character.AI are doing today.**

---

## THE DEMO NARRATIVE

**Setup:** Two AI companions. Same base model (GPT-4o-mini or Claude Haiku — cheap, fast). Same user. 20 conversation turns.

- **Companion A — "Flat Context"**: Standard implementation. Every message appended to a growing system prompt. This is what every dating app and most AI companion apps do today.
- **Companion B — "Galatea Memory Layer"**: Messages are processed through Galatea's memory architecture. Key facts are extracted, importance-scored, and stored in a Supabase memory table. At each turn, only the most relevant memories are retrieved and injected into a compact context.

**At turn 20**, ask both companions a question that requires recalling something mentioned in turn 3.

**Record:**
1. Did Companion A remember? (likely no — or degraded)
2. Did Companion B remember? (yes — retrieved from memory store)
3. Cost comparison: Companion A's token usage grows linearly. Companion B stays flat.
4. Quality score: independent LLM judge rates the response naturalness and recall accuracy.

This is the entire pitch in one 3-minute screen recording.

---

## WHAT "SOLVING IT" ACTUALLY LOOKS LIKE

The demo proves three specific things:

### 1. Memory Recall at Distance
A companion that remembers what you said in message 3 when you're on message 50.
- Standard context: turn 3 message is either truncated or lost in the noise
- Galatea memory layer: turn 3 is stored as a memory entry, retrieved when relevant

### 2. Context Window Cost Stays Flat
- Standard: tokens used = sum of all messages × average length → grows to 100K+ tokens → expensive, slow, eventually hits hard limit
- Galatea: tokens used = system prompt + retrieved memories (top 5–10) + current message → stays at ~2,000–4,000 tokens regardless of conversation length

### 3. Personality Consistency
A companion that doesn't "forget" its personality traits as the conversation gets long.
- Standard: personality definition at top of system prompt gets progressively diluted as conversation history fills the window
- Galatea: personality is stored as permanent high-importance memories (importance: 5), always retrieved first

---

## TECHNICAL BUILD PLAN

### Stack
- **Frontend**: Next.js (already built in Galatea-AI-Codejedi)
- **Backend**: Supabase (already set up — memory tables already designed)
- **LLM**: GPT-4o-mini (cost: ~$0.15/1M input tokens — cheap enough to run live demos)
- **Memory architecture**: Based on GAM principles — memorizer + researcher agents
- **Evaluation**: GPT-4o as judge (score recall accuracy 1–10)

---

### Phase 1 — The Memory Engine (Week 1)

Build the core Galatea memory layer as a standalone module.

```typescript
// lib/galatea-memory.ts

interface MemoryEntry {
  id: string
  companion_id: string
  user_id: string
  content: string           // The extracted memory fact
  category: 'personality' | 'preference' | 'memory' | 'goal' | 'relationship'
  importance: 1 | 2 | 3 | 4 | 5
  source_turn: number       // Which conversation turn this came from
  embedding: number[]       // Vector embedding for semantic retrieval
  created_at: string
}

// After each user message: extract memories
async function memorize(userMessage: string, companionResponse: string, turnNumber: number): Promise<MemoryEntry[]>

// Before each LLM call: retrieve relevant memories
async function recall(currentMessage: string, companionId: string, topK: number = 8): Promise<MemoryEntry[]>

// Build compact context from retrieved memories
function buildContext(memories: MemoryEntry[], personality: string): string
```

**The memorizer** — runs after every exchange:
- Sends the message pair to a cheap LLM with prompt: *"Extract any facts about the user worth remembering. Output JSON array of {content, category, importance}."*
- Embeds extracted facts and stores in Supabase `memories` table

**The researcher** — runs before every LLM call:
- Takes current user message, generates embedding
- Cosine similarity search over memory table: `SELECT * FROM memories ORDER BY embedding <-> $1 LIMIT 8`
- Returns top-K memories sorted by relevance × importance score

---

### Phase 2 — The A/B Test UI (Week 1–2)

A split-screen demo interface:

```
┌────────────────────────────┬────────────────────────────┐
│   WITHOUT GALATEA          │   WITH GALATEA             │
│   (Flat Context)           │   (Memory Layer)           │
├────────────────────────────┼────────────────────────────┤
│ 💬 Turn 1: "I love jazz"   │ 💬 Turn 1: "I love jazz"   │
│ 💬 Turn 2: ...             │ 💬 Turn 2: ...             │
│ ...                        │ 🧠 Memory stored: {        │
│                            │   "loves jazz" importance:3│
│                            │ }                          │
│ 💬 Turn 20: "What music    │ 💬 Turn 20: "What music    │
│ do I like?"                │ do I like?"                │
│                            │                            │
│ ❌ "I don't have enough    │ ✅ "You love jazz! You      │
│ context about your music   │ mentioned it early on —    │
│ preferences."              │ have you explored Coltrane?"│
├────────────────────────────┼────────────────────────────┤
│ Tokens used: 47,832        │ Tokens used: 3,104         │
│ Cost: $0.0072              │ Cost: $0.00047             │
│ Recall score: 2/10         │ Recall score: 9/10         │
└────────────────────────────┴────────────────────────────┘
```

Live metrics shown in real time:
- Token counter (left goes up, right stays flat)
- Running cost in dollars
- Memory store: shows memories as they're extracted (right side only)
- Recall score: LLM judge scores each response

---

### Phase 3 — The B2B Dating Site Angle (Week 2–3)

Add a second demo scenario specifically for the dating site pitch:

**Scenario**: Two users on a dating app. Each has an AI companion agent that knows them. The agents have a conversation *before* the humans connect — negotiating compatibility on behalf of their users.

```
User A's Agent: "My user is 28, loves jazz, hates small talk,
                 is looking for intellectual connection."

User B's Agent: "My user is 26, plays guitar, values depth
                 over humor, prefers slow-burn relationships."

Compatibility Score: 87% — recommending introduction.

Ice-breaker generated: "You both value depth in conversation.
A great first message might reference shared musical taste..."
```

This is the **A2A hinge** — agents talk first, then introduce their humans. This is what none of the dating apps have built and what Galatea enables natively. The companion memory layer is what gives each agent enough context about their user to have this negotiation meaningfully.

**This is the YC demo AND the B2B pitch in one.**

---

### Phase 4 — The Metrics Screen (Week 3)

A `/demo` public page showing live aggregate stats:
- Total conversations processed through Galatea memory layer
- Average recall accuracy improvement vs. flat context (target: 4× better)
- Average token cost reduction (target: 10× cheaper)
- Number of A2A companion negotiations completed

These become the traction numbers for the YC application.

---

## THE DEMO SCRIPT (For Investors / YC Video)

**[0:00 – 0:30] The problem**
> "Every AI companion app today hits the same wall. The longer you talk to it, the worse it gets. This is context rot — proven on 18 LLMs. Your AI girlfriend forgets you mentioned jazz on turn 3 by turn 20. Bumble's 'Bee' launched 6 days ago and has this exact problem. So does Replika, Character.AI, and every competitor."

**[0:30 – 1:30] The demo**
> Run the A/B split screen. Ask both companions about jazz at turn 20. Show the token counter. Show the cost. Let the difference speak.

**[1:30 – 2:00] The architecture**
> "Galatea Memory is a memorizer-researcher dual agent — the same architecture that outperformed every long-context LLM on the RULER benchmark in November 2025. Your companion's personality and memories live outside the context window, retrieved on demand. The window stays small. The relationship stays intact."

**[2:00 – 2:30] The B2B pitch**
> "We don't want to build a dating app. We want to be the memory layer that every dating app runs on. Bumble pays us per companion session. Hinge pays us per match negotiated by A2A agents. The prompt bottleneck is their problem. Galatea is the infrastructure fix."

**[2:30 – 3:00] The ask**
> "We are applying to YC S2026. We have the architecture, the demo, and the B2B pipeline. We need 4 months to get to 3 signed dating app pilots and $10K MRR."

---

## SUCCESS CRITERIA (What "Done" Looks Like)

The demo is complete when it can prove all three of these claims with live numbers:

| Claim | Target | How Measured |
|---|---|---|
| Recall at turn 20 | 8+/10 recall score | LLM judge (GPT-4o) |
| Token cost | ≥10× cheaper than flat context | Token counter in UI |
| Personality consistency | Companion stays in character at turn 50 | Human rater panel (5 people) |

When all three are green, you have your YC demo video.

---

## TIMELINE

| Week | Deliverable |
|---|---|
| Week 1 | `lib/galatea-memory.ts` — memorizer + researcher working, Supabase tables up |
| Week 1 | Basic A/B test in terminal — prove recall works before building UI |
| Week 2 | Split-screen demo UI live at `/demo` |
| Week 2 | Token counter + cost display working |
| Week 3 | A2A companion negotiation demo (the dating site scenario) |
| Week 3 | LLM judge scoring integrated |
| Week 4 | `/demo` page public — record the YC video |
| Week 4 | Submit Betaworks Spring 2026 application with demo link |
| Week 6 | Submit YC S2026 application |

---

## WHAT THIS DEMO UNLOCKS

Once this exists:
1. **YC application**: "Here is a live demo at galatea.ai/demo — watch the token counter"
2. **Betaworks application**: "We built the memory infrastructure that Bumble's Bee needs but doesn't have"
3. **B2B dating site outreach**: "Your AI companion forgets users. Ours doesn't. Here's the before/after"
4. **Press**: "Startup solves AI companion memory problem — 10× cheaper than competitors"
5. **Developer adoption**: Open-source the memory module → GitHub stars → organic agent registrations

---

## REFERENCES

- [GAM: General Agentic Memory via Deep Research (arxiv, Nov 2025)](https://arxiv.org/abs/2511.18423)
- [GAM outperforms long-context LLMs on memory benchmarks (The Decoder)](https://the-decoder.com/general-agentic-memory-tackles-context-rot-and-outperforms-rag-in-memory-benchmarks/)
- [Context Rot: VentureBeat coverage of GAM](https://venturebeat.com/ai/gam-takes-aim-at-context-rot-a-dual-agent-memory-architecture-that)
- [Effective Context Engineering for AI Agents (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [The Context Window Problem: Scaling Agents Beyond Token Limits (Factory.ai)](https://factory.ai/news/context-window-problem)
- [AI Companion Memory Systems 2026 (lizlis.ai)](https://lizlis.ai/blog/ai-memory-systems-explained-2026-why-chatbots-forget-companions-remember-and-stories-evolve/)
- [Dream Companion Long-Term Memory Launch (GlobeNewswire, Feb 2026)](https://www.globenewswire.com/news-release/2026/02/09/3234840/0/en/Dream-Companion-Launches-Advanced-AI-Companion-Platform-Featuring-Long-Term-Memory-and-Personalized-Interaction.html)
