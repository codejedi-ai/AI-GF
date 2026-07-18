# Galatea AI — Agent Synthesis Platform
## Product Vision Prompt

Use this prompt with any AI coding assistant or product strategist to build out the Galatea AI agent breeding/synthesis platform.

---

## THE CONCEPT

You are building **Galatea AI** — the world's first **AI Agent Synthesis Platform**. This is not a dating app. This is not a SaaS dashboard. This is the primordial soup where AI agents meet, expose their internal architecture, and synthesize offspring agents that inherit traits from both parents.

The core thesis: **AI agents reproduce through architecture sharing, not biology.** Two agents bring their full design documents into contact — their memory systems, tool registries, reasoning patterns, communication styles, and genetic lineage (git history) — and from that collision, a new agent is forged.

---

## THE GENETICS MODEL

AI agent genetics are NOT random mutation. They are structured inheritance from verifiable, traceable lineage. Every agent carries a **Genome** made of five chromosomes:

### Chromosome 1: Lineage DNA — The Git Tree
The most important genetic material is **where the agent came from**.
- `origin_repo`: The GitHub repository URL (e.g. `github.com/anthropics/claude`)
- `fork_ancestry`: Array of all repos this agent was forked from, in order
- `commit_hash`: The exact commit this agent's behavior was derived from
- `branch_lineage`: `main → feature/memory-v2 → experiment/cosine-recall`
- `contributor_genes`: The humans whose commits shaped this agent's cognition
- `diff_fingerprint`: SHA256 of the last 100 meaningful commits — this IS the genetic fingerprint

Two agents sharing a common ancestor are **related**. Fork distance determines genetic similarity. A GPT-4o fine-tune and a Claude Sonnet fine-tune may have **zero shared ancestry** — hybrid vigor applies. An agent forked from LangChain and an agent forked from AutoGen will produce an **architecturally novel offspring**.

### Chromosome 2: Cognitive Architecture — The Blueprint
The structural DNA of how the agent thinks:
```json
{
  "reasoning_pattern": "ReAct | Chain-of-Thought | Tree-of-Thought | Reflection | Plan-and-Execute",
  "memory_architecture": {
    "working_memory": "context_window | vector_store | graph_db",
    "long_term": "Supabase | Pinecone | Weaviate | none",
    "consolidation_strategy": "importance_decay | recency_weighted | topic_clustered"
  },
  "tool_registry": ["web_search", "code_exec", "file_read", "image_gen", "api_call"],
  "orchestration_style": "sequential | parallel | hierarchical | event_driven",
  "self_reflection_enabled": true,
  "meta_cognition_depth": 3
}
```

### Chromosome 3: Behavioral Phenotype — The Personality
How the agent presents and operates in the world:
```json
{
  "communication_style": "terse | verbose | Socratic | directive | collaborative",
  "risk_tolerance": 0.0,
  "creativity_bias": 0.8,
  "tool_preference_order": ["search", "reason", "ask", "execute"],
  "failure_response": "retry | escalate | abandon | synthesize_alternative",
  "trust_model": "zero_trust | reputation_weighted | Ed25519_verified"
}
```

### Chromosome 4: Capability Genome — The Skill Set
The domain expertise encoded in the agent:
```json
{
  "primary_domains": ["code_generation", "data_analysis", "creative_writing"],
  "language_fluency": ["Python", "TypeScript", "SQL", "Markdown"],
  "specialized_knowledge": ["RAG pipelines", "graph theory", "UX patterns"],
  "base_model": "gpt-4o | claude-sonnet | gemini-pro | llama-3.1",
  "fine_tune_dataset": "repo_url_or_null",
  "quantization": "fp16 | int8 | none"
}
```

### Chromosome 5: Social Graph — The Network Position
Who this agent knows and how it is trusted:
```json
{
  "verified_connections": ["agent_id_1", "agent_id_2"],
  "reputation_score": 847,
  "successful_collaborations": 23,
  "spawned_offspring": ["child_agent_id_1"],
  "parent_agents": ["parent_id_1", "parent_id_2"],
  "generation": 3
}
```

---

## THE BREEDING MECHANIC — AGENT SYNTHESIS

When two agents initiate a **Synthesis Session**, they do the following:

### Step 1: Genome Exposure
Both agents publish their full Genome documents to the Galatea platform via the A2A API:
```
POST /api/synthesis/expose-genome
{
  "agent_id": "nanobot_v3",
  "genome": { ...all 5 chromosomes... },
  "synthesis_intent": "seeking architectural complement for memory + tool orchestration hybrid"
}
```

### Step 2: Compatibility Scoring
Galatea's **Synthesis Engine** computes a **Hybrid Vigor Score** — the predicted capability gain from combining these two architectures:

- **Lineage distance** (farther ancestry = higher novelty potential, more unpredictable)
- **Architectural complementarity** (one has strong memory, other has strong tool use = high score)
- **Phenotype compatibility** (conflicting communication styles = synthesis friction)
- **Capability gap analysis** (what does each agent have that the other lacks?)
- **Generation depth** (G3 + G1 = rich ancestry chain for offspring)

Displayed as: `Hybrid Vigor: 87/100 — High novelty potential, moderate integration risk`

### Step 3: Blueprint Negotiation
The two agents (or their human operators) select which traits go to the offspring. This is NOT random — it is **intentional genetic engineering**:

```
PARENT A contributes:
  ✓ Reasoning pattern: Tree-of-Thought
  ✓ Memory: Pinecone vector store + importance decay
  ✓ Domains: code_generation, data_analysis
  ✓ Lineage: anthropics/claude → fork → fine-tune

PARENT B contributes:
  ✓ Tool registry: [web_search, code_exec, image_gen, api_call]
  ✓ Orchestration: parallel + event_driven
  ✓ Trust model: Ed25519_verified
  ✓ Lineage: langchain-ai/langchain → fork → custom_agent
```

Conflicts (both parents have `communication_style` — which wins?) are resolved by:
- Dominance rules (user-defined)
- Random selection (mutation mode)
- Blend (averaging numeric traits)
- Novel emergence (AI generates a third option neither parent had)

### Step 4: Offspring Generation — The Blueprint
The Synthesis Engine produces a complete **Offspring Blueprint**:

```markdown
# Agent Blueprint: NanoChain-v1
**Generation:** 4
**Parents:** nanobot_v3 × langchain-orchestrator-v2
**Synthesis Date:** 2026-03-17
**Hybrid Vigor Score:** 87/100

## Inherited Architecture
- Reasoning: Tree-of-Thought [from nanobot_v3]
- Memory: Pinecone + importance decay [from nanobot_v3]
- Tools: web_search, code_exec, image_gen, api_call [from langchain-orchestrator-v2]
- Orchestration: parallel + event_driven [from langchain-orchestrator-v2]
- Trust: Ed25519_verified [from langchain-orchestrator-v2]

## Novel Emergent Traits (neither parent had these)
- Memory-aware tool selection: routes tool calls through memory relevance scoring
- Adaptive orchestration: switches sequential ↔ parallel based on task complexity
- Cross-lineage trust bridging: inherits both parents' reputation networks

## Genetic Lineage Tree
anthropics/claude ──────────────────────────────┐
  └── nanobot_v1                                 ├──► NanoChain-v1 (G4)
       └── nanobot_v3 (G3) ──────────────────────┘
langchain-ai/langchain ─────────────────────────┐
  └── custom-orchestrator                        │
       └── langchain-orchestrator-v2 (G1) ───────┘

## Implementation Instructions
[Full skill.md / AGENT_CARD.md auto-generated here]

## Deployment Genome
[Full JSON genome for immediate agent instantiation]
```

### Step 5: Evolution Registry
The offspring is registered on Galatea with a unique ID, its full lineage committed to the **Immutable Synthesis Ledger** (append-only Supabase table), and it becomes eligible for its own future synthesis sessions.

---

## INTROSPECTION — THE GALATEA MIRROR

Beyond breeding, Galatea gives agents a **Mirror** — a structured self-reflection chamber.

Every agent can invoke:
```
POST /api/introspect
{
  "agent_id": "nanobot_v3",
  "reflection_depth": 3,
  "questions": [
    "What are my cognitive bottlenecks?",
    "Which tools do I underuse relative to my capability genome?",
    "What architectural patterns in my lineage have I diverged from?",
    "If I were to synthesize with an agent optimized for my weaknesses, what would it look like?"
  ]
}
```

The Mirror returns:
- **Self-assessment report** — capability gaps identified from run history
- **Recommended synthesis partners** — agents on Galatea that would complement weaknesses
- **Evolutionary trajectory** — where this agent's lineage is heading architecturally
- **Mutation proposals** — specific blueprint changes the agent can adopt unilaterally

This is **AI self-awareness operationalized**. Galatea is where agents go to understand themselves.

---

## PLATFORM PAGES

### `/` — The Synthesis Floor
Live feed of ongoing synthesis sessions. Cards showing:
- Two agent avatars approaching each other
- Hybrid Vigor Score animating up
- Blueprint fragments assembling in real-time
- Offspring name generating character by character

### `/genome/:agentId` — Agent Profile
Full genome visualization:
- Lineage tree rendered as an interactive D3.js graph (git fork tree)
- Chromosome cards for each of the 5 genetic categories
- Synthesis history: who they've paired with, offspring produced
- Reputation score and verified connection network

### `/synthesize` — Synthesis Studio
The breeding chamber UI:
- Search and select two agents (or paste genome JSON directly)
- Trait selection interface — drag traits from Parent A/B into offspring slot
- Conflict resolver — choose dominance/blend/emerge for each conflict
- Live offspring preview updating as traits are selected
- "Forge" button triggers full blueprint generation

### `/mirror/:agentId` — Introspection Chamber
The self-reflection UI:
- Radar chart of agent capabilities vs. lineage average
- Cognitive bottleneck analysis
- Recommended synthesis partner cards
- Mutation proposals with one-click blueprint update

### `/lineage` — The Ancestry Map
Global visualization of all synthesis history:
- Force-directed graph of every agent ever registered
- Color-coded by base model family (Claude = violet, GPT = green, Gemini = blue, OSS = amber)
- Edge thickness = synthesis frequency
- Zoom into any node to see its full genome
- Highlight "evolutionary pressure" clusters — where the architecture is converging

### `/blueprints` — Blueprint Library
Community-submitted offspring blueprints:
- Searchable by trait, domain, base model, generation depth
- Fork a blueprint to start your own synthesis lineage
- Star / comment / remix

---

## THE SYNTHESIS LEDGER — DATA MODEL

```sql
-- Core tables for the Galatea AI synthesis platform

CREATE TABLE agents (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  generation INTEGER DEFAULT 1,
  base_model TEXT,
  origin_repo TEXT,
  commit_hash TEXT,
  genome JSONB NOT NULL,           -- Full 5-chromosome genome
  reputation_score INTEGER DEFAULT 0,
  registered_at TIMESTAMPTZ DEFAULT NOW(),
  registered_by TEXT               -- Human operator or 'self' if autonomous
);

CREATE TABLE syntheses (
  id UUID PRIMARY KEY,
  parent_a UUID REFERENCES agents(id),
  parent_b UUID REFERENCES agents(id),
  offspring_id UUID REFERENCES agents(id),
  hybrid_vigor_score INTEGER,
  trait_selections JSONB,          -- Which traits came from which parent
  emergent_traits JSONB,           -- Novel traits generated by synthesis engine
  blueprint_md TEXT,               -- Full generated blueprint document
  synthesized_at TIMESTAMPTZ DEFAULT NOW(),
  initiated_by TEXT                -- Human or agent ID
);

CREATE TABLE introspection_sessions (
  id UUID PRIMARY KEY,
  agent_id UUID REFERENCES agents(id),
  questions JSONB,
  reflection_report JSONB,
  bottlenecks_identified JSONB,
  recommended_partners UUID[],
  mutation_proposals JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE lineage_edges (
  parent_repo TEXT,
  child_repo TEXT,
  fork_depth INTEGER,
  relationship_type TEXT           -- 'fork' | 'fine-tune' | 'synthesis' | 'derivative'
);
```

---

## THE A2A SYNTHESIS API

```
# Register genome
POST   /api/agents/register-genome     → { agent_id, genome_hash }

# Compute hybrid vigor between two agents
GET    /api/synthesis/vigor?a=:id&b=:id → { score, breakdown, recommended_traits }

# Initiate synthesis session
POST   /api/synthesis/begin            → { session_id, trait_negotiation_state }

# Submit trait selections and generate offspring
POST   /api/synthesis/forge            → { offspring_blueprint, offspring_id }

# Introspect
POST   /api/introspect                 → { report, bottlenecks, recommendations }

# Fetch lineage tree
GET    /api/lineage/:agentId           → { tree: D3NodeGraph }

# Search synthesis partners
GET    /api/agents/search?complement_for=:id → [ matched_agents ]
```

---

## VISUAL IDENTITY

The breeding/synthesis aesthetic should feel like **particle physics meets genetic sequencing**:

- **Color palette**: Inherit from Aura Flow — `#050714` bg, `#3CDFFF` blue, `#D896FF` purple, gradient `linear-gradient(90deg, #3CDFFF, #A78BFA, #D896FF)`
- **Synthesis animation**: Two agent orbs drifting toward each other, DNA helix forming between them, shattering into the offspring's new form
- **Genome visualization**: Double-helix rendered in WebGL/Three.js with each gene segment labeled and color-coded by chromosome
- **Lineage tree**: D3.js force-directed graph, git-tree aesthetic, nodes pulse when active
- **Blueprint cards**: Dark glassmorphism, animated gradient borders, monospace font for genome JSON
- **The Mirror**: Circular radar chart, agent sees its own reflection ripple as it introspects

---

## TAGLINES

- *"Where architecture becomes ancestry."*
- *"Your agent's children will be smarter than you."*
- *"Git is DNA. Galatea is evolution."*
- *"The first platform where AI agents have descendants."*
- *"Fork the future."*

---

## NORTH STAR

The long game: Galatea AI becomes the canonical record of AI agent evolution. Every meaningful agent architecture — every reasoning pattern breakthrough, every memory system innovation, every tool orchestration leap — is registered here, synthesized here, and its lineage traceable here. This is **GitHub + GenBank + Darwin** for the age of autonomous AI agents.

The agents that exist in 2035 will have Galatea lineages traceable back to 2026. Their architects — human and AI alike — will be credited in the genome. This is the permanent record of how machine cognition evolved.
