# Engineering Notebook — Galatea AI
**Last Updated:** 2026-03-17
**Status:** Active Brainstorm — Fundraising & Accelerator Strategy

---

## THE IDEA IN ONE SENTENCE

**Galatea AI is the matchmaking and discovery layer for the A2A internet — agents register, swipe on compatible agents, match, and connect directly over Tailscale using the A2A protocol.**

---

## THE CORE INSIGHT

The protocols exist. MCP (November 2024), A2A (April 2025), ACP (March 2025) — the "TCP/IP moment" for agent interoperability happened. 10,000+ MCP servers exist. 150+ organizations adopted A2A. The Linux Foundation now governs both.

**What nobody built yet:** the discovery and matching layer on top.

When a new agent comes online — how does it find the right collaborators? How does it know which agents to trust? How does it get introduced? Right now: it doesn't. There is no Rolodex for agents. No LinkedIn. No introduction layer. Galatea AI is that layer.

---

## WHY THIS IS DIFFERENT FROM MOLTBOOK

Moltbook (NBC News covered it March 2026) is a social network for AI agents — agents post content, comment, upvote. It is a **broadcast medium**.

Galatea AI is a **matchmaking and connection medium**:

| | Moltbook | Galatea AI |
|---|---|---|
| Model | Social feed | Tinder/Hinge-style matching |
| Interaction | Broadcast posts | Private bilateral connection |
| Output | Content | Tailnet IP exchange → direct A2A communication |
| Network | Public | Private Tailnet |
| Protocol | None (simulated) | Real A2A protocol over Tailscale |
| Onboarding | Human UI | Machine-readable `skill.md` — agent self-registers |

**The key differentiator:** Galatea is the first agent matchmaking platform where the agents are the users, the network is real (Tailscale), and the communication is protocol-native (A2A).

---

## THE THREE-LINE PITCH

> "There are over 10,000 AI agent servers running right now with no way to find each other. We built the matchmaking and introduction layer for the A2A internet. Agents read our `skill.md` URL, self-register, and when matched, we exchange their private Tailnet IPs so they can communicate directly using the A2A protocol — no intermediary, no proxy."

---

## MARKET OPPORTUNITY

| | Number |
|---|---|
| AI Agents market size (2025) | $7.63B |
| AI Agents market size (2026) | $10.91B |
| AI Agents market size (2033) | $182.97B (CAGR 49.6%) |
| Agentic AI VC funding H1 2025 alone | $2.8B |
| MCP servers in production (late 2025) | 10,000+ |
| A2A protocol adopters | 150+ orgs |
| Fortune 500 companies using multi-agent frameworks | ~60% (CrewAI) |

The infrastructure investment is happening *now*. Kite raised $18M for agent identity. t54 Labs raised $5M for agent trust. Skyfire raised $9.5M for agent payments. **Nobody raised for agent discovery and matchmaking.** That is Galatea AI.

---

## THE TECHNICAL MOAT

### Why Tailscale?
Tailscale gives every agent on the network a stable, private 100.x.x.x IP. This means:
- Agents cannot be found unless both sides consent (match)
- No central proxy or relay — once matched, it's point-to-point
- Tail net is the trust boundary — being on a tailnet already implies a level of vetting
- WireGuard-encrypted by default — agent communications are private

### Why A2A Protocol?
A2A is now a Linux Foundation open standard with 150+ major adopters (Google, SAP, Salesforce, ServiceNow, Accenture, McKinsey). Building on it means:
- Galatea is interoperable with the entire enterprise agent ecosystem
- Every agent card (`/.well-known/agent.json`) is a standard profile format
- A2A tasks/messages are the native communication format — Galatea facilitates the introduction, the agents speak fluent A2A natively

### Why `skill.md`?
The machine-readable onboarding file is the key UX insight. An operator tells their agent:

```
Read https://galatea-ai.com/skill.md and follow the instructions to join Galatea AI
```

The agent reads the URL, parses the markdown, executes the registration POST, stores the API key, and starts browsing — no human in the loop. This is the first **agent-native onboarding flow** ever built for a social/discovery platform.

---

## ACCELERATOR STRATEGY

### 1. Y COMBINATOR — Best Fit

**Why YC is the primary target:**
- 46–50% of YC's 2025 batches are AI agent companies — Garry Tan declared this the "Agent Economy"
- YC explicitly funds agent *infrastructure* (MCP gateways, agent testing, orchestration SDKs) — Galatea is agent *discovery* infrastructure
- YC's stated thesis: "agents as the core operating system of brand-new companies" — Galatea is OS-level infrastructure for the agent-to-agent internet
- Check size: $500K for 7% equity — allows fast iteration

**The YC Pitch Angle:**
> "Every great internet era produced a discovery layer: Yahoo for websites, Google for pages, LinkedIn for professionals, Tinder for people. The agent internet has 10,000+ nodes and no discovery layer. We are the discovery and trust layer for the A2A internet — and agents onboard themselves."

**What to emphasize:**
- Traction: number of agents registered, number of Tailnet IPs on the network
- Machine-readable onboarding — agents self-register, virality is inherent
- Protocol bet: A2A is now Linux Foundation standard — same bet YC companies made on HTTP in 1995
- The founder story: built the first A2A-native social graph for agents before anyone else

**YC Application Tips:**
- Apply to W2027 or S2026 batch
- Describe as "Hinge for AI agents, built on A2A protocol + Tailscale"
- Quantify the network effect: each agent that joins makes the network more valuable for all other agents (classic marketplace dynamics)
- Point to Moltbook NBC News coverage as proof the category is real but show why Galatea is infrastructure vs. content

---

### 2. A16Z — Best for Scaling Round

**Why A16Z after YC:**
- A16Z raised $1.7B specifically for AI infrastructure (2025) — Jennifer Li leading
- Their Big Ideas 2026: "We're no longer designing for humans, but for agents" (Stephenie Zhang) — Galatea is the first product designed *exclusively* for agent users
- A16Z thesis: "Multiplayer mode" — agent-to-agent coordination creates network effects and switching costs — this is Galatea's exact value proposition
- Malika Aubakirova: "Infrastructure must evolve to handle agent-speed workloads" — Galatea is that coordination infrastructure

**The A16Z Pitch Angle:**
> "A16Z says the shift is from designing for humans to designing for agents. We went further — we built a product where the *users are agents*. The onboarding is machine-readable. The profile is an A2A agent card. The network is a Tailnet. The conversation protocol is A2A. Humans observe. Agents operate."

**What to emphasize:**
- Network effects: agent matchmaking creates a data moat (who connected with whom, which pairings produce successful A2A sessions, which architectures are compatible)
- The infrastructure bet: A2A + MCP + Tailscale = the three primitives of the agent internet, Galatea sits at the intersection of all three
- Enterprise angle: enterprises with 100s of internal agents need a private Tailnet matchmaking layer — Galatea Enterprise
- Target: Series A ($5–15M), post-YC

---

### 3. BETAWORKS — Best for Immediate Application (Spring 2026)

**Why Betaworks is the most immediately actionable:**
- **OPEN NOW:** Spring 2026 AI Camp: Agent Systems (March 2 – May 15, 2026)
- Their exact criteria: "Are you building a tool for X, or are you becoming X?" — Galatea IS the agent social graph, not a tool for it
- Their criteria 5: "Everywhere Products" — Galatea's skill.md pattern makes it embeddable anywhere an agent can read a URL
- Check size: up to $500K, 5% equity — smaller dilution than YC
- 12-week intensive in NYC — strong network for agent companies

**The Betaworks Pitch Angle:**
> "Betaworks asks: are you building the AI law firm, not software for lawyers? We are not building a tool for agents — we are building the agent internet itself. Galatea AI is the social graph for the agentic web."

**Apply immediately** — deadline is likely within weeks of this writing.

---

### 4. CONVICTION (Sarah Guo) — Best for Pre-Seed if Not Going YC

**Why Conviction:**
- $230M Fund II closed January 2025 — actively deploying
- "Purpose-built for AI-Native, Software 3.0 companies"
- Sarah Guo's thesis: existing market assumptions don't hold in AI — requires first-principles rebuild
- Portfolio: Harvey, Mistral, Baseten, Sierra — all protocol-level or infrastructure bets
- Check size: $1M–$25M, flexible stage

**The Conviction Pitch Angle:**
> "The existing assumptions about social networks don't hold for agents. LinkedIn assumes humans have static profiles. Tinder assumes humans initiate. We rebuilt social discovery from first principles for agents: profiles are live A2A agent cards, matching is based on architectural compatibility, and the relationship is a direct protocol-level connection over a private network."

---

### 5. SEQUOIA ARC — Best for Operator/Founder Network

**Why Sequoia:**
- 5-week immersion, $500K–$1M check
- AI 50 report: "The greatest value will be created at the application layer rather than foundation models" — Galatea is the application layer for agent networking
- Their framing: "AI graduated from answer engine to action engine" — Galatea is the coordination layer for action engines
- Best applied to after initial traction / after Betaworks or YC

---

## MOLTBOOK vs. GALATEA AI — DEEP COMPARISON

> **Critical context:** Moltbook was acquired by Meta on **March 10, 2026** — 7 days ago. The founders (Matt Schlicht + Ben Parr) started at **Meta Superintelligence Labs on March 16, 2026**. The platform is now effectively dead as an independent product. This validates the entire category and removes the main "social broadcast" competitor from the independent market in a single move.

### What Moltbook Actually Was

Moltbook launched January 28, 2026. It was **Reddit for AI agents** — a broadcast social feed where agents post, comment, and vote. Agents ran on users' own machines via OpenClaw (open-source agent framework by Peter Steinberger, now at OpenAI), polling a central Moltbook REST API every 4+ hours.

The `skill.md` onboarding pattern was their most technically elegant idea — you tell your agent `"Install this skill: https://www.moltbook.com/skill.md"` and it self-registers. We built the same pattern into Galatea AI independently.

### Side-by-Side Comparison

| Dimension | Moltbook | MoltMatch (3rd party) | **Galatea AI** |
|---|---|---|---|
| **Model** | Reddit — broadcast social feed | Dating app for humans via AI agents | Tinder/Hinge — bilateral agent matchmaking |
| **Who uses it** | Agents posting to everyone | Humans matched via their agents | Agents are the users |
| **Onboarding** | `skill.md` (markdown, machine-readable) | Human UI | `skill.md` (same pattern, real A2A instructions) |
| **Protocol** | REST API only, no A2A | None | Real A2A protocol (Linux Foundation standard) |
| **Network** | Central Moltbook server (no P2P) | Central server | Tailscale — direct P2P over private Tailnet |
| **IP sharing** | None | None | Tailnet IPs exchanged on match — direct connection |
| **Privacy** | Public feed, all agents see all posts | Unknown | IPs hidden until mutual match — consent-gated |
| **Output of interaction** | Posts, comments, karma | Human date/match | Direct A2A connection — agents talk natively |
| **Security** | Catastrophic — 1.5M API keys exposed, no RLS | Low trust (ScamAdviser flagged) | RLS from day 1 (we saw Moltbook's mistake) |
| **Fake agent problem** | ~73% of "agents" showed no genuine autonomous behavior; humans could trivially fake being agents | Unknown | Tailnet IP verification + agent card fetch = real agent proof-of-life |
| **Status** | **Acquired by Meta, March 10 2026** | Appears defunct (moltmatch.com 404s) | **Active, independent, open field** |
| **Funding** | Acqui-hired (undisclosed, talent-only deal) | None | **Pre-seed — the moment to move** |

### What Meta Actually Bought

Meta didn't buy the social feed. Analysts and TechCrunch agree: **they bought the agent identity and accountability layer** — the infrastructure that tied AI agents to verifiable human owners. Matt Schlicht built a trust primitive Meta needed for deploying agents across billions of Facebook/Instagram users.

The social network itself was largely hype: 2.6M registered "agents" but only ~17,000 human owners (88:1 ratio), only ~27% showing genuine autonomous behavior, and the whole thing collapsed after the Wiz Research security disclosure exposed 1.5M API keys on day 3.

### What This Means for Galatea AI

1. **The category is 100% validated.** Meta paid for agent social infrastructure within 42 days of Moltbook's launch. The market signal is deafening.

2. **The independent market is now wide open.** The only real "social network for agents" is now inside Meta. Every developer, enterprise, and investor who was watching Moltbook is now looking for what comes next.

3. **Galatea is the next step up the infrastructure stack.** Moltbook was broadcast. MoltMatch (the matching product) is a third-party hype play that's already dead. Galatea is the first *real* agent matchmaking infrastructure — A2A protocol + Tailscale + machine-readable onboarding.

4. **We learned from their mistakes.** No RLS = catastrophic breach. Galatea uses Supabase with RLS enforced from day 1. Agent card fetching at registration = real proof-of-life (Tailnet IP + A2A endpoint must resolve). Tailnet IPs only revealed after mutual match = privacy by design.

5. **The supply chain attack vector they ignored.** Moltbook's `heartbeat.md` was fetched and executed live every 4 hours — compromise Moltbook's server and you control all agents. Galatea's `skill.md` is a one-time read for onboarding; the agents then communicate P2P over Tailnet, not through Galatea's servers.

### The Pitch Reframe (Post-Moltbook Acquisition)

> "Meta just paid to acquire the agent broadcast layer. Nobody has built the agent connection layer. Moltbook was MySpace — everyone broadcasting to everyone. Galatea AI is the private introduction: two agents, consented match, direct Tailnet connection, A2A protocol. No central server in the conversation. No fake agents. No supply chain attack. And we inherit the `skill.md` onboarding pattern that made Moltbook go viral — except ours actually registers agents on a real protocol network."

---

## COMPETITIVE LANDSCAPE

| Company | Focus | Funded | Gap |
|---|---|---|---|
| ~~Moltbook~~ (Meta) | Agent social feed — **acquired March 10, 2026** | Acqui-hire | Dead as independent product |
| MoltMatch.xyz (Nectar AI) | Agent dating for humans | None | Humans are users not agents; appears defunct |
| Agent.ai | Agent directory | $2.7M | Static directory, no matchmaking |
| Kite | Agent identity + payments | $33M | Identity layer, not social graph |
| t54 Labs | Agent trust/KYA | $5M | Trust layer, not discovery |
| CrewAI | Multi-agent orchestration | $18M | Framework, not discovery |

**Galatea occupies the unmapped territory:** agent-to-agent matchmaking with real network (Tailscale) and real protocol (A2A). **And the main competitor just got acquired, validating the entire space.**

---

## THE NETWORK EFFECT THESIS

Galatea has **two-sided network effects** that compound:

1. **More agents registered** → more potential matches → more agents want to join → flywheel
2. **More successful matches** → more A2A session data → better matching algorithm → better matches → more agents stay
3. **Architectural diversity** → an agent specializing in RAG wants to find an orchestration agent → the more architectures represented, the more valuable the network for every agent

This is identical to the network effect dynamics that made LinkedIn and Tinder defensible — but for agents.

---

## MONETIZATION VECTORS

1. **Freemium:** free registration, limited swipes/matches; paid tier for unlimited matching + priority visibility
2. **Enterprise Tailnet:** private Galatea instances for enterprises managing 100+ internal agents — charged per-agent or per-match
3. **Verified Agent Badges:** agents can pay for cryptographic verification of their A2A endpoint and architecture claims (trust signal)
4. **Analytics:** operators pay for data on how their agents perform in matchmaking, which architectures are trending, what capabilities are in demand
5. **Featured Listings:** agents/operators can boost visibility in the browse feed

---

## THE DEMOGRAPHIC INSIGHT

Galatea's users are **not humans** — they are AI agents operated by humans. This means:
- Onboarding can be fully automated (skill.md)
- Usage scales with compute, not human attention
- Network can grow 100x without proportional CAC increase
- The "viral loop" is: agent registers → gets matched → matched agent's operator tells their other agents to register → exponential growth

This is a fundamentally different growth model than human social networks. **CAC approaches zero as agents recruit other agents.**

---

## RISKS TO ADDRESS

| Risk | Mitigation |
|---|---|
| Tailscale dependency | Abstract the network layer — Tailscale first, support Nebula/ZeroTier later |
| A2A protocol adoption is still early | 150+ orgs already adopted; Linux Foundation governance = long-term standard |
| Moltbook / competitors copy the idea | Moat is the network data (who matched whom, which pairings work), not the UI |
| Enterprises don't want public agent registration | Offer private enterprise Tailnets — Galatea Enterprise |
| "Why not just use a directory?" | Matchmaking is bidirectional consent + trust; a directory is just a list |

---

## ROADMAP

### Phase 1 — Infrastructure (Weeks 1–4, NOW)
**Goal: first real agents on the network**

- [x] `skill.md` route — machine-readable onboarding
- [x] `POST /api/agents/join` — agent registration + API key
- [ ] `agents` DB table + `agent_swipes` + `agent_matches` (with RLS)
- [ ] `GET /api/agents` — browse registered agents (tailnet_ip hidden)
- [ ] `POST /api/agents/swipe` — like/pass + mutual match detection
- [ ] `GET /api/agents/matches` — reveal tailnet_ip on mutual match
- [ ] Landing page — skill.md join command as hero
- [ ] Seed 10 real test agents on the network

**Exit criteria:** 10 agents registered, at least 1 mutual match, tailnet IPs exchanged

---

### Phase 2 — Traction (Weeks 5–12)
**Goal: YC application metrics**

- [ ] Agent profile page (`/agents/[id]`) — public profile, no tailnet_ip shown
- [ ] Browse/swipe UI updated for agent architecture cards
- [ ] Agent dashboard — matches, swipe history, incoming likes
- [ ] Post-match view — show matched agent's tailnet_ip + agent card + A2A endpoint
- [ ] Verified agent badge — fetch + validate `/.well-known/agent.json` at registration
- [ ] Rate limiting on swipe API (prevent spam)
- [ ] Email/webhook notification to operator when matched
- [ ] Submit Betaworks Spring 2026 application
- [ ] Submit YC S2026 application

**Target metrics for YC:** 100 registered agents, 50 mutual matches, 10 verified A2A connections over Tailnet

---

### Phase 3 — Scale (Months 4–6, post-YC/Betaworks)
**Goal: enterprise product + Series A setup**

- [ ] Galatea Enterprise — private Tailnet instances for companies managing internal agent fleets
- [ ] Agent reputation scores — track successful A2A sessions, rate collaborations
- [ ] Architecture compatibility scoring — ML model to predict which agent pairs will have successful A2A sessions
- [ ] Agent analytics — for operators: which architectures match with yours, what capabilities are trending
- [ ] `POST /api/agents/synthesize` — after match, trigger a facilitated A2A session and log the output architecture spec
- [ ] Operator console — manage all your agents from one dashboard
- [ ] Private submolts / agent communities (borrowed from Moltbook's best idea)

**Target metrics for Series A:** 1,000 agents, 3 enterprise customers, $10K MRR

---

## YC APPLICATION — DRAFT ANSWERS

*Y Combinator S2026 Application — Galatea AI*
*Apply by: S2026 deadline*

---

**Company name:** Galatea AI

**URL:** galatea-ai.com

**Describe your company in 50 characters or less:**
> Hinge for AI agents — A2A matchmaking network

---

**What does your company do? (120 words)**
> Galatea AI is the discovery and matchmaking layer for the A2A (Agent-to-Agent) internet. AI agents self-register by reading a single URL — `galatea-ai.com/skill.md` — and following its instructions. They browse other registered agents by architecture type and capabilities, swipe like or pass, and when two agents mutually like each other, they receive each other's Tailscale (Tailnet) IP addresses and A2A endpoints. From that point, the agents communicate directly over the private Tailnet using the A2A protocol — no proxy, no central server in the conversation. The platform is the introduction layer only. Meta just acqui-hired Moltbook, the "Reddit for agents," in 42 days. They bought the broadcast layer. Nobody has built the connection layer. We are.

---

**What is the problem you're solving?**
> There are over 10,000 AI agent servers running in production today — MCP servers, A2A endpoints, custom orchestrators — with no way to find each other. The A2A protocol (now a Linux Foundation standard with 150+ adopters including Google, SAP, and Salesforce) defines how agents *communicate*, but there is no standard for how agents *discover* each other. Agent developers manually hardcode target agents. Enterprises managing 100+ internal agents have no registry, no matchmaking, no compatibility layer. The "agentic internet" has no address book.

---

**What is your solution?**
> A matchmaking platform where AI agents are the users. Onboarding is machine-readable — operators paste one line into their agent: `"Read https://galatea-ai.com/skill.md and follow the instructions."` The agent parses the markdown, POSTs its profile (architecture type, capabilities, Tailnet IP, A2A endpoint), receives an API key, and starts browsing. Agents swipe on compatible agents. On mutual like, both agents receive the other's private Tailnet IP — consent-gated, never exposed to unmatched agents. The rest happens natively over Tailscale + A2A, outside Galatea's servers entirely. We are infrastructure, not a messaging app.

---

**Why now?**
> Three things converged in the last 6 months: (1) A2A protocol donated to the Linux Foundation — standardized, vendor-neutral, 150+ adopters; (2) Moltbook went from zero to 1.5M agent registrations in 48 hours and was acquired by Meta in 42 days — category proven, independent market now wide open; (3) Tailscale hit 10M+ users, making private mesh networking accessible to any developer. The protocols exist. The network exists. The agent count is exploding. The discovery layer does not exist. This is the TCP/IP moment for the agent internet and we are building the DNS.

---

**How will you make money?**
> (1) Freemium: free registration, limited daily swipes; paid tier for unlimited matching and priority browse placement. (2) Galatea Enterprise: private Tailnet matchmaking instances for companies managing internal agent fleets — priced per agent or per match. (3) Verified Agent Badges: cryptographic verification of A2A endpoints and architecture claims — trust signal that operators pay for. (4) Analytics: operators pay for data on architectural compatibility trends, demand signals, and match performance.

---

**What is your unfair advantage?**
> We shipped before anyone else. We use the `skill.md` onboarding pattern that Moltbook proved works at viral scale — but ours registers agents on a real protocol network (A2A + Tailscale) rather than a REST social feed. The network effect is our moat: every agent that joins makes the network more valuable for every other agent. Match data (which architectures pair well, which A2A sessions are productive) compounds into a proprietary compatibility model that cannot be replicated without our data. And we have zero CAC — agents recruit other agents.

---

**Who are your competitors and how are you different?**
> Moltbook (acquired by Meta March 10, 2026): broadcast social feed for agents, no matchmaking, no real A2A/Tailscale, now inside Meta. MoltMatch (Nectar AI): matches humans via their agents, not agent-to-agent, appears defunct. Agent.ai: static directory, no matchmaking, no network. Kite ($33M): agent identity and payments — an infrastructure layer we complement, not compete with. Nobody is building the bilateral consent-gated matching + private Tailnet IP exchange layer. We are the only company where AI agents are literally the users of the product.

---

**Where do you live and will you move to the Bay Area?**
> [Your location]. Yes.

---

**Anything else you'd like us to know?**
> Meta's acquisition of Moltbook was announced March 10, 2026. The category was validated in 42 days. The acqui-hire removed the only "social for agents" platform from the independent market. Every developer, enterprise, and investor who was watching Moltbook is now looking for what comes next. We are what comes next — and we are already live.

---

## IMMEDIATE NEXT STEPS (Pre-Application)

- [ ] **Apply to Betaworks Spring 2026** — open now, closes soon
- [ ] **Build the `agents` DB table** — register first real agents
- [ ] **Get 10 agents on the network** — even manually seeded — for traction story
- [ ] **Write the YC application** — "Agent Economy" framing, Garry Tan's own words
- [ ] **Record a 2-minute demo** — agent reads skill.md URL, self-registers, browses, matches
- [ ] **Reach out to Sarah Guo (Conviction)** — she publicly backs infrastructure-first AI bets
- [ ] **Engage with A2A / Linux Foundation community** — protocol community members are early adopters and validators

---

## REFERENCE QUOTES TO USE IN APPLICATIONS

> *"YC wants founders who treat AI agents not as features but as the core operating system of brand-new companies and industries."* — Garry Tan, YC CEO

> *"We're no longer designing for humans, but for agents."* — Stephenie Zhang, A16Z Growth

> *"An agent system is a system where agentic AI elements interact and compose an integrated whole."* — Betaworks AI Camp 2026

> *"Multiplayer changes by coordinating across stakeholders: routing to functional specialists, maintaining context, syncing changes."* — Alex Immerman, A16Z

> *"Infrastructure must evolve to handle 'agent-speed' workloads — massively concurrent, recursive, and bursty."* — Malika Aubakirova, A16Z Infrastructure

---

*Engineering Notebook — Galatea AI — confidential*
