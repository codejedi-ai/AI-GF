# Aesopic Exercise: Autonomous Web Navigation Agent

## Deliverable: Observations Document

**Author:** Darcy Liu  
**Date:** February 3, 2026

---

## 1. Executive Summary

I have successfully built a **Self-Supervised, Cost-Aware Navigation Agent** that autonomously navigates GitHub and extracts release information using a multi-model architecture. The system completed the task of finding the latest release for `openclaw/openclaw` in a single navigation step with a total cost of ~12 units.

**Sample Output:**
```json
{
  "repository": "openclaw/openclaw",
  "latest_release": {
    "version": "v2026.2.1",
    "commit": "d84b228",
    "author": "steipete"
  }
}
```

---

## 2. Architectural Approach: The "Split-Brain" Model

Rather than using a single monolithic LLM for all tasks, I designed a **biologically-inspired architecture** that separates perception from reasoning:

| Component | Model | Role |
|-----------|-------|------|
| **Eyes** | GPT-4o (Vision) | Visual perception and page description |
| **Brain** | Claude 3.5 Sonnet | Strategic reasoning, goal-checking, heuristic generation |
| **Spinal Cord** | MCP Server + Playwright | Low-level browser actions, DOM queries, semantic element filtering |

### Why This Design?

1. **GPT-4o excels at visual understanding** – It accurately identifies UI elements, badges ("Latest"), and page structure from screenshots.
2. **Claude excels at reasoning under constraints** – It generates effective heuristics and makes strategic navigation decisions.
3. **MCP (Model Context Protocol) provides fast, reliable browser control** – It handles the "muscle memory" of clicking, typing, and querying the DOM without expensive LLM calls.

---

## 3. Key Innovations

### Innovation I: Prediction Model (Upfront Heuristic Generation)

**Problem:** Traditional agents call the LLM for every link evaluation, leading to O(n) API costs per page.

**Solution:** At the start of navigation, the Brain generates a **Weighted Bag-of-Words** model:

```json
{
  "weights": {
    "Releases": 1.0,
    "Latest": 0.9,
    "v[0-9]": 0.7,
    "Tags": 0.5
  },
  "decay_rate": 0.6
}
```

This allows the agent to score all links **locally** using fast string matching, reducing API costs to O(1) per page.

### Innovation II: Dual A* Search Strategy

I implemented two complementary search algorithms:

| Algorithm | Role | Cost |
|-----------|------|------|
| **Lesser A*** (DOM Searcher) | Fast candidate retrieval using CSS selectors | ~$0.01 |
| **Greater A*** (Page Searcher) | Semantic scoring using `Score = WordValue × Count × FontCoefficient` | ~$0.05 |

The **Font Coefficient** assigns higher value to prominent elements (e.g., `<h1>` = 3.0, `<button>` = 1.2), mimicking how humans prioritize visually prominent elements.

### Innovation III: Self-Supervised Q-Learning

The agent maintains a **Knowledge Base** (Q-Table) that persists across runs:

- **State:** URL hash
- **Action:** `click:{element_text}`
- **Reward:** +1.0 when goal is reached

On subsequent runs, the agent immediately prioritizes known-successful paths, bypassing exploration entirely.

---

## 4. What Worked Well

1. **Single-Step Navigation:** The agent reached the latest release page in just **1 navigation step** (directly from `/releases` to the release tag page).
2. **Accurate Extraction:** The Vision model correctly identified the version, commit hash, and author.
3. **Cost Efficiency:** Total cost of ~12 units vs. estimated 50+ units for a naive Chain-of-Thought approach.
4. **Generalization:** The same code works for any GitHub repository via the `--repo` argument.

---

## 5. Challenges & Trade-offs

### Challenge 1: DOM Volatility
GitHub's DOM structure is complex and includes many dynamic elements. I addressed this by:
- Using BeautifulSoup for cleaner DOM parsing
- Filtering to semantic elements only (`a`, `button`, `summary`)
- Avoiding hardcoded selectors

### Challenge 2: Vision Model Latency
Visual analysis adds ~2-3 seconds per step. I mitigated this by:
- Using Eyes only for initial page understanding and goal verification
- Relying on local heuristics for link scoring

### Trade-off: Accuracy vs. Cost
The Weighted Bag-of-Words model is a **lossy approximation** of semantic understanding. It works well for structured sites like GitHub but may need tuning for other domains.

---

## 6. Limitations & Future Improvements

| Limitation | Potential Fix |
|------------|---------------|
| Q-Table uses only URL for state | Use DOM structure hash for finer-grained states |
| No negative reward backpropagation | Implement penalty for dead-ends |
| Font Coefficient is tag-based only | Integrate actual computed CSS font-size |
| Headless mode not tested | Add `--headless` flag for CI/CD |

---

## 7. Honest Reflection

1. **The Split-Brain architecture is overkill for this task** – A single GPT-4o call could likely handle the entire flow. However, the architecture is designed for **scalability and cost-efficiency** in production scenarios with hundreds of navigation tasks.

2. **Q-Learning is slow to converge** – With only positive rewards, the agent needs many successful runs to build a reliable policy. A more sophisticated approach (e.g., curiosity-driven exploration) would help.

3. **I relied heavily on AI coding assistants** – Claude (via cursor) wrote ~70% of the boilerplate code, allowing me to focus on architecture and heuristic design.

---

## 8. Running the Solution

```bash
# Setup
pip install -r requirements.txt
playwright install chromium

# Run
python navigate.py --repo "openclaw/openclaw"

# Output
# → data/output.json (extracted data)
# → data/cost_log.json (cost metrics)
```

---

## 9. Conclusion

This project demonstrates that **intelligent navigation does not require brute-force LLM calls**. By combining:
- **GPT-4o** for visual perception (Eyes)
- **Claude** for strategic reasoning (Brain)
- **MCP + Playwright** for fast browser control (Spinal Cord)
- **Local heuristics** for cost-efficient scoring

...we can build agents that are both **smart and economical**.

The Self-Supervised Q-Learning layer ensures the agent **gets better over time**, learning from its own experience without human intervention.

---

With gratitude,

**Darcy Liu**