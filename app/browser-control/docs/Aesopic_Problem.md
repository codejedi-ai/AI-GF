Here is the complete content of the **Autonomous Web Navigation with Vision Models** exercise (Aesopic Exercise) transcribed directly from the PDF provided.

---

# Autonomous Web Navigation with Vision Models

## Overview

* **Time Expectation:** 4-6 hours
* **Tools:** Use any programming language, frameworks, libraries, and AI coding assistants

## The Task

Build a tool that uses vision models to autonomously navigate GitHub and extract release information. Your solution should not rely on hardcoded selectors - it must work even if GitHub's HTML structure changes.

### Navigation Flow

1. Start at **github.com**
2. Search for **"openclaw"**
3. Click on the correct repository (**[github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)**)
4. Find and click the **"Releases"** section
5. Extract the latest release information

### Expected Interface

Your tool should be runnable from the command line. Here's an example (you can use any language):

```bash
python navigate.py --url "https://github.com" --prompt "search for openclaw and get the current release and related tags"

```

Or with a simpler interface:

```bash
node navigate.js --repo "openclaw/openclaw"

```

The specific interface is up to you — choose what makes sense for your implementation.

### Required Output Example

```json
{
  "repository": "openclaw/openclaw",
  "latest_release": {
    "version": "v2026.1.29",
    "tag": "77e703c",
    "author": "steipete"
  }
}

```

## Requirements

### Must Have:

* Use vision models (GPT-4 Vision, Claude, etc.) to understand page content and make navigation decisions.
* Navigate autonomously - no hardcoded CSS selectors or XPath.
* Handle the full flow from homepage to release extraction.
* Output structured JSON data.
* Include clear setup and run instructions.

### Constraints:

* Assume no authentication required (GitHub public pages).
* You may use any browser automation framework (Playwright, Puppeteer, Selenium, etc.).
* You may use AI coding assistants (Claude, Cursor, GitHub Copilot, etc.).

### Bonus Challenges (Optional):

* Extract additional metadata: release notes, download links, publish dates.
* Make it work for any GitHub repository via command-line argument.
* Accept natural language prompts for flexible queries (e.g., "Find the latest React release and list its key features").

## Deliverables

1. **Working Code** – Source code, README.md with setup/run instructions, dependencies file.
2. **Observations Document (1-2 pages)** – Your approach, what worked/didn't work, trade-offs, limitations, and future improvements.
3. **Sample Output** – JSON file with extracted openclaw data.

## Evaluation Criteria

We will evaluate your submission on:

* **Code quality** – Clear structure, appropriate tool choices, good documentation.
* **Design decisions** – Architecture, vision model integration, error handling.
* **Edge case handling** – Failures, timeouts, unexpected states.
* **Testing approach** – Evidence of validation and scenario testing.
* **Critical thinking** – Quality of observations, understanding of trade-offs, honest reflection.

**Important:** We value thoughtful analysis and honest reflection over perfect solutions. If something doesn't work, explaining why is just as valuable as making it work.

## Submission

* **Format:** GitHub repository (public, or private with access granted) OR ZIP file.
* **Include:** Source code, README.md, observations document, sample_output.json.
* **Deadline:** 02/02/2025

**Questions?** Email sai@aesopic.ai with clarifying questions. Make reasonable assumptions and document them.

---

With gratitude,

Darcy Liu