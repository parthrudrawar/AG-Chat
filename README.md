# AG-Chat

> Documentation sits across dozens of Confluence pages. Finding a specific answer means clicking through spaces, reading full pages, and still not being sure you found the right one. AG-Chat lets you ask a plain English question and get a structured answer pulled directly from your Confluence space — in seconds.

---

## The Problem it Solves

Teams using Confluence at scale end up with hundreds of pages that are hard to navigate. This app connects directly to a Confluence space, extracts page content, matches it to your question using fuzzy title matching, and feeds the relevant content to Mixtral-8x7B to generate a clean, structured markdown answer — with source links included.

---

## How it Works

```
User Question
        ↓
Extract entities from question
(Mixtral via HF Inference API)
        ↓
Fetch all page titles from Confluence space (IID)
        ↓
Fuzzy-match entities → real page titles
(difflib SequenceMatcher, threshold = 0.7)
        ↓
Fetch matched page content + parent page content
(Confluence REST API)
        ↓
Build prompt: content + last 2 conversation turns + question
        ↓
Mixtral-8x7B generates structured markdown answer
        ↓
Streamlit chat UI — answer + matched pages displayed
```

---

## Answer Format — Adapts to Question Intent

The LLM prompt instructs the model to shape its response based on what the question is asking:

| Question Type | Response Format |
|--------------|----------------|
| "What is X?" | Definition + 2-3 detail bullet points |
| "Difference between X and Y?" | Definitions + comparison table (if content exists for both) |
| "How to use / choose X?" | Definitions + 3-5 step guide |
| No content found | Rule-based generic answer with Inferyx context fallback |

---

## LangChain Agent Tools

The app uses a `CONVERSATIONAL_REACT_DESCRIPTION` agent with two tools:

| Tool | When Used |
|------|----------|
| **Confluence Search** | Extracts entities from query → fuzzy matches to page titles → fetches page + parent content |
| **Generic Answer** | Fallback when no Confluence content is found or API fails |

---

## Technical Decisions Worth Noting

**Fuzzy title matching over exact search** — Confluence's search API requires exact or near-exact terms. Users rarely type page titles exactly. `difflib.SequenceMatcher` with a 0.7 threshold handles variations like abbreviations and partial names without any external search dependency.

**Parent page content fetched alongside matched page** — Confluence pages are hierarchical. A child page often lacks full context without its parent. Fetching both and including them in the prompt significantly improves answer quality for nested documentation.

**Conversation memory limited to last 2 turns** — Full history in every prompt inflates token count quickly. Two turns preserves conversational context without hitting Mixtral's context window limits or increasing latency unnecessarily.

**Page mappings cached to disk** — On first load, all Confluence page titles and IDs are fetched and saved to `inferyx_doc_links.json`. Subsequent runs load from disk — avoiding repeated API calls across sessions and keeping startup fast.

**Entity extraction with regex fallback** — If Mixtral fails to return valid JSON entities, the app falls back to regex-based capitalized noun extraction. This keeps the pipeline functional even when the HF Inference API is rate-limited or down.

**Generic answer with domain context** — When no Confluence content is found, the app doesn't return a blank error. It serves a pre-built context map for known Inferyx terms (Datapod, Dashboard, Datasource, Business Rules, DQ Rules) so the user still gets a useful response.

---

## Project Structure

```
AG-Chat/
├── main.py          # All logic — Confluence client, entity extraction,
│                    # fuzzy matching, LangChain agent, Streamlit UI
├── requirement.txt
├── .env
└── README.md
```

---

## Environment Variables

```env
HF_TOKEN=your_huggingface_token
CONFLUENCE_API_TOKEN=your_confluence_api_token
CONFLUENCE_USERNAME=your_confluence_email
CONFLUENCE_URL=https://your-domain.atlassian.net
```

> HF token requires **Write** access for the Inference API.
> Confluence space is hardcoded to `IID` — change `SPACE_KEY` in `main.py` to use a different space.

---

## Setup

```bash
git clone https://github.com/parthrudrawar/AG-Chat.git
cd AG-Chat

pip install -r requirement.txt

cp .env .env.local  # fill in credentials

streamlit run main.py
```

---

## Example Questions

```
What is Dashboard?
What is the difference between Datapod and Datasource?
What are Business Rules?
How to choose between Datapod and Datasource?
What are DQ Rules?
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | Chat UI |
| **LangChain** | Agent orchestration + conversation memory |
| **Mixtral-8x7B** (HF Inference API) | Entity extraction + answer generation |
| **Atlassian Python API** | Confluence page + parent page fetching |
| **BeautifulSoup** | HTML → plain text parsing |
| **difflib** | Fuzzy title matching |
| **python-dotenv** | Env config |

---

## Notes

- If HF token has billing issues (402) or permission issues (403), the app automatically falls back to the generic answer generator — no crash, no blank screen
- Page mappings cached in `inferyx_doc_links.json` after first fetch — delete this file to force a fresh pull from Confluence
- Both matched page and its parent page are included in the LLM context for better answer accuracy on nested documentation

---

## Author

[parthrudrawar](https://github.com/parthrudrawar)
