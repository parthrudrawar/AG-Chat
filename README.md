# AG-Chat

A Streamlit-based chat app that lets you query your Confluence documentation using natural language. It uses a LangChain agent backed by Mixtral-8x7B (via Hugging Face) to find relevant Confluence pages, extract their content, and return structured markdown answers.

---

## What it does

You type a question like *"What is the difference between Datapod and Datasource?"* and the app:

1. Extracts entities from the question using the LLM (`Datapod`, `Datasource`)
2. Fetches all page titles from the Confluence space (`IID`)
3. Fuzzy-matches those entities to real page titles using `difflib`
4. Pulls the content of matched pages + their parent pages via the Confluence API
5. Feeds the content to Mixtral-8x7B to generate a structured markdown answer
6. Falls back to a generic rule-based answer if the LLM or Confluence is unavailable

Answers are formatted based on question intent — definitions for "what is", comparison tables for "difference between", step guides for "how to".

---

## How the agent works
```
User Question
     ↓
Extract Entities (Mixtral via HF Inference API)
     ↓
Fuzzy Match to Confluence Page Titles (difflib)
     ↓
Fetch Page + Parent Page Content (Confluence REST API)
     ↓
Feed Content to LLM → Structured Markdown Answer
     ↓
Streamlit UI (chat history, matched pages, answer)
```

The LangChain agent has two tools:
- **Confluence Search** — fetches and returns page content by matched title
- **Generic Answer** — rule-based fallback when no content is available

---

## Project Structure
```
AG-Chat/
├── main.py          # All logic — agent, tools, Confluence client, Streamlit UI
├── requirement.txt  # Dependencies
├── .env             # API keys (not committed)
└── README.md
```

---

## Environment Variables

Create a `.env` file with the following:
```env
HF_TOKEN=your_huggingface_token
CONFLUENCE_API_TOKEN=your_confluence_api_token
CONFLUENCE_USERNAME=your_confluence_email
CONFLUENCE_URL=https://your-domain.atlassian.net
```

> The HF token needs **Write** access for the Inference API to work.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Streamlit** | UI |
| **LangChain** | Agent orchestration + memory |
| **Mixtral-8x7B** (HF Inference API) | Entity extraction + answer generation |
| **Atlassian Python API** | Confluence page fetching |
| **BeautifulSoup** | HTML → plain text parsing |
| **difflib** | Fuzzy title matching |
| **python-dotenv** | Env config |

---

## Setup
```bash
git clone https://github.com/parthrudrawar/AG-Chat.git
cd AG-Chat

pip install -r requirement.txt

# Add your credentials
cp .env .env.local  # fill in the values

streamlit run main.py
```

---

## Example Questions
```
What is Dashboard?
What is the difference between Datapod and Datasource?
What are Business Rules?
How to choose between Datapod and Datasource?
```

---

## Notes

- Confluence space is hardcoded to `IID` — change `SPACE_KEY` in `main.py` to use a different space
- Page mappings are cached locally in `inferyx_doc_links.json` after first fetch to avoid repeated API calls
- If the HF token has billing issues (402) or permission issues (403), the app falls back to the generic answer generator

---

## Author

[parthrudrawar](https://github.com/parthrudrawar)
