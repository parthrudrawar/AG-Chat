import streamlit as st
from atlassian import Confluence
from dotenv import load_dotenv
import os
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from huggingface_hub import InferenceClient
from bs4 import BeautifulSoup
import json
import time
import difflib
import re

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")

if not HF_TOKEN or not CONFLUENCE_API_TOKEN or not CONFLUENCE_USERNAME or not CONFLUENCE_URL:
    st.error("Missing required environment variables. Please check configuration in Hugging Face Spaces secrets.")
    st.stop()

SPACE_KEY = "IID"
LLM_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

def get_confluence_client():
    """Initialize Confluence client."""
    try:
        confluence = Confluence(
            url=CONFLUENCE_URL,
            username=CONFLUENCE_USERNAME,
            password=CONFLUENCE_API_TOKEN,
            timeout=10
        )
        confluence.get_space(SPACE_KEY)
        return confluence
    except Exception as e:
        st.error(f"Error initializing Confluence client: {e}")
        return None

def get_all_pages():
    """Fetch all page titles and IDs from the IID space."""
    confluence = get_confluence_client()
    if not confluence:
        return {}
    
    dict_page_id = {}
    start = 0
    limit = 100
    
    while True:
        try:
            pages = confluence.get_all_pages_from_space(
                space=SPACE_KEY,
                start=start,
                limit=limit,
                status='current'
            )
            if not pages:
                break
            for page in pages:
                dict_page_id[page['title']] = page['id']
            start += limit
            time.sleep(1)
        except Exception as e:
            st.error(f"Error fetching pages: {e}")
            break
    return dict_page_id

def find_best_matching_titles(entities, titles, question, threshold=0.7):
    """Find the most similar titles for a list of entities using difflib."""
    matched_titles = []
    
    for entity in entities:
        if not entity:
            entity = question
        
        entity = re.sub(r'[\*\d\.\s]*(Answer|Response)?:?\s*|[^\w\s]', '', entity, flags=re.IGNORECASE).strip()
        if len(entity) > 50:
            entity = re.sub(r'[\*\d\.\s]*(Answer|Response)?:?\s*|[^\w\s]', '', question, flags=re.IGNORECASE).strip()
        
        best_match = None
        highest_score = None
        
        for title in sorted(titles, key=lambda x: len(str(x))):  # Sort to avoid TypeError
            score = float(difflib.SequenceMatcher(None, entity.lower(), str(title).lower()).ratio())
            if score > highest_score and score >= threshold:
                highest_score = score
                best_match = title
        
        if best_match:
            matched_titles.append(best_match)
    
    return matched_titles

def get_page_and_parent_content(page_id, title):
    """Fetch content of the page and its parent page, if it exists."""
    confluence = Confluence(
        url=CONFLUENCE_URL,
        username=CONFLUENCE_USERNAME,
        password=CONFLUENCE_API_TOKEN,
        timeout=10,
    )
    if not confluence:
        return None
    
    try:
        page = confluence.get_page_by_id(page_id, expand='body.storage,ancestors')
        page_content = page.get('body', {}).get('storage', {}).get('value', '')
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            page_content = soup.get_text(separator=' ', strip=True).strip()
        else:
            st.warning(f"No content found for '{title}' (ID: {page_id})")
            page_content = None
        
        parent_content = None
        parent_title = None
        ancestors = page.get('ancestors', [])
        if ancestors:
            parent_id = ancestors[-1]['id']
            try:
                parent_page = confluence.get_page_by_id(
                    page_id=parent_id,
                    expand='body.storage'
                )
                parent_content = parent_page.get('body', {}).get('storage', {}).get('value', '')
                parent_title = parent_page.get('title', 'Unknown')
                if parent_content:
                    soup = BeautifulSoup(parent_content, 'html.parser')
                    parent_content = soup.get_text(separator=' ', strip=True).strip()
                else:
                    st.warning(f"No content found for parent '{parent_title}' (ID: {parent_id})")
            except Exception as e:
                st.warning(f"Error fetching parent content for ID {parent_id}: {e}")
        else:
            st.info(f"No parent page found for '{title}' (ID: {page_id})")
        
        return page_content, parent_content, parent_title
    except Exception as e:
        st.error(f"Error fetching content for '{title}' (ID: {page_id}): {e}")
        return None, None, None

def extract_entities(question):
    """Extract multiple entities using LLM with fallback keyword extraction."""
    try:
        client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
        prompt = (
            "Extract all entities that could match Confluence page titles from the following question. "
            "Return a JSON list of entities, each a concise phrase (typically 1-3 words). "
            "Do not include prefixes like 'Answer: ' or additional text. "
            "Example: For 'What is the difference between Datapod and Datasource?', return ['Datapod', 'Datasource']. "
            "For 'What are Business Rules?', return ['Business Rules']. "
            f"Question: {question}"
        )
        output = client.text_generation(prompt, max_new_tokens=100)
        try:
            entities = json.loads(output.strip())
            if not isinstance(entities, list):
                entities = [entities]
        except json.JSONDecodeError:
            entities = [e.strip() for e in ''.join(output.strip().splitlines()) if e.strip()]
        
        # Clean entities
        cleaned_entities = []
        for entity in entities:
            entity = re.sub(r'[\*\d\s]*(?:Answer|Response)?:?\s*|[^\w\s]', '', entity, flags=re.IGNORECASE).strip()
            if entity and len(entity) <= 30:
                cleaned_entities.append(entity)
        
        if not cleaned_entities:
            raise ValueError("No valid entities extracted")
        
        return cleaned_entities
    except Exception as e:
        st.error(f"LLM failed for entity extraction: {e}. Ensure your HF_TOKEN has 'Write' access at https://huggingface.co/settings/tokens.")
        # Fallback to regex-based extraction
        pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, question)
        entities = [m for m in sorted(matches, key=len) if len(m) <= 30]
        return entities if entities else [question]

def generate_generic_answer(question, page_contents, matched_titles):
    """Generate a generic answer based on available content or question context."""
    markdown = ["## Answer"]
    
    # Inferyx-specific context for common terms
    inferyx_context = {
        "Datapod": "A Datapod in Inferyx is likely a data processing or storage component, but specific details are not available in the provided Confluence content.",
        "Dashboard": "A Dashboard in Inferyx typically displays visual analytics and KPIs, aggregating data from multiple sources for user insights.",
        "Datasource": "A Datasource in Inferyx is a source of raw data, such as databases, APIs, or files, used for data ingestion.",
        "Business Rules": "Business Rules in Inferyx define operational logic for data processing and workflows.",
        "DQ Rules": "DQ Rules in Inferyx focus on ensuring data quality, such as validating completeness and accuracy."
    }
    
    if not page_contents or not any(content[0] for content in page_contents.values()):
        markdown.append("### Definitions")
        for title in matched_titles or [question]:
            context = inferyx_context.get(title, f"No specific information found for '{title}' in the Confluence space 'IID'.")
            markdown.append(f"- **{title}**: {context}")
        markdown.append("\n### Details")
        markdown.append("- Limited information available. Please check the Confluence space 'IID' or consult Inferyx documentation.")
        return "\n".join(markdown)
    
    markdown.append("### Definitions")
    for title in matched_titles:
        content, _ = page_contents.get(title, (None, None))
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            headings = soup.find_all(['h1', 'h2', 'h3'])
            definition = headings[0].get_text().strip()[:200] + "." if headings else content.split('.')[0][:200] + "."
            markdown.append(f"- **{title}**: {definition}")
        else:
            context = inferyx_context.get(title, f"No content available in the Confluence page for '{title}'.")
            markdown.append(f"- **{title}**: {context}")
    
    markdown.append("\n### Details")
    for title in matched_titles:
        content, _ = page_contents.get(title, (None, None))
        markdown.append(f"**{title}**:")
        if content:
            soup = BeautifulSoup(content, 'html.parser')
            list_items = soup.find_all('li')[:3]
            if list_items:
                points = [li.get_text().strip()[:200] + "." for li in list_items][:2]
            else:
                sentences = content.split('.')[:3]
                points = [s.strip() + "." for s in sentences if s.strip()][:2]
            points = points or ["Limited details available from content."]
            for point in points:
                markdown.append(f"- {point}")
        else:
            markdown.append(f"- No details available. Refer to {infernix_context.get(title, 'Confluence documentation')}.")
    
    if len(matched_titles) > 1 and ("difference" in question.lower() or "compare" in question.lower()):
        markdown.append("\n### Comparison")
        if all(page_contents.get(title, (None, None))[0] for title in matched_titles):
            markdown.append("| Aspect | " + " | ".join(matched_titles) + " |")
            markdown.append("|--------|" + "|".join(["-" * len(title) for title in matched_titles]) + "|")
            markdown.append("| Purpose | " + " | ".join(
                [page_contents.get(title, (None, None))[0].split('.')[0][:50] + "..." if page_contents.get(title, (None, None))[0] else inferyx_context.get(title, "N/A")[:50] + "..." for title in matched_titles]
            ) + " |")
        else:
            markdown.append(
                "A direct comparison is not possible due to missing content for some entities. "
                f"Based on available information, {', '.join([t for t in matched_titles if page_contents.get(t, (None, None))[0] or t in inferyx_context])} "
                "are described above. Check Confluence for additional details."
            )
    
    if "how to" in question.lower() or "choose" in question.lower():
        markdown.append("\n### Stepwise Guide")
        if all(page_contents.get(title, (None, None))[0] for title in matched_titles):
            markdown.append("Based on available content:")
            markdown.append("1. **Identify Requirements**: Determine the specific needs (e.g., visualization for Dashboard, data ingestion for Datasource).")
            markdown.append("2. **Review Documentation**: Check Confluence pages for detailed configurations.")
            markdown.append("3. **Test Components**: Validate functionality in Inferyx using available content.")
        else:
            markdown.append("Due to limited or missing content, a general guide is provided:")
            markdown.append("1. **Verify Documentation**: Check the Inferyx Confluence space 'IID' for relevant pages.")
            markdown.append("2. **Consult Experts**: Contact the Inferyx team for clarification on terms.")
            markdown.append("3. **Explore Context**: Use available content to infer usage and purpose.")
    
    return "\n".join(markdown)

def feed_content_to_llm(page_contents, question):
    """Generate answer using LLM with content from multiple pages in a dynamic format."""
    if not any(page_contents.values()):
        return generate_generic_answer(question, page_contents, page_contents.keys())
    
    try:
        client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
        content_snippets = []
        for title, (content, parent_content) in page_contents.items():
            page_snippet = f"Content for '{title}':\n{content[:1500] if content else 'None'}"
            parent_snippet = f"Parent Content for '{title}':\n{parent_content[:500] if parent_content else 'None'}"
            content_snippets.append(f"{page_snippet}\n{parent_snippet}")
        
        combined_content = "\n\n".join(content_snippets)
        prompt = (
            "Based on the following Confluence page contents, answer the question in a structured markdown format. "
            "Analyze the question to determine its intent (e.g., comparison, process, definition) and tailor the response:\n"
            "- **Definitions**: Always provide a one-sentence definition for each entity (if content is available; otherwise, note 'No content available').\n"
            "- **Details**: Provide 2-3 bullet points per entity with key information (if content is available).\n"
            "- **Comparison Table**: Include a table comparing entities on 2-3 key aspects (e.g., purpose, usage) ONLY if the question asks for a comparison (e.g., 'difference', 'compare') AND content is available for ALL entities. Otherwise, provide a narrative comparison.\n"
            "- **Stepwise Guide**: Include a 3-5 step guide ONLY if the question implies a process (e.g., 'how to use', 'how to choose'). Otherwise, omit it.\n"
            "Use markdown for formatting. Keep each section concise (3-5 sentences or points). If content is missing for an entity, clearly state it and base the answer on available content or note limitations.\n\n"
            f"{combined_content}\n\n"
            f"Question: {question}"
        )
        output = client.text_generation(prompt, max_new_tokens=1500)
        answer = output.strip()
        return answer
    except Exception as e:
        st.error(f"Error in LLM processing: {e}. Ensure your HF_TOKEN has 'Write' access at https://huggingface.co/settings/tokens or check your Inference API credits at https://huggingface.co/settings/billing.")
        return generate_generic_answer(question, page_contents, page_contents.keys())

# LangChain agent setup
memory = ConversationBufferMemory(memory_key="chat_history")
llm = ChatOpenAI(model=LLM_MODEL, api_key=HF_TOKEN, base_url="https://api-inference.huggingface.co")

# Tools for LangChain agent
def confluence_search_tool(input_str):
    """Fetch Confluence page content by title."""
    dict_page_id = get_all_pages()
    titles = list(dict_page_id.keys())
    entities = extract_entities(input_str)
    matched_titles = find_best_matching_titles(entities, titles, input_str)
    if not matched_titles:
        return "No matching pages found."
    page_contents = {}
    for title in matched_titles:
        page_id = dict_page_id[title]
        page_content, parent_content, _ = get_page_and_parent_content(page_id, title)
        page_contents[title] = (page_content, parent_content)
    return json.dumps(page_contents)

tools = [
    Tool(
        name="Confluence Search",
        func=confluence_search_tool,
        description="Fetches content from Confluence pages in the IID space by matching titles to entities extracted from the query."
    ),
    Tool(
        name="Generic Answer",
        func=lambda x: generate_generic_answer(x, {}, [x]),
        description="Generates a generic answer when no Confluence content is available."
    )
]

#  LangChain agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=False
)

# Prompt template for structured answers
prompt_template = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""
    Answer the question in markdown format using the Confluence Search tool to fetch relevant page content.
    Follow the original format:
    - **Definitions**: One-sentence definition per entity (use content if available, else note 'No content').
    - **Details**: 2-3 bullet points per entity.
    - **Comparison**: Table for 'difference' or 'compare' questions if content exists, else narrative.
    - **Steps**: 3-5 step guide for 'how to' or 'choose' questions.
    Keep answers concise. If no content, use the Generic Answer tool.
    Question: {question}
    Chat History: {chat_history}
    """
)

def process_user_query(question, dict_page_id):
    """Process a user question using the LangChain agent."""
    try:
        # Run agent with question and history
        response = agent.run(prompt_template.format(question=question, chat_history=memory.buffer))
        # Extract matched titles from Confluence Search tool output
        try:
            page_contents = json.loads(confluence_search_tool(question))
            matched_titles = list(page_contents.keys())
        except:
            matched_titles = []
        return response, page_contents, matched_titles
    except Exception as e:
        st.error(f"Agent error: {e}. Falling back to generic answer.")
        return generate_generic_answer(question, {}, [question]), {}, [question]

def main():
    st.title("Inferyx Confluence Query App")
    st.write("Ask a question about Inferyx components (e.g., 'What is Dashboard?', 'What is the difference between Datapod and Datasource?').")
    st.info("If you see a 403 or 402 error, your Hugging Face token may lack permissions or credits. Check your token at https://huggingface.co/settings/tokens and billing at https://huggingface.co/settings/billing.")
    
    # Load or fetch page mappings
    @st.cache_data
    def load_page_mappings():
        try:
            with open('inferyx_doc_links.json', 'r') as f:
                data = json.load(f)
                return data.get('page_id_mappings', {})
        except FileNotFoundError:
            dict_page_id = get_all_pages()
            if dict_page_id:
                with open('inferyx_doc_links.json', 'w') as f:
                    json.dump({'page_id_mappings': dict_page_id}, f, indent=2)
            return dict_page_id
    
    dict_page_id = load_page_mappings()
    
    if not dict_page_id:
        st.error("Failed to retrieve page mappings from Confluence.")
        st.stop()
    
    with st.expander("View Confluence Page Mappings"):
        st.write("Sample Page ID Mappings:")
        for title, pid in list(dict_page_id.items())[:10]:
            st.write(f"Title: {title}, Page ID: {pid}")
        if len(dict_page_id) > 10:
            st.write(f"... and {len(dict_page_id) - 10} more")
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("Enter your question:", placeholder="e.g., What is Dashboard?")
    if st.button("Submit"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing your question..."):
                answer, page_contents, matched_titles = process_user_query(question, dict_page_id)
                if answer:
                    st.subheader("Matched Pages")
                    st.write(", ".join(matched_titles) if matched_titles else "None (generic answer)")
                    st.subheader("Answer")
                    st.markdown(answer, unsafe_allow_html=True)
                    st.session_state.chat_history.append(f"Q: {question}\nA: {answer}")
                else:
                    st.error("Unable to generate an answer.")
    
    # Display chat history
    with st.expander("Chat History"):
        for entry in st.session_state.chat_history:
            st.write(entry)

if __name__ == "__main__":
    main()
