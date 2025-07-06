from atlassian import Confluence
from dotenv import load_dotenv
import os
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
    raise ValueError("Missing HF_TOKEN, CONFLUENCE_API_TOKEN, CONFLUENCE_USERNAME, or CONFLUENCE_URL in .env")

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
        print(f"❌ Error initializing Confluence client: {e}")
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
            print(f"❌ Error fetching pages: {e}")
            break
    return dict_page_id

def find_best_matching_titles(entities, titles, question, threshold=0.7):
    """Find the most similar titles for a list of entities using difflib."""
    matched_titles = []
    
    for entity in entities:
        if not entity:
            entity = question
        
        # Clean entity: remove markdown, numbers, and extra text
        entity = re.sub(r'[\*\d\.\s]*(Answer|Response)?:?\s*|[^\w\s]', '', entity, flags=re.IGNORECASE).strip()
        if len(entity) > 50:
            entity = re.sub(r'[\*\d\.\s]*(Answer|Response)?:?\s*|[^\w\s]', '', question, flags=re.IGNORECASE).strip()
        
        best_match = None
        highest_score = 0
        
        for title in titles:
            score = difflib.SequenceMatcher(None, entity.lower(), title.lower()).ratio()
            if score > highest_score and score >= threshold:
                highest_score = score
                best_match = title
        
        if best_match:
            matched_titles.append(best_match)
    
    return matched_titles

def get_page_and_parent_content(page_id, title):
    """Fetch content of the page and its parent page, if it exists."""
    confluence = get_confluence_client()
    if not confluence:
        return None, None, None
    
    try:
        page = confluence.get_page_by_id(
            page_id=page_id,
            expand='body.storage,ancestors'
        )
        page_content = page.get('body', {}).get('storage', {}).get('value', '')
        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            page_content = soup.get_text(separator=' ', strip=True).strip()
        else:
            print(f"❌ Error: No content found for '{title}' (ID: {page_id})")
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
                    print(f"❌ Error: No content found for parent '{parent_title}' (ID: {parent_id})")
            except Exception as e:
                print(f"❌ Error fetching parent content for ID {parent_id}: {e}")
        else:
            print(f"No parent page found for '{title}' (ID: {page_id})")
        
        return page_content, parent_content, parent_title
    except Exception as e:
        print(f"❌ Error fetching content for '{title}' (ID: {page_id}): {e}")
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
        # Try to parse as JSON
        try:
            entities = json.loads(output.strip())
            if not isinstance(entities, list):
                entities = [entities]
        except json.JSONDecodeError:
            # Fallback to splitting output
            entities = [e.strip() for e in output.strip().split(',') if e.strip()]
        
        # Clean entities
        cleaned_entities = []
        for entity in entities:
            entity = re.sub(r'^(Answer|Response)?:?\s*|[^\w\s]', '', entity, flags=re.IGNORECASE).strip()
            if entity and len(entity) <= 30:
                cleaned_entities.append(entity)
        
        if not cleaned_entities:
            raise ValueError("No valid entities extracted")
        
        return cleaned_entities
    except Exception as e:
        # Regex fallback: extract capitalized phrases
        pattern = r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b'
        matches = re.findall(pattern, question)
        entities = [m for m in matches if len(m) <= 30]
        return entities if entities else [question]

def feed_content_to_llm(page_contents, question):
    """Generate answer using LLM with content from multiple pages."""
    if not any(page_contents.values()):
        return None
    
    try:
        client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
        # Combine page contents
        content_snippets = []
        for title, (content, parent_content) in page_contents.items():
            page_snippet = f"Content for '{title}':\n{content[:1500] if content else 'None'}"
            parent_snippet = f"Parent Content for '{title}':\n{parent_content[:500] if parent_content else 'None'}"
            content_snippets.append(f"{page_snippet}\n{parent_snippet}")
        
        combined_content = "\n\n".join(content_snippets)
        prompt = (
            "Based on the following Confluence page contents, answer the question in 3-5 sentences. "
            "If multiple entities are mentioned, compare or contrast them as appropriate.\n\n"
            f"{combined_content}\n\n"
            f"Question: {question}"
        )
        output = client.text_generation(prompt, max_new_tokens=1000)
        answer = output.strip()
        return answer
    except Exception as e:
        print(f"❌ Error in LLM processing: {e}")
        return "Unable to generate an answer due to an error. Please check the Confluence pages for details."

def process_user_query(question, dict_page_id):
    """Process a user question by extracting entities, fetching content, and generating answer."""
    entities = extract_entities(question)
    titles = list(dict_page_id.keys())
    
    # Find matching titles for entities
    matched_titles = find_best_matching_titles(entities, titles, question)
    
    if not matched_titles:
        print(f"❌ Error: No valid titles found for question '{question}'")
        return None, {}, None
    
    # Fetch content for all matched pages
    page_contents = {}
    for title in matched_titles:
        page_id = dict_page_id[title]
        page_content, parent_content, parent_title = get_page_and_parent_content(page_id, title)
        if page_content:
            page_contents[title] = (page_content, parent_content)
        else:
            print(f"❌ Warning: Could not retrieve content for '{title}'")
    
    if not page_contents:
        print(f"❌ Error: No content retrieved for any matched titles")
        return None, {}, None
    
    # Generate answer
    answer = feed_content_to_llm(page_contents, question)
    return answer, page_contents, matched_titles

def main():
    # Load existing mappings
    try:
        with open('inferyx_doc_links.json', 'r') as f:
            data = json.load(f)
            dict_page_id = data.get('page_id_mappings', {})
    except FileNotFoundError:
        dict_page_id = {}
    
    # Fetch new mappings if none loaded
    if not dict_page_id:
        dict_page_id = get_all_pages()
        if not dict_page_id:
            print("❌ Error: Failed to retrieve page mappings")
            return
        with open('inferyx_doc_links.json', 'w') as f:
            json.dump({'page_id_mappings': dict_page_id}, f, indent=2)
    
    print("\nPage ID Mappings:")
    for title, pid in list(dict_page_id.items())[:10]:
        print(f"Title: {title}, Page ID: {pid}")
    if len(dict_page_id) > 10:
        print(f"... and {len(dict_page_id) - 10} more")
    
    # Process user query
    question = input("\nEnter your question: ")
    answer, page_contents, matched_titles = process_user_query(question, dict_page_id)
    if answer:
        print("\nMatched Pages:", ", ".join(matched_titles))
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()