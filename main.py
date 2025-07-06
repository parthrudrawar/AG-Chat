import asyncio
from playwright.async_api import async_playwright
from urllib.parse import urljoin, urlparse, quote
import re
import json
import time
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

def extract_page_id(url):
    """Extract page ID and title from a Confluence URL."""
    pattern = r'/wiki/spaces/IID/pages/(\d+)/([^/]+)'
    match = re.search(pattern, url)
    if match:
        page_id = match.group(1)
        title = match.group(2).replace('+', ' ')
        return {'page_id': page_id, 'title': title}
    return None

async def get_all_links(base_url="https://inferyx.atlassian.net/wiki/spaces/IID/pages", max_depth=6):
    """Extract all URLs from a website, crawling internal links up to max_depth, and return a set of URLs."""
    all_links = set()
    visited_urls = set()
    
    async def crawl_page(page, url, depth):
        if depth > max_depth or url in visited_urls:
            return
        print(f"▶️ Crawling {url} (Depth: {depth})")
        visited_urls.add(url)
        
        try:
            await page.goto(url, timeout=30000)
            await page.wait_for_timeout(7000)
            for round_num in range(20):
                print(f"⏳ Scrolling round {round_num + 1} for {url}")
                await page.mouse.wheel(0, 2000)
                await page.wait_for_timeout(5000)
                try:
                    load_more = page.locator("button:has-text('Load more')")
                    if await load_more.is_visible():
                        print(f"👆 Clicking Load More at {url}")
                        await load_more.click()
                        await page.wait_for_timeout(6000)
                except:
                    pass
            anchors = await page.locator('a[href]').all()
            new_links = 0
            for a in anchors:
                href = await a.get_attribute("href")
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue
                full_url = urljoin(base_url, href)
                parsed_url = urlparse(full_url)
                if not parsed_url.netloc:
                    full_url = urljoin(base_url, full_url)
                    parsed_url = urlparse(full_url)
                if full_url not in all_links:
                    all_links.add(full_url)
                    new_links += 1
                    print(f"Found URL: {full_url}")
                    if parsed_url.netloc == urlparse(base_url).netloc and '/wiki/spaces/IID/pages/' in full_url:
                        try:
                            await page.goto(full_url, timeout=15000)
                            await crawl_page(page, full_url, depth + 1)
                        except:
                            print(f"❌ Skipping inaccessible URL: {full_url}")
            print(f"🔄 {url}: Found {new_links} new links")
        except Exception as e:
            print(f"❌ Error crawling {url}: {e}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        # Optional: Handle authentication
        # await page.goto("https://inferyx.atlassian.net/login")
        # await page.fill("#username", "your_username")
        # await page.fill("#password", "your_password")
        # await page.click("#login-submit")
        # await page.wait_for_timeout(5000)
        await crawl_page(page, base_url, 0)
        await browser.close()
    
    return all_links

async def get_page_content(title, dict_page_id):
    """Take a page title, map to its page ID, and return the content of the corresponding Confluence page."""
    page_id = dict_page_id.get(title)
    if not page_id:
        print(f"❌ Error: No page ID found for title '{title}'")
        return None
    encoded_title = quote(title.replace(' ', '+'))
    url = f"https://inferyx.atlassian.net/wiki/spaces/IID/pages/{page_id}/{encoded_title}"
    print(f"▶️ Fetching content from {url}")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        # Optional: Handle authentication
        # await page.goto("https://inferyx.atlassian.net/login")
        # await page.fill("#username", "your_username")
        # await page.fill("#password", "your_password")
        # await page.click("#login-submit")
        # await page.wait_for_timeout(5000)
        try:
            await page.goto(url, timeout=30000)
            await page.wait_for_timeout(7000)
            for round_num in range(5):
                print(f"⏳ Scrolling round {round_num + 1} for {url}")
                await page.mouse.wheel(0, 2000)
                await page.wait_for_timeout(2000)
            content_locator = page.locator('div#wiki-page-body, div#content-body, article')
            if await content_locator.is_visible():
                content = await content_locator.inner_text()
                if content.strip():
                    print(f"✅ Content retrieved for '{title}'")
                    return content.strip()
                else:
                    print(f"❌ Error: No content found in the main content area for '{title}'")
                    return None
            else:
                print(f"❌ Error: Content area not found for '{title}'")
                return None
        except Exception as e:
            print(f"❌ Error fetching content from {url}: {e}")
            return None
        finally:
            await browser.close()

def extract_entity(question):
    """Use LLM to extract the page title (entity) from the user's question via Hugging Face Inference API."""
    try:
        client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)
        prompt = f"""
        Extract the entity that likely corresponds to a Confluence page title from the following question. Return only the entity as a string.
        Question: {question}
        """
        output = client.text_generation(prompt, max_new_tokens=200)
        entity = output.strip().split('\n')[-1].strip()
        print(f"✅ Extracted entity: {entity}")
        return entity
    except Exception as e:
        print(f"❌ Error in entity extraction: {e}")
        return None

def feed_content_to_llm(content, question):
    """Feed content and question to LLM via Hugging Face Inference API and return the answer."""
    if not content:
        print("❌ Error: No content provided to LLM")
        return None
    
    try:
        client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)
        prompt = f"""
        Based on the following Confluence page content, answer the question in 3-5 sentences.

        Content:
        {content[:2000]}  # Truncate to avoid token limit

        Question:
        {question}
        """
        output = client.text_generation(prompt, max_new_tokens=1000)
        # Extract the answer (remove prompt from output)
        answer = output[len(prompt):].strip()
        print("✅ LLM generated answer")
        return answer
    except Exception as e:
        print(f"❌ Error in LLM processing: {e}")
        return None

async def process_user_query(question, dict_page_id):
    """Process a user question by extracting the entity, fetching content, and generating an answer."""
    title = extract_entity(question)
    if not title:
        print("❌ Error: Could not extract a valid title from the question")
        return None
    
    if title not in dict_page_id:
        print(f"❌ Error: Title '{title}' not found in page ID mappings")
        return None
    
    content = await get_page_content(title, dict_page_id)
    if not content:
        print(f"❌ Error: Could not retrieve content for title '{title}'")
        return None
    
    answer = feed_content_to_llm(content, question)
    if not answer:
        print("❌ Error: LLM failed to generate an answer")
        return None
    
    return answer

async def main():
    # Step 1: Crawl and build dict_page_id
    all_links = await get_all_links()
    print(f"\nWe have {len(all_links)} links")
    
    dict_page_id = {}
    for link in all_links:
        page_info = extract_page_id(link)  # Fixed: Corrected 'L' to 'link'
        if page_info:
            dict_page_id[page_info['title']] = page_info['page_id']
    
    # Save results to JSON
    with open('inferyx_doc_links.json', 'w') as f:
        json.dump({'urls': list(all_links), 'page_id_mappings': dict_page_id}, f, indent=2)
    
    print("\nPage ID Mappings:")
    for title, pid in dict_page_id.items():
        print(f"Title: {title}, Page ID: {pid}")
    
    # Step 2: Process user query
    question = input("\nEnter your question: ")
    answer = await process_user_query(question, dict_page_id)
    if answer:
        print("\nAnswer:")
        print(answer)
    
    print(f"\n✅ Done. Saved {len(all_links)} URLs and {len(dict_page_id)} mappings to inferyx_doc_links.json")

if __name__ == "__main__":
    asyncio.run(main())