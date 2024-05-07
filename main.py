from flask import Flask, request, render_template_string
import os
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from dotenv import load_dotenv

# Load environment variables
load_dotenv('./.env')

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>URL Content Extractor</title>
</head>
<body>
    <h1>Enter a URL to Extract Content</h1>
    <form method="post">
        <input type="text" name="url" placeholder="Enter URL here" required>
        <button type="submit">Extract</button>
    </form>
    {% if extracted_content %}
        <h2>Extracted Content:</h2>
        <pre>{{ extracted_content }}</pre>
    {% endif %}
</body>
</html>
'''

def run_playwright(site):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(site, timeout=60000)
        page_source = page.content()
        soup = BeautifulSoup(page_source, "html.parser")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        data = '\n'.join(chunk for chunk in chunks if chunk)
        browser.close()
    return data

@app.route('/', methods=['GET', 'POST'])
def index():
    extracted_content = None
    if request.method == 'POST':
        url = request.form['url']
        try:
            raw_text = run_playwright(url)
            openai_api_key = os.getenv('OPENAI_API_KEY')
            llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=openai_api_key)
            structured_schema = {
                "properties": {
                    "company_name": {"type": "string"},
                    "contacts": {"type": "string"},
                    "industries_that_they_invest_in": {"type": "string"},
                    "investment_rounds_that_they_participate/lead": {"type": "string"}
                },
                "required": ["company_name"],
            }
            extraction_chain = create_extraction_chain(structured_schema, llm)
            extracted_content = extraction_chain.run(raw_text)
        except Exception as e:
            extracted_content = f"Error processing the URL: {e}"
    return render_template_string(HTML_TEMPLATE, extracted_content=extracted_content)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
