import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
from googlesearch import search

# Load environment variables
load_dotenv()

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

def google_search(query):
    """
    Scrapes Google search results for the given query
    Returns the first relevant result
    """
    try:
        # Format the search URL
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
        
        # Make the request
        response = requests.get(search_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the first search result
        search_results = soup.find_all('div', {'class': 'g'})
        
        if search_results:
            first_result = search_results[0]
            
            # Extract title and link
            title_element = first_result.find('h3')
            link_element = first_result.find('a')
            
            if title_element and link_element:
                title = title_element.text
                link = link_element['href']
                
                # Try to get the description
                desc_element = first_result.find('div', {'class': 'VwiC3b'})
                description = desc_element.text if desc_element else "No description available"
                
                # Clean up the title and description
                title = title.split('-')[0].strip()  # Remove anything after a dash
                description = description.split('.')[0].strip()  # Take only the first sentence
                
                return {
                    'title': title,
                    'link': link,
                    'description': description
                }
        return None
    except Exception:
        return None

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
    except smtplib.SMTPAuthenticationError:
        pass
    except Exception:
        pass

def main():
    search_query = input("Enter the search query: ")
    result = google_search(search_query)
    if result:
        print(f"Description: {result['description']}")
        email_body = f"Description: {result['description']}"
        send_email("Web Scraping Results", email_body)
    else:
        print("No results found")

if __name__ == "__main__":
    main()