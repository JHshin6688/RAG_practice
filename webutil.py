import requests
from bs4 import BeautifulSoup

def crawl(url):
    #url = "https://www.nfl.com/news/all-news"
    response = requests.get(url)

    page_links = []
    header = "https://www.nfl.com"

    # Check if request was successful
    if response.status_code == 200:  
        soup = BeautifulSoup(response.text, "html.parser")

        links = soup.find_all("a")
        for link in links:
            page = link.get("href")
            if page != "#main-content":
                #print("Link:", header + page)
                page_links.append(header + page)
    else:
        print("Failed to retrieve the webpage")
    
    return page_links