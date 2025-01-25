from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()
result = search.run("Who is the president of South Korea?")

print(result)