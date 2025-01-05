from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
from dotenv import load_dotenv
import os


import phi 
from phi.playground import Playground, serve_playground_app


load_dotenv()
phi.api = os.getenv("PHI_API_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")



##Web Search Agent

websearch_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for the information",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions = ["Always include source"],
    show_tools_calls = True,
    markdown = True

)


## Financial Agent

finance_agent = Agent(
    name = "Finance Ai Agent",
    model = Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    instructions = ["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True

)

multi_ai_agent = Agent(
    team = [websearch_agent,finance_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include source","Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)


app = Playground(agents=[finance_agent,websearch_agent]).get_app()

if __name__=='__main__':
    serve_playground_app("playground:app",reload = True)



    