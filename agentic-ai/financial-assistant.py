from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


# Web Search Agent
search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information",
    model=OpenAIChat(id='gpt-4o-mini'),
    tools=[
        DuckDuckGo()
        ],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Financial Agent
finance_agent = Agent(
    name="Finance Agent",
    role="Investment Analyst for stock prices and recommendations",
    model=OpenAIChat(id='gpt-4o-mini'),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True)
        ],
    instructions=["Format your response using markdown and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True
)

# Combines all agent as a team
multi_ai_agent = Agent(
    model=OpenAIChat(id='gpt-4o-mini'),
    team=[search_agent, finance_agent],
    instructions=["Always include sources", "Use table to display data"],
    show_tool_calls=True,
    markdown=True
)

# Query
multi_ai_agent.print_response("Summarize analyst recommendation and share the latest news for NVDA", stream=True)