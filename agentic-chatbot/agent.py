import os
from typing import Annotated, TypedDict
from io import BytesIO
import base64

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field
import google.generativeai as genai
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")  # To use non-GUI backend

genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "")) 
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

class StockAnalysisInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    period: str = Field("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")
    chart_type: str = Field("line", description="Chart type (line, candle)")

@tool(args_schema=StockAnalysisInput)
def StockAnalysisTool(ticker: str, period: str = "1y", chart_type: str = "line") -> dict:
    """
    Fetches stock data based on ticker symbol, using default variables if unspecified.
    Use this tool to get real-time financial data for stocks.
    Always use this tool with abbreviations for time period according to yfinance api, when asked about specific stocks or market data.
    You can also specify the chart type as 'line' or 'candle' for different visualizations.
    You can also invoke the tool multiple times with different ticker symbols to compare stock data.
    """
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period=period)
        
        # Basic stats
        if hist.empty:
            return {"error": f"No data found for ticker {ticker}"}
        
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[0]
        price_change = current_price - prev_price
        percent_change = (price_change / prev_price) * 100
        
        # Create chart
        plt.figure(figsize=(10, 6))
        
        if chart_type.lower() == "candle" and len(hist) > 1:
            from mplfinance.original_flavor import candlestick_ohlc
            import matplotlib.dates as mpdates
            
            # Convert date to numeric format
            hist_plot = hist.reset_index()
            hist_plot['Date'] = hist_plot['Date'].map(mpdates.date2num)
            
            # Create OHLC list
            ohlc = hist_plot[['Date', 'Open', 'High', 'Low', 'Close']].values
            
            # Plot
            candlestick_ohlc(plt.gca(), ohlc, width=0.6, colorup='green', colordown='red')
            plt.gca().xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d'))
        else:
            hist['Close'].plot(title=f"{ticker} Stock Price")
            plt.grid(True)
        
        plt.title(f"{ticker} Stock Price Chart ({period})")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.tight_layout()
        
        # Convert plot to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Company info
        company_name = info.get('shortName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        
        # Financial metrics
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        dividend_yield = info.get('dividendYield', 'N/A')
        if dividend_yield != 'N/A':
            dividend_yield = round(dividend_yield * 100, 2)
            
        # Additional metrics for better analysis
        beta = info.get('beta', 'N/A')
        fifty_two_week_high = info.get('fiftyTwoWeekHigh', 'N/A')
        fifty_two_week_low = info.get('fiftyTwoWeekLow', 'N/A')
        avg_volume = info.get('averageVolume', 'N/A')
        
        # Calculate some basic technical indicators
        if len(hist) >= 50:
            # 50-day moving average
            hist['MA50'] = hist['Close'].rolling(window=50).mean()
            ma_50 = hist['MA50'].iloc[-1]
            ma_50_status = "above" if current_price > ma_50 else "below"
        else:
            ma_50 = "Insufficient data"
            ma_50_status = "unknown"
            
        if len(hist) >= 200:
            # 200-day moving average
            hist['MA200'] = hist['Close'].rolling(window=200).mean()
            ma_200 = hist['MA200'].iloc[-1]
            ma_200_status = "above" if current_price > ma_200 else "below"
        else:
            ma_200 = "Insufficient data"
            ma_200_status = "unknown"
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "current_price": round(current_price, 2),
            "price_change": round(price_change, 2),
            "percent_change": round(percent_change, 2),
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "dividend_yield": dividend_yield,
            "beta": beta,
            "52_week_high": fifty_two_week_high,
            "52_week_low": fifty_two_week_low,
            "average_volume": avg_volume,
            "ma_50": ma_50 if isinstance(ma_50, str) else round(ma_50, 2),
            "ma_50_status": ma_50_status,
            "ma_200": ma_200 if isinstance(ma_200, str) else round(ma_200, 2),
            "ma_200_status": ma_200_status,
            "chart_image": f"data:image/png;base64,{image_base64}"
        }
    
    except Exception as e:
        return {"error": f"Error analyzing stock: {str(e)}"}

class State(TypedDict):
        messages: Annotated[list, add_messages]

def start_workflow(): 
    tools = [StockAnalysisTool]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        print(type(state["messages"]))
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    while True:
        config = {"configurable": {"thread_id": "1"}}
        graph = start_workflow()
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        system_prompt = """You are a stock market analyst. Always respond in this structured format:
        - **Company Overview:** [Brief summary]
        - **Financial Performance:** [Key financial highlights]
        - **Market Trends:** [Current industry and market trends]
        - **Investment Risks:** [Key risks to consider]
        - **Final Thoughts:** [Summary of the analysis]
        Keep responses short and fact-based.
        Always retrieve stock data before analysis.
        Use currency names like USD instead of $.
        Transform large numbers to short ones in human readable format like 1.7 million instead of 1,700,000.
        Remember that you can invoke the tools at your disposal as many times as you need to give complete response.
        """   

        for event in graph.stream({"messages": [("system", system_prompt), ("user", user_input)]}, config):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)