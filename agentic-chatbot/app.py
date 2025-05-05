import streamlit as st
import base64
from io import BytesIO
import json

from agent import start_workflow

# Initialize workflow graph only once per session
if "graph" not in st.session_state:
    st.session_state.graph = start_workflow()

if "messages" not in st.session_state:
    st.session_state.messages = [
        ("system", """You are a stock market analyst. Always respond in this structured format:
        - **Company Overview:** [Brief summary]
        - **Financial Performance:** [Key financial highlights]
        - **Market Trends:** [Current industry and market trends]
        - **Investment Risks:** [Key risks to consider]
        - **Final Thoughts:** [Summary of the analysis]
        Keep responses short and fact-based.
        Always retrieve stock data before analysis.
        Use currency names like USD instead of $.
        Transform large numbers to short ones in human readable format like 1.7 million instead of 1,700,000.
        Remember that you can invoke the tools at your disposal as many times as you need to give complete response."""   
        )
    ]

# Streamlit UI Setup
st.set_page_config(page_title="Stock Market Chatbot", layout="centered")
st.title("📈 Stock Market Analysis Chatbot")

# Display chat history
for role, text in st.session_state.messages:
    if role!="system":
        if text.startswith("data:image/png;base64,"):
            base64_data = text.split(",")[1]
            image_data = BytesIO(base64.b64decode(base64_data))
            st.image(image_data, caption="Stock Chart", use_container_width=True)
        else:
            with st.chat_message(role):
                # print(st.session_state.messages)
                st.write(text)

# User input
user_input = st.chat_input("Ask about a stock (e.g., 'Analyze TSLA for 6 months')")
if user_input:
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # Process response in real-time
    config = {"configurable": {"thread_id": "1"}}
    response = None
    chart_image = []

    for event in st.session_state.graph.stream({"messages": st.session_state.messages}, config):
        for value in event.values():
            response = value["messages"][-1].content  # Extract text response
            # To retrieve chart images
            for v in value["messages"]:
                if v.content and v.content[0]=='{' and "chart_image" in json.loads(v.content):
                    chart_image.append(json.loads(v.content).get("chart_image", None))  # Extract chart image (if available)

    if response:
        st.session_state.messages.append(("assistant", response))
        # st.session_state.messages.append(("assistant", chart_image))
        with st.chat_message("assistant"):
            print(response)
            st.write(response)
            
            # Check if a chart image is available and display it
            if chart_image:
                for chart in chart_image:
                    if chart.startswith("data:image/png;base64,"):  # If it's Base64 encoded
                        base64_data = chart.split(",")[1]
                        image_data = BytesIO(base64.b64decode(base64_data))
                        st.session_state.messages.append(("assistant", chart))
                        st.image(image_data, caption="Stock Chart", use_container_width=True)
                    elif chart.startswith("http"):  # If it's a URL
                        st.session_state.messages.append(("assistant", chart))
                        st.image(chart, caption="Stock Chart", use_container_width=True)
                    else:
                        st.error("Unsupported image format received.")