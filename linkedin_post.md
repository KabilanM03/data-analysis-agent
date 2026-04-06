# LinkedIn Post Draft

---

I built a Data Analysis Agent that lets you explore any CSV dataset using plain English — no SQL or Python required.

You upload a file. You ask a question. The agent figures out what to do and gives you the answer.

**What it can do:**
- Describe and profile your dataset automatically
- Filter, aggregate, and summarise data on request
- Generate 6 chart types (bar, line, scatter, histogram, box, heatmap)
- Run correlation analysis across numeric columns
- Produce a structured markdown report with findings and recommendations

**What it's built with:**
- smolagents (Hugging Face) — the agent framework
- Qwen2.5-72B as the reasoning model
- pandas + seaborn for analysis and visualisation
- Gradio for the chat interface

The agent uses a CodeAgent — it writes and runs Python code to answer your question, rather than following a fixed pipeline. So it can handle unexpected questions without breaking.

This project came out of working through the Hugging Face AI Agents Course, which covers how agents reason, plan, and use tools — and how to build them from scratch using smolagents, LlamaIndex, and LangGraph.

GitHub repo + full documentation in the comments.

---

**Tags to add:** #AI #DataAnalysis #MachineLearning #Python #HuggingFace #Agents #DataScience #smolagents #OpenSource #LLM
