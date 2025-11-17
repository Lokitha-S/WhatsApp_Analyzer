WhatsApp Chat Analyzer 📱💬

This is an interactive web application built with Streamlit that provides a comprehensive analysis of your WhatsApp chat history.

Upload your exported chat file (either .txt or .csv) and the app will generate interactive visualizations and key metrics. It features a robust parser that can handle different file types and date formats, and it uses an NLTK-based machine learning model to perform sentiment analysis on your messages.

Core Features

Flexible Upload: Natively parses both exported WhatsApp .txt files and pre-formatted .csv files.

Smart Date Parser: Includes a Day/Month/Year toggle to correctly interpret ambiguous timestamps (e.g., 10/11/25), preventing common parsing errors.

Interactive Dashboard: The entire dashboard is filterable by one or more authors. Selecting users will dynamically update all charts and metrics.

Key Metrics: Get at-a-glance "Top Stats" for:

Total Messages

Total Words

Total Media Files Shared

Machine Learning (NLP):

Sentiment Analysis: A pie chart breaks down the overall chat sentiment (Positive, Negative, Neutral) using NLTK's pre-trained VADER model.

Detailed Visualizations (via Plotly):

Most Active Users: A bar chart of the top 10 most active members.

Activity Heatmap: A 7x24 grid showing the most active times of day and days of the week.

Message Timeline: A line chart showing the volume of messages over time to spot trends.

Word Cloud: A visual representation of the most frequently used words in the chat.

Robust Error Handling: Provides clear error messages if the file format is incorrect or timestamps cannot be parsed.

Technology Stack

Web Framework: Streamlit

Data Manipulation: Pandas

Machine Learning (NLP): NLTK (VADER for sentiment)

Data Visualization: Plotly (for interactive charts), Matplotlib & Seaborn (for the heatmap), and WordCloud.