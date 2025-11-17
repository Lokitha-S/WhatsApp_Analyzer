import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

@st.cache_resource
def download_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')

download_vader()



@st.cache_data
def parse_chat(uploaded_file, dayfirst_toggle):
    """
    Parses the uploaded WhatsApp chat file (handles .txt and .csv).
    
    Args:
        uploaded_file: The file object uploaded by the user.
        dayfirst_toggle (bool): True if the date format is Day/Month/Year.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Timestamp', 'Author', 'Message'].
    """
    file_name = uploaded_file.name
    
    if file_name.endswith('.csv'):
     
        try:
            df = pd.read_csv(uploaded_file)
            
           
            required_cols = ['Timestamp', 'Author', 'Message']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV file must contain the exact columns: {', '.join(required_cols)}")
                return pd.DataFrame(columns=['Timestamp', 'Author', 'Message'])

            try:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=dayfirst_toggle)
            except Exception as e:
                st.error(f"Error parsing 'Timestamp' column from CSV: {e}.")
                st.error("Please ensure your 'Timestamp' column is in a recognizable date format.")
                return pd.DataFrame(columns=['Timestamp', 'Author', 'Message'])
            
            st.info("CSV file loaded. Note: CSV parsing assumes pre-formatted columns.")
           
            return df[required_cols]

        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")
            return pd.DataFrame(columns=['Timestamp', 'Author', 'Message'])

    elif file_name.endswith('.txt'):
    
        data = uploaded_file.getvalue().decode("utf-8")
        
        pattern = re.compile(
            r'^(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[ap]m)\s-\s(.*?):\s(.*)$',
            re.MULTILINE
        )
        
        matches = pattern.findall(data)
        
        if not matches:
            st.error("Failed to parse the .txt file. Please ensure it's in the correct format (e.g., 'M/D/YY, H:MM am/pm - Author: Message').")
            st.info("Note: This parser expects the AM/PM time format. 24-hour formats may not work.")
            return pd.DataFrame(columns=['Timestamp', 'Author', 'Message'])

        df = pd.DataFrame(matches, columns=['Timestamp', 'Author', 'Message'])
        
   
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=dayfirst_toggle)
        except Exception as e:
            st.error(f"Error parsing timestamps even with 'mixed' format: {e}.")
            st.error("This can happen if your chat log contains system messages in a different language or format.")
            return pd.DataFrame(columns=['Timestamp', 'Author', 'Message'])
        # --- END OF FIX ---
                
        return df

    else:
        st.error("Unsupported file type. Please upload a .txt or .csv file.")
        return pd.DataFrame(columns=['Timestamp', 'Author', 'Message'])

@st.cache_data
def perform_sentiment_analysis(_df):
    """
    Performs sentiment analysis on the 'Message' column.
    Uses a copy to avoid Streamlit caching issues.
    
    Args:
        _df (pd.DataFrame): The chat DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with added sentiment columns.
    """
    df = _df.copy() 
    sid = SentimentIntensityAnalyzer()
    
    df['Message_str'] = df['Message'].astype(str)
    df['Sentiment'] = df['Message_str'].apply(lambda msg: sid.polarity_scores(msg))
    df['Compound_Score'] = df['Sentiment'].apply(lambda score_dict: score_dict['compound'])
    
    def get_sentiment_label(compound):
        if compound >= 0.05:
            return 'Positive'
        elif compound <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
            
    df['Sentiment_Label'] = df['Compound_Score'].apply(get_sentiment_label)
    return df

# --- Streamlit App ---

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("WhatsApp Chat Analyzer 📊")

st.sidebar.header("Upload Your Chat")
st.sidebar.markdown("""
1.  Open your WhatsApp chat.
2.  Tap on the group/contact name.
3.  Tap on `Export Chat`.
4.  Choose `Without Media`.
5.  Upload the `.txt` file here.
""")

uploaded_file = st.sidebar.file_uploader("Upload a WhatsApp chat (.txt or .csv)", type=["txt", "csv"])

dayfirst_toggle = st.sidebar.toggle(
    "Is your date format Day/Month/Year?", 
    value=False, 
    help="Toggle this if your dates look like 'DD/MM/YY' (e.g., 25/12/24) instead of 'MM/DD/YY' (e.g., 12/25/24)."
)

# --- Main App Body ---

if uploaded_file is not None:
    # --- 1. Parsing ---
    with st.spinner("Parsing your chat file..."):
        df_chat = parse_chat(uploaded_file, dayfirst_toggle)
    
    if not df_chat.empty:
        st.success("Chat file parsed successfully!")
        
        # --- NEW: Fun Animation ---
        st.balloons()
        
        # --- 2. ML: Sentiment Analysis ---
        with st.spinner("Running sentiment analysis model..."):
            df_chat_with_sentiment = perform_sentiment_analysis(df_chat)
        
        # --- NEW: Interactive Filters ---
        st.sidebar.markdown("---")
        st.sidebar.header("Dashboard Filters")
        
        all_authors = df_chat_with_sentiment['Author'].unique()
        # Select all authors by default
        selected_authors = st.sidebar.multiselect("Filter by Author:",
                                                  options=all_authors,
                                                  default=all_authors)

        if not selected_authors:
            st.warning("Please select at least one author to see the analysis.")
            st.stop()
            
        # --- Filter the DataFrame based on selection ---
        df_filtered = df_chat_with_sentiment[df_chat_with_sentiment['Author'].isin(selected_authors)]

        st.subheader("Filtered Dashboard")
        st.markdown(f"Showing analysis for **{len(selected_authors)}** selected author(s).")
        
        # --- 3. Top-Level Stats (Now based on df_filtered) ---
        total_messages = df_filtered.shape[0]
        total_media = df_filtered[df_filtered['Message'] == '<Media omitted>'].shape[0]
        
        df_filtered_text_only = df_filtered[df_filtered['Message'] != '<Media omitted>']
        total_words = df_filtered_text_only['Message'].apply(lambda s: len(s.split())).sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", f"{total_messages:,}")
        col2.metric("Total Words", f"{total_words:,}")
        col3.metric("Media Messages", f"{total_media:,}")
        
        with st.expander("Show Filtered Data"):
            st.dataframe(df_filtered)

        # --- 4. Analysis & Visualization (Now based on df_filtered) ---
        st.markdown("---")
        st.subheader("Deep Dive Analysis")

        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            # --- Most Active Users ---
            st.markdown("#### Most Active Users (from selection)")
            author_counts = df_filtered['Author'].value_counts().head(10)
            fig_author = px.bar(author_counts, 
                                x=author_counts.values, 
                                y=author_counts.index, 
                                orientation='h',
                                title="Top 10 Most Active Users",
                                labels={'x': 'Number of Messages', 'y': 'User'},
                                color=author_counts.values,
                                color_continuous_scale=px.colors.sequential.Cividis)
            fig_author.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_author, use_container_width=True)

            # --- Word Cloud ---
            st.markdown("#### Most Common Words")
            text_for_cloud = " ".join(df_filtered_text_only['Message'])
            if text_for_cloud:
                wordcloud = WordCloud(width=800, 
                                      height=400, 
                                      background_color='white',
                                      colormap='viridis',
                                      stopwords=None).generate(text_for_cloud)
                
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc)
            else:
                st.info("Not enough text from selected authors to generate a word cloud.")

        with viz_col2:
            # --- ML: Sentiment Analysis Results ---
            st.markdown("#### F🤖 ML: Sentiment Analysis")
            sentiment_counts = df_filtered['Sentiment_Label'].value_counts()
            fig_sentiment = px.pie(sentiment_counts, 
                                   values=sentiment_counts.values, 
                                   names=sentiment_counts.index,
                                   title="Overall Chat Sentiment",
                                   color=sentiment_counts.index,
                                   color_discrete_map={'Positive':'#2ca02c', # Green
                                                       'Neutral':'#8c8c8c', # Grey
                                                       'Negative':'#d62728'}) # Red
            fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # --- Activity Heatmap ---
            st.markdown("#### Chat Activity Heatmap")
            # --- FIX: We must handle the SettingWithCopyWarning ---
            # Create a copy to safely add new columns
            df_filtered_heatmap = df_filtered.copy()
            df_filtered_heatmap['Day_of_Week'] = df_filtered_heatmap['Timestamp'].dt.day_name()
            df_filtered_heatmap['Hour'] = df_filtered_heatmap['Timestamp'].dt.hour
            
            activity_pivot = df_filtered_heatmap.pivot_table(index='Day_of_Week', 
                                                             columns='Hour', 
                                                             values='Message', 
                                                             aggfunc='count')
            
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            hours = list(range(24)) # Explicitly define all 24 hours
            
            # --- THIS IS THE FIX ---
            # Reindex both rows and columns to create a full 7x24 grid.
            # Fill any missing (NaN) values with 0.
            activity_pivot = activity_pivot.reindex(index=days, columns=hours, fill_value=0)
            
            fig_heatmap = px.imshow(activity_pivot,
                                    labels=dict(x="Hour of Day", y="Day of Week", color="Messages"),
                                    x=[f"{h}:00" for h in hours], # Use the 'hours' list
                                    y=days,
                                    title="Message Activity by Day and Hour",
                                    color_continuous_scale=px.colors.sequential.Viridis)
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # --- Message Timeline ---
        st.markdown("---")
        st.subheader("Message Timeline")
        timeline = df_filtered.set_index('Timestamp').resample('D').count()['Message']
        fig_timeline = px.line(timeline, 
                               x=timeline.index, 
                               y=timeline.values,
                               title="Messages Over Time",
                               labels={'x': 'Date', 'y': 'Number of Messages'})
        fig_timeline.update_traces(line_color='#636EFA', line_width=3)
        st.plotly_chart(fig_timeline, use_container_width=True)

    else:
        # This will show if parsing failed
        st.warning("Could not parse the chat file. Please check the sidebar for instructions.")

else:
    st.info("Please upload your WhatsApp .txt file from the sidebar to begin analysis.")