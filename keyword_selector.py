import streamlit as st
import pandas as pd
import re
import nltk
import base64
from collections import Counter  # Import Counter for word frequency
import string

# Download the NLTK stopwords dataset and punkt tokenizer
nltk.download('stopwords')
nltk.download('punkt')

# Import stopwords
from nltk.corpus import stopwords

# Function to clean and concatenate text
def clean_and_concat(row):
    title = row['Title'] if not pd.isna(row['Title']) else ""
    abstract = row['Abstract'] if not pd.isna(row['Abstract']) else ""
    author_keywords = row['Author Keywords'] if not pd.isna(row['Author Keywords']) else ""
    index_keywords = row['Index Keywords'] if not pd.isna(row['Index Keywords']) else ""

    # Remove punctuation from Title and Abstract
    title = title.translate(str.maketrans('', '', string.punctuation))
    abstract = abstract.translate(str.maketrans('', '', string.punctuation))

   # Remove numbers from Title and Abstract
    title = re.sub(r'\d+', '', title)
    abstract = re.sub(r'\d+', '', abstract)

    # Remove text including and after the copyright symbol (©) in the Abstract
    abstract = re.sub(r'©.*', '', abstract)

    # Tokenize Title and Abstract
    title_tokens = nltk.word_tokenize(title)
    abstract_tokens = nltk.word_tokenize(abstract)

    # Remove stopwords from Title and Abstract
    title_tokens = [word for word in title_tokens if word.lower() not in stopwords.words('english')]
    abstract_tokens = [word for word in abstract_tokens if word.lower() not in stopwords.words('english')]

    # Concatenate Title and Abstract with semicolon delimiter
    summary = '; '.join(title_tokens + abstract_tokens)

    # Concatenate Author Keywords, Index Keywords, and Summary
    summary = summary + " " + author_keywords + " " + index_keywords

    return summary

# Streamlit app
def main():
    st.title("Keyword Selector")

    # File Upload
    uploaded_file = st.file_uploader("Upload a CSV file from Scopus. Your file must include, at minimum, the columns for Title, Abstract, Author Keywords, and Index Keywords.", type=["csv"])

    if uploaded_file is not None:
        # Display a message while processing
        st.text("Processing the file... Please wait.")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Create a new 'Summary' column by applying the clean_and_concat function
        df['Summary'] = df.apply(clean_and_concat, axis=1)

        # Export the updated DataFrame to a new CSV file
        st.subheader("Updated CSV File:")
        st.dataframe(df, height=400)

        # Download the updated CSV file
        st.markdown(get_csv_download_link(df), unsafe_allow_html=True)

        # Display a message while processing the top 1,000 words
        st.text("Processing the top 1,000 words... Please wait.")

        # Export the most frequent words in the Summary column
        st.markdown(get_top_words_csv(df), unsafe_allow_html=True)

        # Display the final message
        st.markdown("Done! Now you may download your updated spreadsheet and a list of the top 1,000 keywords overall. Please note that some phrases such as 'Carbon Dioxide' may have been recognized as separate terms by the program.")


# Function to create a download link for a DataFrame as a CSV file
def get_csv_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="updated_keywords.csv">Download Updated CSV File</a>'
    return href

# Function to get the 1,000 most frequent words (excluding punctuation) from the Summary column and export them to a CSV
def get_top_words_csv(df):
    # Combine all the Summary text into a single string
    all_summary_text = ' '.join(df['Summary'].tolist())

    # Tokenize the text and count word frequencies (excluding punctuation)
    words = nltk.word_tokenize(all_summary_text)
    # Exclude punctuation and stopwords
    words = [word for word in words if word not in string.punctuation and word.lower() not in stopwords.words('english')]

    word_counts = Counter(words)

    # Get the 1,000 most frequent words
    most_common_words = word_counts.most_common(1000)

    # Create a DataFrame with the most frequent words
    top_words_df = pd.DataFrame(most_common_words, columns=['Word', 'Frequency'])

    # Export the top words to a CSV
    top_words_csv = top_words_df.to_csv(index=False)
    b64 = base64.b64encode(top_words_csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="top_words.csv">Download Top Words CSV</a>'
    return href

if __name__ == "__main__":
    main()
