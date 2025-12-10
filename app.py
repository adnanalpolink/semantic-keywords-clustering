import streamlit as st
import pandas as pd
import time
from collections import Counter
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Semantic Keyword Clustering",
    page_icon="🔗",
    layout="wide"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_model(model_name: str):
    """Load and cache the SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def create_unigram(cluster: str) -> str:
    """Create unigram from the cluster and return the most common word."""
    if pd.isna(cluster) or cluster == "":
        return ""
    words = str(cluster).split()
    if not words:
        return ""
    most_common_word = Counter(words).most_common(1)[0][0]
    return most_common_word


def detect_encoding(uploaded_file):
    """Detect file encoding."""
    import chardet
    raw_data = uploaded_file.read()
    result = chardet.detect(raw_data)
    uploaded_file.seek(0)  # Reset file pointer
    return result["encoding"]


def load_csv(uploaded_file):
    """Load CSV file with automatic encoding detection."""
    encoding = detect_encoding(uploaded_file)
    
    try:
        if encoding and "UTF-16" in encoding.upper():
            df = pd.read_csv(uploaded_file, encoding=encoding, delim_whitespace=True, on_bad_lines='skip')
        else:
            df = pd.read_csv(uploaded_file, encoding=encoding or 'utf-8', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Error reading file: {e}")
        # Fallback to utf-8
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
    
    return df


def cluster_keywords(df, column_name, model_name, min_similarity, remove_dupes, device="cpu"):
    """Perform semantic clustering on keywords."""
    from sentence_transformers import SentenceTransformer
    from polyfuzz import PolyFuzz
    from polyfuzz.models import SentenceEmbeddings
    
    # Prepare dataframe
    df = df.copy()
    df.rename(columns={column_name: 'keyword'}, inplace=True)
    
    if remove_dupes:
        df.drop_duplicates(subset='keyword', inplace=True)
    
    df = df[df["keyword"].notna()]
    df['keyword'] = df['keyword'].astype(str)
    
    # Filter out empty strings
    df = df[df['keyword'].str.strip() != '']
    
    if len(df) == 0:
        st.error("No valid keywords found in the selected column.")
        return None
    
    from_list = df['keyword'].to_list()
    
    # Load model and create embeddings
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Loading embedding model...")
    progress_bar.progress(10)
    
    embedding_model = SentenceTransformer(model_name, device=device)
    distance_model = SentenceEmbeddings(embedding_model)
    
    status_text.text("Clustering keywords...")
    progress_bar.progress(30)
    
    start_time = time.time()
    
    model = PolyFuzz(distance_model)
    model = model.fit(from_list)
    
    progress_bar.progress(60)
    status_text.text("Grouping clusters...")
    
    model.group(link_min_similarity=min_similarity)
    
    progress_bar.progress(80)
    status_text.text("Finalizing results...")
    
    df_cluster = model.get_matches()
    df_cluster.rename(columns={"From": "keyword", "Similarity": "similarity", "Group": "spoke"}, inplace=True)
    
    df = pd.merge(df, df_cluster[['keyword', 'spoke']], on='keyword', how='left')
    
    # Calculate cluster sizes
    df['cluster_size'] = df['spoke'].map(df.groupby('spoke')['spoke'].count())
    df.loc[df["cluster_size"] == 1, "spoke"] = "no_cluster"
    
    # Clean up spoke column
    df['spoke'] = df['spoke'].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')
    
    # Create hub (unigram) column
    df['hub'] = df['spoke'].apply(create_unigram)
    
    # Reorder columns
    cols = ['hub', 'spoke', 'cluster_size', 'keyword'] + [col for col in df.columns if col not in ['hub', 'spoke', 'cluster_size', 'keyword']]
    df = df[[col for col in cols if col in df.columns]]
    
    # Sort by spoke and cluster size
    df.sort_values(["spoke", "cluster_size"], ascending=[True, False], inplace=True)
    
    # Clean up spoke formatting
    df['spoke'] = df['spoke'].str.split().str.join(' ')
    
    elapsed_time = time.time() - start_time
    
    progress_bar.progress(100)
    status_text.text(f"✅ Clustering complete! Took {elapsed_time:.2f} seconds")
    
    return df


def create_chart(df, chart_type):
    """Create visualization chart."""
    import plotly.express as px
    
    # Prepare data for chart (need cluster_size)
    chart_df = df.copy()
    if 'cluster_size' not in chart_df.columns:
        chart_df['cluster_size'] = chart_df['spoke'].map(chart_df.groupby('spoke')['spoke'].count())
    
    # Remove no_cluster for cleaner visualization
    chart_df = chart_df[chart_df['spoke'] != 'no_cluster']
    
    if len(chart_df) == 0:
        st.warning("No clusters found for visualization (all keywords are unique).")
        return None
    
    if chart_type == "sunburst":
        fig = px.sunburst(
            chart_df, 
            path=['hub', 'spoke'], 
            values='cluster_size',
            color_discrete_sequence=px.colors.qualitative.Pastel2,
            title="Keyword Clusters - Sunburst Chart"
        )
    elif chart_type == "treemap":
        fig = px.treemap(
            chart_df, 
            path=['hub', 'spoke'], 
            values='cluster_size',
            color_discrete_sequence=px.colors.qualitative.Pastel2,
            title="Keyword Clusters - Treemap"
        )
    else:
        return None
    
    fig.update_layout(height=700)
    return fig


def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download."""
    return df.to_csv(index=False).encode('utf-8')


# Main App
def main():
    st.title("🔗 Semantic Keyword Clustering")
    st.markdown("""
    This app clusters keywords based on their semantic similarity using SentenceTransformers and PolyFuzz.
    Upload a CSV file with keywords to get started.
    """)
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    
    # Model selection
    model_options = {
        "paraphrase-MiniLM-L3-v2": "Fastest (Lower accuracy)",
        "all-MiniLM-L6-v2": "Balanced (Recommended)",
        "all-mpnet-base-v2": "Slowest (Highest accuracy)"
    }
    
    model_name = st.sidebar.selectbox(
        "Embedding Model",
        options=list(model_options.keys()),
        index=1,
        format_func=lambda x: f"{x} - {model_options[x]}"
    )
    
    # Similarity threshold
    min_similarity = st.sidebar.slider(
        "Minimum Similarity",
        min_value=0.5,
        max_value=0.99,
        value=0.85,
        step=0.01,
        help="Higher values create tighter, more specific clusters"
    )
    
    # Remove duplicates option
    remove_dupes = st.sidebar.checkbox("Remove Duplicates", value=True)
    
    # Chart type
    chart_type = st.sidebar.selectbox(
        "Chart Type",
        options=["treemap", "sunburst"],
        index=0
    )
    
    # Device selection
    device = st.sidebar.selectbox(
        "Processing Device",
        options=["cpu", "cuda"],
        index=0,
        help="Select 'cuda' if you have a GPU available"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Tips:**
    - Use higher similarity (0.90+) for tighter clusters
    - Use lower similarity (0.70-0.80) for broader groupings
    - The 'hub' column shows the most common word in each cluster
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file containing your keywords"
    )
    
    if uploaded_file is not None:
        # Load and display the file
        df = load_csv(uploaded_file)
        
        st.subheader("📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Total rows: {len(df)}")
        
        # Column selection
        column_name = st.selectbox(
            "Select the column containing keywords",
            options=df.columns.tolist(),
            index=0
        )
        
        # Preview selected column
        st.write(f"**Sample keywords from '{column_name}':**")
        sample_keywords = df[column_name].dropna().head(5).tolist()
        st.write(", ".join([str(k) for k in sample_keywords]))
        
        # Cluster button
        if st.button("🚀 Start Clustering", type="primary"):
            with st.spinner("Processing..."):
                try:
                    result_df = cluster_keywords(
                        df=df,
                        column_name=column_name,
                        model_name=model_name,
                        min_similarity=min_similarity,
                        remove_dupes=remove_dupes,
                        device=device
                    )
                    
                    if result_df is not None:
                        # Store results in session state
                        st.session_state['result_df'] = result_df
                        st.session_state['chart_type'] = chart_type
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.exception(e)
    
    # Display results if available
    if 'result_df' in st.session_state:
        result_df = st.session_state['result_df']
        chart_type = st.session_state.get('chart_type', 'treemap')
        
        st.markdown("---")
        st.subheader("📈 Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_keywords = len(result_df)
        unique_clusters = result_df[result_df['spoke'] != 'no_cluster']['spoke'].nunique()
        unclustered = len(result_df[result_df['spoke'] == 'no_cluster'])
        clustered = total_keywords - unclustered
        
        col1.metric("Total Keywords", total_keywords)
        col2.metric("Unique Clusters", unique_clusters)
        col3.metric("Clustered Keywords", clustered)
        col4.metric("Unclustered", unclustered)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 Full Data", "📥 Download"])
        
        with tab1:
            fig = create_chart(result_df, chart_type)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Filter options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                show_unclustered = st.checkbox("Show unclustered keywords", value=True)
            with filter_col2:
                search_term = st.text_input("Search keywords", "")
            
            display_df = result_df.copy()
            
            if not show_unclustered:
                display_df = display_df[display_df['spoke'] != 'no_cluster']
            
            if search_term:
                display_df = display_df[
                    display_df['keyword'].str.contains(search_term, case=False, na=False) |
                    display_df['spoke'].str.contains(search_term, case=False, na=False) |
                    display_df['hub'].str.contains(search_term, case=False, na=False)
                ]
            
            st.dataframe(display_df, use_container_width=True, height=400)
        
        with tab3:
            # Prepare download (without cluster_size column)
            download_df = result_df.drop(columns=['cluster_size'], errors='ignore')
            csv_data = convert_df_to_csv(download_df)
            
            st.download_button(
                label="📥 Download Clustered Keywords (CSV)",
                data=csv_data,
                file_name="keywords_clustered.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            st.markdown("**Cluster Summary:**")
            
            # Show cluster summary
            summary_df = result_df[result_df['spoke'] != 'no_cluster'].groupby(['hub', 'spoke']).agg({
                'keyword': 'count'
            }).reset_index()
            summary_df.columns = ['Hub', 'Cluster', 'Keyword Count']
            summary_df = summary_df.sort_values('Keyword Count', ascending=False)
            
            st.dataframe(summary_df, use_container_width=True)


if __name__ == "__main__":
    main()
