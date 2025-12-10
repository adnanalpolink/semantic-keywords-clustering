# Semantic Keyword Clustering App

A Streamlit app that clusters keywords based on semantic similarity using SentenceTransformers and PolyFuzz.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then upload a CSV file containing your keywords and select the appropriate column.

## Features

- **Automatic encoding detection** for CSV files (UTF-8, UTF-16)
- **Three embedding models** to choose from (speed vs accuracy tradeoff)
- **Adjustable similarity threshold** for tighter or broader clusters
- **Interactive visualizations** (treemap or sunburst charts)
- **Downloadable results** as CSV
- **Search and filter** functionality

## Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Embedding Model | Controls clustering accuracy and speed | all-MiniLM-L6-v2 |
| Minimum Similarity | Higher = tighter clusters (0.5–0.99) | 0.85 |
| Remove Duplicates | Deduplicate keywords before clustering | Enabled |
| Chart Type | Treemap or sunburst visualization | Treemap |

## Output Columns

- **hub**: Most common word in the cluster (top-level grouping)
- **spoke**: Full cluster name
- **keyword**: Original keyword

## Credits

Based on the keyword clustering script by [Lee Foot](https://www.leefoot.com).
