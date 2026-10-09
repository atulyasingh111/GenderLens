
# GenderLens
### Auditing Gender Bias in Pretrained Word Embeddings

## Problem Statement
GenderLens examines gender associations in occupation-related words using pretrained Word2Vec embeddings and compares associations before and after gender-direction neutralization.

## Objectives
- Measure gender associations in occupation words.
- Apply gender-direction neutralization.
- Compare association scores before and after neutralization.
- Examine statistical results and semantic preservation.

## Technologies
- Python
- Gensim and Word2Vec
- NumPy and Pandas
- Matplotlib
- Streamlit
- Docker and Docker Compose

## Project Pipeline
1. Calculate baseline gender associations.
2. Apply gender-direction neutralization.
3. Calculate statistical results.
4. Evaluate semantic preservation.
5. Generate before-and-after comparison plots.
6. Display results through a Streamlit dashboard.

## Setup and Execution

### Prerequisites
Install Git and Docker Desktop. Ensure Docker Desktop is running.

### Clone the repository
```bash
git clone https://github.com/atulyasingh111/GenderLens.git
cd GenderLens
```

### Run the project
```bash
docker compose up --build
```

Open the dashboard at:
http://localhost:8501

### Stop the project
Press `Ctrl+C`, then run:
```bash
docker compose down
```

## Outputs
The pipeline produces before-and-after association results, a comparison plot, and semantic preservation results.

## Limitations
- The analysis uses one pretrained embedding model.
- Results depend on the selected gender attribute words.
- Association does not establish causation.
- Neutralization does not guarantee the complete removal of gender-related information.

## Reproducibility
Docker Compose defines the analysis and dashboard services. The pretrained model may need to be downloaded on the first run.

## License
See the LICENSE file.
