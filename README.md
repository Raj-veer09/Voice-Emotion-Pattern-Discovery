# Voice Emotion Pattern Discovery

An unsupervised machine learning pipeline designed to discover and analyze underlying acoustic patterns in human emotional speech. This project extracts audio features from speech recordings, applies clustering algorithms to group similar vocal characteristics, and visualizes the high-dimensional feature space.

## Dataset

This project uses the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset, specifically the `Audio_Speech_Actors_01-24` subset.

- **Total Samples:** 1440 audio files (.wav)
- **Emotions Present:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised (represented across 8 target clusters).

## Pipeline & Methodology

1. **Feature Extraction:** We process the raw `.wav` files to extract 39 key acoustic features using `librosa`:
   - 13 MFCCs (Mel-Frequency Cepstral Coefficients)
   - 13 Delta coefficients (first-order derivatives)
   - 13 Delta-Delta coefficients (second-order derivatives)
2. **Feature Normalization:** Standardization of the 39-dimensional feature space to ensure algorithms treat all acoustic properties with equal weight.
3. **Clustering:**
   - **Gaussian Mixture Models (GMM):** Configured for 8 clusters (mirroring the 8 intended emotions in the RAVDESS dataset).
   - **DBSCAN:** Used for density-based spatial clustering to identify noise and non-globular cluster shapes.
4. **Dimensionality Reduction:** **t-SNE** (t-Distributed Stochastic Neighbor Embedding) is applied to reduce the 39-dimensional feature space into a 2D plane for visual interpretation.

## Project Structure

```text
├── data/
│   └── processed/
│       ├── audio_features.csv                 # Raw 39 extracted features
│       ├── normalized_features.csv            # Standardized features
│       └── audio_features_with_clusters_tsne.csv # Final dataset w/ clusters & 2D t-SNE coords
├── notebooks/
│   ├── 01_dataset_exploration.ipynb           # Initial RAVDESS data loading & EDA
│   ├── 02_feature_extraction.ipynb            # MFCC & Delta extraction logic
│   ├── 03_feature_normalization.ipynb         # Data scaling
│   ├── 04_clustering.ipynb                    # GMM and DBSCAN implementation
│   ├── 05_tsne_visualization.ipynb            # Dimensionality reduction
│   └── 06_interpretation_results.ipynb        # Final metric analysis
├── presentation/                              # Project slide deck
├── results/                                   # Exported visualizations and metrics
│   ├── cluster_emotion_distribution.png
│   ├── clustering_results.csv
│   ├── clustering_summary.txt
│   ├── tsne_dbscan_clusters.png
│   ├── tsne_emotion_visualization.png
│   └── tsne_gmm_clusters.png
└── requirements.txt                           # Dependency list
```

## Key Results & Observations

- **Clustering Performance:** The GMM algorithm achieved a Silhouette Score of **0.0248**.
- **Acoustic Overlap:** The relatively low silhouette score and visualizations (see `results/tsne_gmm_clusters.png`) indicate a high degree of overlap between clusters. This confirms the well-documented phenomenon that human emotional speech features are highly continuous and often share overlapping acoustic characteristics, rather than forming strictly isolated mathematical boundaries.

## Tech Stack & Dependencies

The project is built using Python 3. The primary libraries required are:

- **Audio Processing:** `librosa`, `soundfile`
- **Data Manipulation:** `numpy` (>=1.23), `pandas` (>=1.5)
- **Machine Learning:** `scikit-learn` (>=1.3), `scipy` (>=1.10)
- **Visualization:** `matplotlib` (>=3.7), `seaborn` (>=0.12)
- **Environment:** `jupyter` (>=1.0), `ipykernel` (>=6.25), `tqdm` (>=4.65)

## How to Run

1. Clone the repository and navigate to the root directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Place the extracted RAVDESS dataset in a `data/raw/` directory if you plan to re-run the extraction from scratch.
4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
5. Execute the notebooks sequentially from `01_dataset_exploration.ipynb` through `06_interpretation_results.ipynb`.

## Academic Context

**LAB:** Unsupervised Learning (USL)

**Group:** Manhattan Distance

**Members:**

1. Harshith Agarwal [BTECH/10721/23]
2. Rajveer Singh Chabda [BTECH/10726/23]
3. Raghava [BTECH/10667/22]
4. Rishi Talluri [BTECH/10829/23]
5. Venkat Saahit Kamu [BTECH/10904/23]
