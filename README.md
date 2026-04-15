# NewVibe
A song recommendation script that uses unsupervised machine learning to find similar songs that match the audio profiles of tracks.

## Motivation
I am a big music listener, but earlier my playlist stopped expanding at the rate it used to. To find new music that alligns with my music and to further explore Unsupervised Machine Learning, I created NewVibe.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
* **Tools:** Jupyter Notebooks

## Methodology
1. **Feature Selection** I put all 13 numberical values into a correlation heatmap using Seaborn. I found a strong, positive correlation between Energy and Loudness. I also found a strong, inverse correlation between Energy and Acoustiness. Week audio features like Popularity are going to be excluded to avoid noise, leaving me with 8 features: [`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `tempo`, `valence`]

2. **Data Cleaning** Dropped all null value rows in dataset and make the track_name and artists lowercased for easier use in the future. 

3. **Data Normalization** Used StandardScaler to normalize all features, preventing any single feature to dominate the results

4. **Clustering** Recorded inertia of KMeans for values of k from 1-10. Plotted inertia scores and selected k=6 as the optimal number of clusters. Created KMeans model with k=6 and assigned a number 0-5 to each of the songs. 

5. **User Input** Taking a song name and a number of recommendations as inputs. Finds song's numerical values and cluster number from the dataset.

6. **Recommendation** Computing Euclidean distances between musical values of other songs in cluster with user's song values to order songs based on similarity. Returning the top songs based on the number of recomendations the user requested.


## Key Features
1. **Clustered Recommendation** Recommenations only come from songs that were sorted into similar clusters

2. **Euclidean Distances** Sorts songs in the cluster based on similarity to the user's song from most to least. Recommends based on that. 

3. **Flexibility** User gets to choose how many recommendations they want to recieve. 


## 🚀 How to Use

**Prerequisites**
- Python 3.8+

**Setup**
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/NewVibe.git
   cd NewVibe
   ```

2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the recommendation engine:
   ```bash
   python src/main.py
   ```

4. Follow the prompts:
   ```
   Enter the name of the song for recommendations: blinding lights
   How many recommendations do you want: 5
   ```

**Note:** The pre-processed and clustered data is already included in the `data/` folder, so no additional setup is required. The notebooks in `notebooks/` are available if you'd like to explore the data cleaning and model training process yourself.

## 📈 Future Improvements
Integrate the Spotify API for analyzing and recommending based on a user's playlist. Using API to train model on data from more songs. Allow the user to filter languages and to recommend songs based on artist rather than just song names. Using genres of music for training and recommendation as well.
