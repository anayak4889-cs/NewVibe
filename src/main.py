import pandas as pd
import time

#getting the clustered dataframe based on path
df = pd.read_csv(r"C:\Users\saura\IdeaProjects\CSA\out\production\CSA\Song Recomender\data\clustered_data.csv")

#getting inputs from user
song = input("Enter the name of the song for recommendations: ").lower()
recs = int(input("How many recomendations do you want: "))

#importing scaler 
from sklearn.preprocessing import StandardScaler

#importing euclidean_distances to find more accurate song recomendation
from sklearn.metrics.pairwise import euclidean_distances

#function to get recommendations for a song
def song_recommendation(song, num_recs=3, df = df):

    #asthetic print statement
    print(f"Finding recommendations for {song}...")
    time.sleep(2)

    #finding the row number based on the name of the user's song
    row = df[df['track_name'] == song]

    #keeps asking for a new song until the user enters one that is in the dataset
    while row.empty:
        print(f"{song} not found in dataset. Please enter a valid song name.")
        song = input("Enter the name of the song for recommendations: ").lower()
        row = df[df['track_name'] == song]

    #getting the song data, creating a temp without the artist, track_name, and clusters so that euclidean distances can be calculated
    song_data = row.iloc[0]
    song_temp = song_data.drop(labels=['artists', 'track_name', 'cluster_label']).values.reshape(1,-1)
    
    #finding the cluster that the user's song is in
    song_cluster = song_data['cluster_label']

    #getting all other songs in that same cluster except for the user's song, creating a temp without artists, track_name, and clusters so that distances can be calculated
    cluster_songs = df[(df['cluster_label'] == song_cluster) & (df['track_name'] != song)].copy()
    cluster_temp = cluster_songs.drop(columns = ['artists', 'track_name', 'cluster_label'])

    #calculating distances and recomending the number of songs
    cluster_songs['distance'] = euclidean_distances(song_temp, cluster_temp.values)[0]
    recomendations = cluster_songs.sort_values('distance').head(num_recs)

    #printing out the songs
    for row in range(len(recomendations)):
        print(f"{row+1}) {recomendations.iloc[row]['track_name']} by {recomendations.iloc[row]['artists']}")
        print()

#calling function
song_recommendation(song, recs)