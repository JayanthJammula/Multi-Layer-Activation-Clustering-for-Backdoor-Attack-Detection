import numpy as np
import matplotlib.pyplot as plt

# Load the detection results
results = np.load('./MLAC_results/detection_results.npy', allow_pickle=True).item()

# Function to visualize clustering for each layer
def visualize_layer(layer, result):
    # Extract PCA-transformed data and labels
    pca = result["pca"]
    kmeans_labels = result["kmeans_labels"]
    
    # Reconstruct PCA-transformed data if needed
    X_pca = pca.transform(np.concatenate((X_clean, X_backdoor)))  # Assuming X_clean and X_backdoor exist
    n_clean = len(X_clean)
    
    # Plot clean and backdoor data with cluster labels
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:n_clean, 0], X_pca[:n_clean, 1], c=kmeans_labels[:n_clean], cmap='coolwarm', label="Clean", alpha=0.7)
    plt.scatter(X_pca[n_clean:, 0], X_pca[n_clean:, 1], c=kmeans_labels[n_clean:], cmap='coolwarm', marker='x', label="Backdoor", alpha=0.7)
    plt.title(f"Clustering Visualization for {layer}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.legend()
    plt.show()

# Iterate over layers and visualize
for layer, result in results.items():
    print(f"Visualizing clustering for {layer}...")
    visualize_layer(layer, result)
