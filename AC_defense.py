import argparse
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, silhouette_score


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-layer Activation Clustering backdoor detector.')
    parser.add_argument('--clean-dir', default='features', help='Directory with clean feature .npy batches.')
    parser.add_argument('--backdoor-dir', default='features_backdoor', help='Directory with backdoor feature .npy batches.')
    parser.add_argument('--output-dir', default='MLAC_results', help='Directory where detection artifacts are written.')
    parser.add_argument('--layers', nargs='+', default=['layer1.0.conv1', 'layer2.0.conv2', 'layer4.1.conv2', 'linear'], help='Model layers to evaluate.')
    parser.add_argument('--num-batches', type=int, default=10, help='Maximum batches to load per layer.')
    parser.add_argument('--pca-dim', type=int, default=2, help='Number of PCA dimensions to retain for clustering.')
    parser.add_argument('--silhouette-threshold', type=float, default=0.4, help='Silhouette score threshold for flagging attacks.')
    parser.add_argument('--clean-template', default='{layer}_batch_{batch_idx}.npy', help='Filename template for clean feature batches.')
    parser.add_argument('--backdoor-template', default='{layer}_backdoor_batch_{batch_idx}.npy', help='Filename template for backdoor feature batches.')
    parser.add_argument('--visualize', action='store_true', help='Plot PCA scatter plots of the clusters.')
    return parser.parse_args()


def load_feature_batches(base_dir: Path, template: str, layer: str, num_batches: int):
    batches = []
    for batch_idx in range(num_batches):
        batch_path = base_dir / template.format(layer=layer, batch_idx=batch_idx)
        if batch_path.exists():
            batches.append(np.load(batch_path, allow_pickle=False))
    return batches


def run_detection(args):
    detection_results = {}
    detection_flag = False

    for layer in args.layers:
        print(f'Processing {layer}...')
        clean_batches = load_feature_batches(args.clean_dir, args.clean_template, layer, args.num_batches)
        backdoor_batches = load_feature_batches(args.backdoor_dir, args.backdoor_template, layer, args.num_batches)

        if not clean_batches or not backdoor_batches:
            print(f'No valid clean/backdoor batches for {layer}. Skipping this layer.')
            continue

        X_clean = np.vstack(clean_batches)
        X_backdoor = np.vstack(backdoor_batches)

        y_clean = np.zeros(X_clean.shape[0])
        y_backdoor = np.ones(X_backdoor.shape[0])
        y_true = np.concatenate((y_clean, y_backdoor))

        X = np.concatenate((X_clean, X_backdoor))
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        X = X - np.mean(X, axis=0)

        pca = PCA(n_components=args.pca_dim, whiten=True)
        X_pca = pca.fit_transform(X)

        kmeans = KMeans(n_clusters=2, random_state=0).fit(X_pca)
        silhouette = silhouette_score(X_pca, kmeans.labels_)
        print(f'Silhouette Score for {layer}: {silhouette:.4f}')

        cluster_labels = kmeans.labels_
        if np.mean(cluster_labels[y_true == 1]) < 0.5:
            cluster_labels = 1 - cluster_labels

        accuracy = accuracy_score(y_true, cluster_labels)
        attack_success_rate = np.mean(cluster_labels[y_true == 1] == 0)

        detected = silhouette > args.silhouette_threshold
        detection_results[layer] = {
            'silhouette_score': float(silhouette),
            'accuracy': float(accuracy),
            'attack_success_rate': float(attack_success_rate),
            'detected': bool(detected),
            'kmeans_labels': cluster_labels,
            'pca': pca,
        }

        detection_flag |= detected
        print(f'Accuracy for {layer}: {accuracy:.4f}')
        print(f'Attack Success Rate for {layer}: {attack_success_rate:.4f}')

    return detection_flag, detection_results


def visualize_results(args, detection_results):
    import matplotlib.pyplot as plt

    for layer, result in detection_results.items():
        print(f'Visualizing clustering for {layer}...')

        clean_batches = load_feature_batches(args.clean_dir, args.clean_template, layer, args.num_batches)
        backdoor_batches = load_feature_batches(args.backdoor_dir, args.backdoor_template, layer, args.num_batches)
        if not clean_batches or not backdoor_batches:
            print(f'No valid clean/backdoor batches for {layer}. Skipping visualization.')
            continue

        X_clean = np.vstack(clean_batches)
        X_backdoor = np.vstack(backdoor_batches)

        X = np.concatenate((X_clean, X_backdoor))
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        X_pca = result['pca'].transform(X)
        labels = result['kmeans_labels']
        n_clean = len(X_clean)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:n_clean, 0], X_pca[:n_clean, 1], c=labels[:n_clean], cmap='coolwarm', label='Clean', alpha=0.7)
        plt.scatter(X_pca[n_clean:, 0], X_pca[n_clean:, 1], c=labels[n_clean:], cmap='coolwarm', marker='x', label='Backdoor', alpha=0.7)
        plt.title(f'Clustering Visualization for {layer}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.colorbar(label='Cluster')
        plt.legend()
        plt.show()


def main():
    args = parse_args()
    args.clean_dir = Path(args.clean_dir)
    args.backdoor_dir = Path(args.backdoor_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detection_flag, detection_results = run_detection(args)

    if detection_flag:
        print('Attack detected in one or more layers!')
    else:
        print('No attacks detected across all layers.')

    results_path = args.output_dir / 'detection_results.npy'
    metrics_path = args.output_dir / 'detection_results_with_metrics.npy'
    np.save(results_path, detection_results)
    np.save(metrics_path, detection_results)
    print(f'Detection results saved to {results_path} and {metrics_path}.')

    if args.visualize:
        visualize_results(args, detection_results)


if __name__ == '__main__':
    main()
