#MoKiTo modules modified for ABM-data
#MoKiTo code source: https://github.com/donatiluca/MoKiTo




import numpy as np
import networkx as nx
import matplotlib.cm as cm

from sklearn_extra.cluster import CommonNNClustering
from sklearn.cluster import DBSCAN, HDBSCAN,  KMeans

from collections import Counter

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform #to optimize distance function
from sklearn.neighbors import NearestNeighbors #to build a sparse knn graph
import hdbscan
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler





## Modules to load data
def most_frequent(numbers):

    # remove numbers equal to -1
    filtered_numbers = [num for num in numbers if num != -1]
    # Count the frequency of each number in the list
    counts = Counter(filtered_numbers)
    # Find the number with the highest frequency
    most_common = counts.most_common(1)
    # Return the number with the highest frequency
    return most_common[0][0] if most_common else None

# def distance_matrix(X, metric='L2norm', R=None, periodic=False):
#     """
#     Create a symmetric distance matrix where each element (i,j) represents the 
#     L2norm between conformation i and conformation j.
#     """
#     num_conformations = len(X)
#     distance_matrix = np.zeros((num_conformations, num_conformations))
    
#     for i in range(num_conformations):
#         for j in range(i + 1, num_conformations):
#             if metric =='L2norm':
#                 value = np.linalg.norm(X[i] - X[j], axis = None)
        
#             distance_matrix[i, j] = value
#             distance_matrix[j, i] = value

#     return distance_matrix

def distance_matrix(X, metric='L2norm', R=None, periodic=False):
    #     """
#     Create a symmetric distance matrix where each element (i,j) represents the 
#     L2norm between conformation i and conformation j.
#     """
    
    if metric == 'L2norm':
        # pdist with 'euclidean' computes the L2 norm
        return squareform(pdist(X, metric='euclidean'))
    else:
        raise ValueError(f"Unsupported metric: {metric}")



def distance_matrix2(X, metric='L2norm', R=None, periodic=False):
    """
    Create a symmetric distance matrix where each element (i,j) represents the 
    RMSD between conformation i and conformation j.
    This function needs MDTRAJ
    """
    num_conformations = X.n_frames
    distance_matrix = np.zeros((num_conformations, num_conformations))
    
    for i in range(num_conformations):
        distance_matrix[i] = md.rmsd(X, X, i)

    return distance_matrix
    
###########################################################################################
## Class to store fundamental parameters
class OrganizeData:
    def __init__(self, X0, Xt, chi0, MDtraj=None):
        
        """
        OrganizeData(X0, Xt, chi0, chit, MDtraj, frame=0)
        """        


        self.N               = X0.shape[0]
        self.Ndims           = X0.shape[1]
        self.M               = Xt.shape[1]

        self.X0 = X0
        self.Xt = Xt        
        self.chi0 = chi0
        self.MDtraj = MDtraj

        print("Check shape of input data")
        print("X0.shape   = ", X0.shape)
        print("Xt.shape   = ", Xt.shape)
        print("chi0.shape = ", chi0.shape)

        print("  ")
        
###########################################################################################


class FindIntervals:
    def __init__(self, data, Nintervals=10, clustering='grid', random_state=None):
        """
        Creates intervals for chi function with reproducible results.
        
        Parameters:
        -----------
        data : object
            Data object with chi0, X0, N attributes
        Nintervals : int
            Number of intervals to create
        clustering : str
            'grid' or 'kmeans'
        random_state : int or None
            Random seed for reproducibility (affects kmeans)
        """
        
        N = data.N
        chi = np.copy(data.chi0).reshape(-1, 1)
            
        if clustering == 'grid':
            # Divide chi function in Nintervals
            chi_min = np.min(chi) - 0.0001
            chi_max = np.max(chi) + 0.0001
            chi_edges = np.linspace(chi_min, chi_max, Nintervals + 1)
            dchi = chi_edges[1] - chi_edges[0]
            chi_centers = chi_edges + 0.5 * dchi
            chi_centers = chi_centers[0:-1]
            
            chi_intervals = np.digitize(chi, chi_edges) - 1
            chi_intervals = chi_intervals[:, 0]
            labels_clusters, size_intervals = np.unique(chi_intervals, return_counts=True)
            
        elif clustering == 'kmeans':
            # **ADDED**: random_state for reproducibility
            KMintervals = KMeans(
                n_clusters=Nintervals,
                random_state=random_state,
                n_init=10  # Explicit n_init to avoid warnings
            ).fit(chi)
            
            chi_centers = np.copy(KMintervals.cluster_centers_[:, 0])
            chi_intervals = np.copy(KMintervals.labels_)
        
            # Sort centroids
            idx = np.argsort(chi_centers)
            chi_centers = chi_centers[idx]
            
            for nn, n in enumerate(idx):
                chi_intervals[KMintervals.labels_ == n] = nn
                
            labels_clusters, size_intervals = np.unique(chi_intervals, return_counts=True)
        
        # Store basic interval information
        self.chi_centers = chi_centers
        self.Nintervals = Nintervals
        self.chi_intervals = chi_intervals
        self.size_intervals = size_intervals
        
        # **OPTIMIZED**: Faster geometric density computation
        print("Computing geometric interval densities...")
        self.density_est = np.zeros(Nintervals)
        
        # Precompute interval indices (faster than repeated np.where)
        interval_indices = [np.where(self.chi_intervals == i)[0] for i in range(Nintervals)]
        
        for i in range(Nintervals):
            idx = interval_indices[i]
            Xi = data.X0[idx]
            
            if Xi.shape[0] > 10:  # Skip very small intervals
                try:
                    # **SPEED OPTIMIZATION 1**: Reduced k from 10 to 5
                    # Density estimation doesn't need many neighbors
                    k = min(5, Xi.shape[0] - 1)
                    
                    # **SPEED OPTIMIZATION 2**: Use 'auto' algorithm
                    # Lets sklearn choose fastest method (usually kd_tree for low-dim)
                    nn = NearestNeighbors(
                        n_neighbors=k,
                        algorithm='auto',  # Changed from 'ball_tree'
                        n_jobs=-1
                    )
                    nn.fit(Xi)
                    
                    # **SPEED OPTIMIZATION 3**: Sample large intervals
                    # For very large intervals, sample to speed up computation
                    if Xi.shape[0] > 10000:
                        # Sample 20% of points (min 1000, max 10000)
                        n_sample = min(max(int(Xi.shape[0] * 0.2), 1000), 10000)
                        if random_state is not None:
                            np.random.seed(random_state + i)
                        sample_idx = np.random.choice(Xi.shape[0], n_sample, replace=False)
                        distances, _ = nn.kneighbors(Xi[sample_idx])
                    else:
                        distances, _ = nn.kneighbors(Xi)
                    
                    # Use median of k-th nearest neighbor distances
                    mean_d = np.median(distances[:, -1])
                    self.density_est[i] = 1.0 / (mean_d + 1e-8)
                    
                except Exception as e:
                    print(f"Warning: Density calc failed for interval {i}: {e}")
                    self.density_est[i] = 0.0
            else:
                self.density_est[i] = 0.0

        print(f'unormalized density: {self.density_est}')

        # **IMPROVED**: Robust normalization using median
        valid_mask = self.density_est > 0
        if np.any(valid_mask):
            median_density = np.median(self.density_est[valid_mask])
            median_density = max(median_density, 1e-8)  # Avoid division by zero
            
            self.density_est[valid_mask] = self.density_est[valid_mask] / median_density
            
            # Assign small floor value to invalid intervals
            if np.any(~valid_mask):
                self.density_est[~valid_mask] = np.min(self.density_est[valid_mask]) * 0.1
        else:
            self.density_est[:] = 1.0

        # Clip to reasonable range (prevents extreme values)
        self.density_est = np.clip(self.density_est, 0.1, 5.0)

        # Compute global percentiles (25th/75th)
        self.d_low = np.percentile(self.density_est, 25)
        self.d_high = np.percentile(self.density_est, 75)
        
        print(f"Density estimates: min={np.min(self.density_est):.3f}, "
              f"median={np.median(self.density_est):.3f}, "
              f"max={np.max(self.density_est):.3f}")
        print(f"d_low (25th percentile)={self.d_low:.3f}, "
              f"d_high (75th percentile)={self.d_high:.3f}")

        
###########################################################################################

###########################################################################################
def assign_noisy_states(labels, di):
    svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')

    PWD_0_clutered = di[labels>-1]
    PWD_0_noisy    = di[labels==-1]
    
    svm_classifier.fit(PWD_0_clutered, labels[labels>-1])
    new_assignments = svm_classifier.predict(PWD_0_noisy).astype('int')

    labels[labels==-1]  = new_assignments

    return labels
    

# #Original version
# class FindNodes:
#     def __init__(self, data, FCs, eps = None, theta = None, algorithm = 'CNNC', metric=None, R=None, periodic=False):
        
#         """
#         Implements stage 3 of the MoKiTo pipeline: clustering and edge assignment.
#         """ 

#         nodes                   = np.ones(data.N)

#         # index of chi clusters per each node
#         index_chi_node          = []
        
#         # nodes per each chi cluster
#         nodes_for_clusters     = np.empty(FCs.Nintervals, dtype=object)

#         # Count the nodes
#         Nnodes = 0

#         # Loop over the intervals
#         for i in tqdm(range(FCs.Nintervals)):

#             if metric == 'L2norm':
                
#                 Xi = data.X0[FCs.chi_intervals == i]
#                 di    = distance_matrix(Xi)

#             elif metric == 'mdtraj_rmsd':

#                 Xi = data.MDtraj[FCs.chi_intervals == i]
#                 di = distance_matrix2(Xi)

#             # Run clustering
#             if algorithm == 'CNNC':
                
#                 clustering = CommonNNClustering(eps=eps[i], min_samples=theta[i], n_jobs=-1, metric='precomputed').fit(di) #
                    
#                 if np.sum(clustering.labels_==-1) > 0 and np.sum(clustering.labels_==-1)<len(clustering.labels_):
#                     if len(np.unique(clustering.labels_))==2:
#                         clustering.labels_[clustering.labels_==-1] = 0
#                     else:
#                         clustering.labels_ = assign_noisy_states(clustering.labels_,di)

#             elif algorithm == 'DBSCAN':
#                 clustering = DBSCAN(eps=eps, min_samples=theta[i], n_jobs=-1, metric='precomputed').fit(di)
#             elif algorithm == 'HDBSCAN':
#                 clustering = HDBSCAN(min_cluster_size=theta[i], metric='precomputed').fit(di) 
            
#             # Labels of nodes in cluster i
#             nodes_i               = clustering.labels_
#             nodes_i[nodes_i>-1]   = nodes_i[nodes_i>-1] + Nnodes

#             # Find the indeces of the states in the chi-cluster
#             chi_intervals_i    = np.where(FCs.chi_intervals == i)[0]
#             nodes[chi_intervals_i] = nodes_i
            
#             # Number of not noisy nodes in cluster i
#             unique_nodes_i               = np.unique(nodes_i[nodes_i>-1], return_counts=False)
#             unique_nodes_i               = unique_nodes_i + Nnodes
#             Nnodes_i                     = len(unique_nodes_i)

#             # total number of nodes
#             Nnodes = Nnodes + Nnodes_i

#             # assign to each node the interval i
#             for n in range(Nnodes_i):
#                 index_chi_node.append(i)
        
#         index_chi_node = np.asarray(index_chi_node)

#         _, nodes_size = np.unique(nodes[nodes>-1], return_counts=True)

#         self.Nnodes          = Nnodes
#         self.nodes           = nodes

#         self.nodes0          = nodes[0:data.N]
        
#         self.index_chi_node  = index_chi_node
#         self.nodes_size      = nodes_size


######################
class FindNodes:
    def __init__(self, data, FCs, eps=None, theta=None, algorithm='CNNC', 
                 metric='L2norm', R=None, periodic=False, 
                 adaptive_thresholds=None, random_state=None):
        """
        Implements stage 3 of the MoKiTo pipeline: clustering and edge assignment.
        Optimized version with better parameters but maintained speed.
        """
        nodes = np.ones(data.N, dtype=int) * -1  # initialize as noise
        index_chi_node = []
        Nnodes = 0

        # Precompute indices per interval
        interval_indices = [np.where(FCs.chi_intervals == i)[0] 
                           for i in range(FCs.Nintervals)]

        for i in tqdm(range(FCs.Nintervals), desc="Processing intervals"):
            chi_intervals_i = interval_indices[i]
            Xi = data.X0[chi_intervals_i] if metric == 'L2norm' else data.MDtraj[chi_intervals_i]

            n_points = Xi.shape[0]
            if n_points == 0:
                continue

            min_samples_i = theta[i] if theta is not None else 2
            if n_points < min_samples_i:
                continue
            
            if random_state is not None:
                np.random.seed(random_state + i)

            # -------------------
            # Algorithm-specific clustering
            # -------------------
            if algorithm == 'HDBSCAN':
                interval_size = n_points
                total_points = data.N
                interval_fraction = interval_size / total_points

                # 1. Get density estimate
                density_est = FCs.density_est[i]
                if density_est <= 0:
                    density_est = 1e-3

                # --- Adaptive strategy with fallback ---
                # Try reasonable parameters first, then relax if needed
                
                # Mild size-based scaling
                size_factor = np.clip(np.sqrt(interval_size / 60000), 0.90, 1.15)
                
                # Very mild density adjustment
                density_adjustment = np.clip(1.0 / density_est, 0.95, 1.05)
                
                scale_factor = size_factor * density_adjustment

                # --- START WITH MODERATE PARAMETERS ---
                base_min_cluster_size = 50
                base_min_samples = 8

                min_cluster_size = int(base_min_cluster_size * scale_factor)
                
                # Achievable bounds
                min_cluster_size = int(np.clip(
                    min_cluster_size,
                    35,  # Reasonable minimum
                    max(80, int(interval_size * 0.12))
                ))

                min_samples = int(np.clip(
                    base_min_samples * scale_factor,
                    5,
                    max(10, int(np.sqrt(min_cluster_size) * 0.9))
                ))

                # Epsilon
                epsilon = float(np.clip(1.05 * density_adjustment, 0.95, 1.20))

                # Selection method based on size
                if interval_size < 15000:
                    selection_method = 'leaf'
                else:
                    selection_method = 'eom'

                # --- FIRST ATTEMPT ---
                clustering = hdbscan.HDBSCAN(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=int(min_samples),
                    metric='euclidean',
                    allow_single_cluster=False,
                    cluster_selection_method=selection_method,
                    cluster_selection_epsilon=float(epsilon),
                    core_dist_n_jobs=-1
                ).fit(Xi)

                nodes_i = clustering.labels_
                n_clusters_initial = len(np.unique(nodes_i[nodes_i >= 0]))

                # --- FALLBACK: If no clusters found, retry with relaxed parameters ---
                if n_clusters_initial == 0:
                    print(f"  ⚠ No clusters found with min_cluster={min_cluster_size}. Retrying with relaxed parameters...")
                    
                    # CHANGED: Less aggressive relaxation (0.75 instead of 0.6)
                    # This prevents extreme fragmentation
                    min_cluster_size_relaxed = max(30, int(min_cluster_size * 0.80)) #change from 0.75, change from 0.85
                    min_samples_relaxed = max(5, int(min_samples * 0.75))
                    epsilon_relaxed = float(min(1.4, epsilon * 1.2))  # Less aggressive
                    
                    # CHANGED: Use 'eom' for large intervals even in fallback
                    fallback_method = 'eom' if interval_size > 30000 else 'leaf'
                    
                    clustering = hdbscan.HDBSCAN(
                        min_cluster_size=int(min_cluster_size_relaxed),
                        min_samples=int(min_samples_relaxed),
                        metric='euclidean',
                        allow_single_cluster=False,
                        cluster_selection_method=fallback_method,
                        cluster_selection_epsilon=float(epsilon_relaxed),
                        core_dist_n_jobs=-1
                    ).fit(Xi)
                    
                    nodes_i = clustering.labels_
                    min_cluster_size = min_cluster_size_relaxed
                    min_samples = min_samples_relaxed
                    epsilon = epsilon_relaxed
                    selection_method = f'{fallback_method} (fallback)'

                # --- Noise reassignment ---
                if np.sum(nodes_i == -1) > 0:
                    noise_mask = nodes_i == -1
                    clustered_mask = nodes_i >= 0

                    if np.sum(clustered_mask) > 0:
                        nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
                        nn.fit(Xi[clustered_mask])
                        distances, indices = nn.kneighbors(Xi[noise_mask])

                        noise_percentile = 92

                        noise_threshold = np.percentile(distances, noise_percentile)
                        assign_mask = distances.flatten() < noise_threshold

                        noise_labels = nodes_i[clustered_mask][indices.flatten()]
                        nodes_i[noise_mask] = np.where(assign_mask, noise_labels, -1)

                        reassigned_count = np.sum(assign_mask)
                        reassigned_fraction = 100 * reassigned_count / np.sum(noise_mask) if np.sum(noise_mask) > 0 else 0
                    else:
                        reassigned_fraction = 0.0
                else:
                    reassigned_fraction = 0.0

                # --- Enhanced logging ---
                n_clusters = len(np.unique(nodes_i[nodes_i >= 0]))
                n_noise = np.sum(nodes_i == -1)
                print(
                    f"Interval {i}: size={interval_size}, density={density_est:.3f}, "
                    f"min_cluster={min_cluster_size}, min_samples={min_samples}, "
                    f"eps={epsilon:.2f}, method={selection_method} "
                    f"→ {n_clusters} clusters, {n_noise} noise ({100*n_noise/interval_size:.1f}%), "
                    f"reassigned={reassigned_fraction:.1f}%"
                )

            elif algorithm == 'DBSCAN':
                k = max(min_samples_i, 2)
                k = min(k*2, n_points - 1)

                nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
                nn.fit(Xi)
                di_sparse = nn.kneighbors_graph(Xi, mode='distance')

                if eps is not None and i < len(eps):
                    eps_i = eps[i]
                else:
                    distances, _ = nn.kneighbors(Xi)
                    eps_i = np.percentile(distances[:, -1], 90)

                clustering = DBSCAN(
                    eps=eps_i,
                    min_samples=min_samples_i,
                    metric='precomputed',
                    n_jobs=-1
                ).fit(di_sparse)
                nodes_i = clustering.labels_

            elif algorithm == 'CNNC':
                k = max(min_samples_i, 2)
                k = min(k*2, n_points - 1)

                nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
                nn.fit(Xi)
                di_sparse = nn.kneighbors_graph(Xi, mode='distance')

                clustering = CommonNNClustering(
                    eps=eps[i], 
                    min_samples=min_samples_i, 
                    metric='precomputed'
                ).fit(di_sparse)
                nodes_i = clustering.labels_
                
                if np.sum(nodes_i == -1) > 0 and np.sum(nodes_i == -1) < len(nodes_i):
                    if len(np.unique(nodes_i)) == 2:
                        nodes_i[nodes_i == -1] = 0
                    else:
                        nodes_i = assign_noisy_states(nodes_i, di_sparse)

            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            # -------------------
            # Assign results with global offset
            # -------------------
            valid = nodes_i >= 0
            if np.any(valid):
                nodes_i[valid] += Nnodes
                nodes[chi_intervals_i] = nodes_i

                unique_nodes_i = np.unique(nodes_i[valid])
                Nnodes_i = len(unique_nodes_i)
                Nnodes += Nnodes_i
                index_chi_node.extend([i] * Nnodes_i)

        # -------------------
        # Finalize
        # -------------------
        index_chi_node = np.asarray(index_chi_node)
        valid_nodes = nodes[nodes > -1]
        if len(valid_nodes) > 0:
            _, nodes_size = np.unique(valid_nodes, return_counts=True)
        else:
            nodes_size = np.array([])

        self.Nnodes = Nnodes
        self.nodes = nodes
        self.nodes0 = nodes[0:data.N]
        self.index_chi_node = index_chi_node
        self.nodes_size = nodes_size


###########################################################################################
class BuildAdjacencyMatrix:
    def __init__(self, data, FNs, k = 5, C = 1, size_mlp = 100, threshold = 10, algorithm = 'mlp'):
        """
        Select a frame 
        Calculate distance between final point ij and all the starting points n
        Assign the node

        max number of neighbors : k
        """
        
        X0, Xt = data.X0, data.Xt
        C = np.zeros((FNs.Nnodes, FNs.Nnodes))

        if algorithm == 'knn':

            for i in tqdm(range(data.N)):
                m0 = int(FNs.nodes[i])
                for j in range(data.M):
                    
                    norm_ij_n = calculate_L2norm(X0, Xt[i,j], axis=(1)) #, axis=(1,2)
            
                    nearest_neighbors = np.argsort(norm_ij_n)[0:k]
                    
                    mt = int(most_frequent(FNs.nodes[nearest_neighbors]))
                    C[m0,mt] += 1 
                    C[mt,m0] += 1 

        elif algorithm == 'svm':
            
            svm_classifier = SVC(kernel='rbf', C=C, gamma='scale')
            svm_classifier.fit(X0, FNs.nodes)

            for i in tqdm(range(data.N)):
                m0 = int(FNs.nodes[i])
                for j in range(data.M):
                    mt = int(mlp_classifier.predict(data.Xt[i,j][np.newaxis,:]).item())

                    C[m0,mt] += 1 
                    C[mt,m0] += 1 

        elif algorithm == 'mlp':
            
            mlp_classifier = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(size_mlp), random_state=1)
            mlp_classifier.fit(X0, FNs.nodes)

            for i in tqdm(range(data.N)):
                m0 = int(FNs.nodes[i])
                for j in range(data.M):
                    mt = int(mlp_classifier.predict(data.Xt[i,j][np.newaxis,:]).item())

                    C[m0,mt] += 1 
                    C[mt,m0] += 1
    
        if threshold >0:
            C[C<threshold] = 0
        #
        A = (C > 0).astype(int)

        # direct adjacency matrix
        Ad = np.copy(A)
        Ad[np.triu_indices(Ad.shape[0], 0)] = 0

        # Counting matrix
        self.C = C

        # Adjacency matrix
        self.A  = A

        # Direct adjacency matrix
        self.Ad = Ad

        # Prob transition matrix
        P = np.copy(C)
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1
        P = P / row_sums[:, np.newaxis]
        self.P = P

        # Compute cluster centroids and pairwise RMSD
        from scipy.spatial.distance import cdist
        
        cluster_centroids = np.zeros((FNs.Nnodes, data.X0.shape[1]))
        for i in range(FNs.Nnodes):
            cluster_mask = FNs.nodes == i
            if np.sum(cluster_mask) > 0:
                cluster_centroids[i] = np.mean(data.X0[cluster_mask], axis=0)
        
        # Pairwise RMSD between cluster centers
        self.rmsd_nodes = cdist(cluster_centroids, cluster_centroids, metric='euclidean')
        self.cluster_centroids = cluster_centroids

        # Compute Adjacency matrix Weighted (weights for the graph)
        self.Aw = C
        
###########################################################################################
def ProjectFunctionOntoNodes(data, f0, FNs, ft=None, periodic = False):

    """
    # check dimensions
    print("Your function should be organize as:")
    print("f0.shape = (Npoints, )")
    #print("ft.shape = (Nframes, Npoints, Nfinpoints)")
    print("or")
    print("f0.shape = (Npoints, Nd)")
    #print("ft.shape = (Nframes, Npoints, Nfinpoints, Nd)")
    print("where Nd is the dimensionality of the function.")
    print("For example, if you are interested in a Ramachandran plot Nd = 2")
    print(" ")
    print("Let's see...")
    print("f0.shape = ", f0.shape)
    #print("ft.shape = ", ft.shape)
    """

    if f0.ndim == 1:
        f0 = f0[:,np.newaxis]
        Nd = 1
    else:
        Nd = f0.shape[1]
        

    f = f0
    
    f_nodes    =    np.zeros((FNs.Nnodes,Nd))
    
    for i in range(FNs.Nnodes):
    
        for d in range(Nd):
            if periodic == True:
                fx = np.cos(f[FNs.nodes==i,d])
                fy = np.sin(f[FNs.nodes==i,d])
                
                # Compute average vector
                fx_mean = np.mean(fx)
                fy_mean = np.mean(fy)
                
                # Compute the average angle from the average vector
                f_nodes[i,d] = np.arctan2(fy_mean, fx_mean)
            else:
                f_nodes[i,d] = np.mean(f[FNs.nodes==i,d])

    return f_nodes


        
##########################################################################################
class BuildGraph:
    def __init__(self, FNs, BAM):

        """
        Build both undirected and directed graphs with proper weights
        """

        Nnodes     = FNs.Nnodes
        C          = BAM.C
        P          = BAM.P
        
        #Direct graph
        Gd  = nx.from_numpy_array(BAM.Ad, create_using=nx.MultiDiGraph()) #in our case the 
        
        # Initialize the graph

        # Graph with all edges equal to 1
        G  = nx.Graph()
        # Graph with edges equal to number of counts
        GC = nx.Graph()
        # Graph with normalized esges
        GP = nx.Graph()
        
        for i in range(Nnodes):
            for j in range(i + 1, Nnodes):
                GC.add_edge(i, j,  weight=C[i, j])
                GP.add_edge(i, j,  weight=P[i, j])
                if C[i, j] != 0:
                    G.add_edge(i, j, weight=1)
        
        # Extract edges and their weights
        edges   = GP.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]        

        self.Gd      = Gd
        self.G       = G
        self.GC      = GC
        self.GP      = GP
        self.edges   = edges
        self.weights = weights


###########################################################################################
# class FindPaths:
#     def __init__(self, FNs, BG, BAM, cutoff=10):

#         Nnodes = FNs.Nnodes
#         G      = BG.G
#         Gd     = BG.Gd
#         Aw     = BAM.Aw
        
#         imin = 0
#         imax = Nnodes-1
#         #
        
#         #for p in nx.all_shortest_paths(Gd, source=imax, target=imin):
        
#         listPaths = []
        
#         k = 0
#         for p in nx.all_simple_paths(Gd, source=imax, target=imin, cutoff=cutoff):
#             listPaths.append(p)
#             k = k +1
        
#             if k==100:
#                 break
        
#         Npaths = len(listPaths)
#         print("Number of paths:", Npaths)
#         paths_weights = np.zeros(Npaths)
        
#         for i,p in enumerate(listPaths):
#             paths_weights[i] = np.sum([Aw[p[n], p[n+1]] for n in range(len(p)-1)])
            

#         # Sort paths according to the weight
#         indeces_paths = np.argsort( - paths_weights )

#         sortedListPaths = []
#         sortedPathsWeights = []  # NEW: Store sorted weights

#         for p in range(Npaths):
#             sortedListPaths.append(listPaths[indeces_paths[p]])
#             sortedPathsWeights.append(paths_weights[p])  # NEW

#         self.list_paths = sortedListPaths
#         self.paths_weights = np.array(sortedPathsWeights)  # NEW: Store as attribute

class FindPaths:
    def __init__(self, FNs, BG, BAM, cutoff=10, max_paths=100):

        Nnodes = FNs.Nnodes
        G      = BG.G
        Gd     = BG.Gd
        Aw     = BAM.Aw
        
        imin = 0
        imax = Nnodes - 1
        
        listPaths = []
        
        try:
            k = 0
            for p in nx.all_simple_paths(Gd, source=imax, target=imin, cutoff=cutoff):
                listPaths.append(p)
                k = k + 1
            
                if k == max_paths:
                    break
        except nx.NetworkXNoPath:
            print(f"Warning: No path found between node {imax} and node {imin}")
        
        Npaths = len(listPaths)
        print(f"Number of paths found: {Npaths}")
        
        if Npaths == 0:
            self.list_paths = []
            self.paths_weights = np.array([])
            self.paths_edge_weights = []
            return
        
        paths_weights = np.zeros(Npaths)
        paths_edge_weights = []  # NEW: Store edge weights for each path
        
        for i, p in enumerate(listPaths):
            # Calculate edge weights for this path
            edge_weights = [Aw[p[n], p[n+1]] for n in range(len(p)-1)]
            paths_edge_weights.append(edge_weights)
            
            # Total path weight
            paths_weights[i] = np.sum(edge_weights)

        # Sort paths according to the weight (descending)
        indices_paths = np.argsort(-paths_weights)

        sortedListPaths = []
        sortedPathsWeights = []
        sortedPathsEdgeWeights = []  # NEW
        
        for idx in indices_paths:
            sortedListPaths.append(listPaths[idx])
            sortedPathsWeights.append(paths_weights[idx])
            sortedPathsEdgeWeights.append(paths_edge_weights[idx])  # NEW

        self.list_paths = sortedListPaths
        self.paths_weights = np.array(sortedPathsWeights)
        self.paths_edge_weights = sortedPathsEdgeWeights  # NEW: List of lists
        
        # Print summary
        if Npaths > 0:
            print(f"Best path weight: {self.paths_weights[0]:.1f}")
            print(f"Best path: {self.list_paths[0]}")
            
# ###########################################################################################
# class CalculateEnergy:
#     def __init__(self, FNs, beta = 0.40): # kJ^-1

#         nodes_size = FNs.nodes_size
        
#         tot_size   = np.sum(nodes_size)
#         W          = nodes_size / tot_size
#         energy     = - 1 / beta * np.log(W)
#         energy     = energy - np.min(energy)

#         self.energy = energy

# class PlotGraph:
#     def __init__(self, FNs, BAM):

#         """
#         .....
#         """

