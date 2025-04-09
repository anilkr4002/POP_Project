#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

// Constants
#define MAXN 100005
#define INF 1000000000

// Structures
struct Edge {
    int u, v, w, id;
    bool inMST;
    
    // Add comparison operator for sorting
    bool operator<(const Edge& other) const {
        return w < other.w;
    }
};

// Helper functions
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Device functions for Union-Find
__device__ int find(int* parent, int x) {
    if (parent[x] == x) return x;
    parent[x] = find(parent, parent[x]);
    return parent[x];
}

__device__ bool unite(int* parent, int* size, int u, int v) {
    u = find(parent, u);
    v = find(parent, v);
    if (u == v) return false;
    if (size[u] < size[v]) {
        int temp = u;
        u = v;
        v = temp;
    }
    parent[v] = u;
    size[u] += size[v];
    return true;
}

// Initialize DSU array
__global__ void initDSUKernel(int* parent, int* size, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= n) {
        parent[idx] = idx;
        size[idx] = 1;
    }
}

// Build MST excluding a specific edge or forcing inclusion of a specific edge
__global__ void buildMSTKernel(Edge* edges, int m, int skipEdge, int forceEdge, 
                              int* parent, int* size, long long* MST_weight, bool* success) {
    *success = true;
    *MST_weight = 0;
    int edgesAdded = 0;
    
    // First add the forced edge if any
    if (forceEdge != -1) {
        Edge& e = edges[forceEdge];
        if (unite(parent, size, e.u, e.v)) {
            *MST_weight += e.w;
            edgesAdded++;
        } else {
            // This shouldn't happen, but just in case
            *success = false;
            return;
        }
    }
    
    // Then add remaining edges in ascending weight order
    for (int i = 0; i < m; i++) {
        if (i == skipEdge || i == forceEdge) continue;
        
        Edge& e = edges[i];
        if (unite(parent, size, e.u, e.v)) {
            *MST_weight += e.w;
            edgesAdded++;
        }
    }
    
    // Check if we have a valid MST (n-1 edges)
    if (edgesAdded != (blockDim.x - 1)) {
        *success = false;
    }
}

// Build initial MST to determine which edges are in the MST
__global__ void buildInitialMSTKernel(Edge* edges, int m, int n,
                                     int* parent, int* size, long long* MST_weight) {
    *MST_weight = 0;
    
    for (int i = 0; i < m; i++) {
        Edge& e = edges[i];
        if (unite(parent, size, e.u, e.v)) {
            e.inMST = true;
            *MST_weight += e.w;
        } else {
            e.inMST = false;
        }
    }
}

// Host function to build MST and calculate weight changes for each edge
void calculateDynamicMSTChanges(int n, int m, Edge* edges, long long* results) {
    // Device variables
    Edge* d_edges;
    int* d_parent;
    int* d_size;
    long long* d_MST_weight;
    bool* d_success;
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_edges, m * sizeof(Edge)), "Allocate edges");
    checkCudaError(cudaMalloc(&d_parent, (n+1) * sizeof(int)), "Allocate parent");
    checkCudaError(cudaMalloc(&d_size, (n+1) * sizeof(int)), "Allocate size");
    checkCudaError(cudaMalloc(&d_MST_weight, sizeof(long long)), "Allocate MST weight");
    checkCudaError(cudaMalloc(&d_success, sizeof(bool)), "Allocate success flag");
    
    // Sort edges by weight
    std::sort(edges, edges + m);
    
    // Initialize the original indices of edges after sorting
    for (int i = 0; i < m; i++) {
        edges[i].id = i;
    }
    
    // Copy edges to device
    checkCudaError(cudaMemcpy(d_edges, edges, m * sizeof(Edge), cudaMemcpyHostToDevice), "Copy edges");
    
    // Initialize DSU
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    initDSUKernel<<<numBlocks, blockSize>>>(d_parent, d_size, n);
    cudaDeviceSynchronize();
    
    // Build initial MST
    buildInitialMSTKernel<<<1, 1>>>(d_edges, m, n, d_parent, d_size, d_MST_weight);
    cudaDeviceSynchronize();
    
    // Copy back to get updated MST edges and weight
    long long MST_weight;
    checkCudaError(cudaMemcpy(edges, d_edges, m * sizeof(Edge), cudaMemcpyDeviceToHost), "Copy edges back");
    checkCudaError(cudaMemcpy(&MST_weight, d_MST_weight, sizeof(long long), cudaMemcpyDeviceToHost), "Copy MST weight");
    
    // Process each edge
    for (int i = 0; i < m; i++) {
        bool success;
        long long newWeight = 0;
        
        // Initialize DSU for this edge's calculation
        initDSUKernel<<<numBlocks, blockSize>>>(d_parent, d_size, n);
        cudaDeviceSynchronize();
        
        if (edges[i].inMST) {
            // Edge is in MST, skip it and build MST without it
            buildMSTKernel<<<1, n>>>(d_edges, m, i, -1, d_parent, d_size, d_MST_weight, d_success);
            cudaDeviceSynchronize();
            
            checkCudaError(cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost), "Copy success");
            checkCudaError(cudaMemcpy(&newWeight, d_MST_weight, sizeof(long long), cudaMemcpyDeviceToHost), "Copy new weight");
            
            results[edges[i].id] = success ? newWeight : -1;
        } else {
            // Edge is not in MST, force inclusion and build MST
            buildMSTKernel<<<1, n>>>(d_edges, m, -1, i, d_parent, d_size, d_MST_weight, d_success);
            cudaDeviceSynchronize();
            
            checkCudaError(cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost), "Copy success");
            checkCudaError(cudaMemcpy(&newWeight, d_MST_weight, sizeof(long long), cudaMemcpyDeviceToHost), "Copy new weight");
            
            results[edges[i].id] = success ? newWeight : -1;
        }
    }
    
    // Free device memory
    cudaFree(d_edges);
    cudaFree(d_parent);
    cudaFree(d_size);
    cudaFree(d_MST_weight);
    cudaFree(d_success);
}

int main() {
    int n, m;
    scanf("%d %d", &n, &m);
    
    // Read edges
    Edge* edges = (Edge*)malloc(m * sizeof(Edge));
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].w);
        edges[i].id = i;
        edges[i].inMST = false;
    }
    
    // Calculate dynamic MST changes
    long long* results = (long long*)malloc(m * sizeof(long long));
    calculateDynamicMSTChanges(n, m, edges, results);
    
    // Find original MST weight
    long long original_mst_weight = 0;
    for (int i = 0; i < m; i++) {
        if (edges[i].inMST) {
            original_mst_weight += edges[i].w;
        }
    }
    
    // Print results
    printf("Original MST weight: %lld\n", original_mst_weight);
    printf("\nDynamic MST Updates for Each Edge:\n");
    
    // Sort back to original order
    std::vector<std::pair<int, Edge>> sorted_edges;
    for (int i = 0; i < m; i++) {
        sorted_edges.push_back({edges[i].id, edges[i]});
    }
    
    // Define a custom comparison function for the std::pair<int, Edge>
    std::sort(sorted_edges.begin(), sorted_edges.end(), 
        [](const std::pair<int, Edge>& a, const std::pair<int, Edge>& b) {
            return a.first < b.first;
        });
    
    for (int i = 0; i < m; i++) {
        int id = sorted_edges[i].first;
        Edge& e = sorted_edges[i].second;
        
        printf("Edge %d: %d-%d (weight=%d) - ", id + 1, e.u, e.v, e.w);
        
        if (e.inMST) {
            printf("In MST, if removed: ");
            if (results[id] == -1) {
                printf("No MST possible\n");
            } else {
                printf("New MST weight = %lld\n", results[id]);
            }
        } else {
            printf("Not in MST, if added: ");
            if (results[id] == -1) {
                printf("No MST possible\n");
            } else {
                printf("New MST weight = %lld\n", results[id]);
            }
        }
    }
    
    // Clean up
    free(edges);
    free(results);
    
    return 0;
}
