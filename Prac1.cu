#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <algorithm>
#include <vector>
#include <limits>
#include <sys/time.h>

const int MAXN = 100005;
const int LOG = 17;
const int ThreadsPerBlock = 256;

typedef long long ll;

struct Edge {
    int u, v, w, id;
    bool inMST;
    
    bool operator<(const Edge& other) const {
        return w < other.w;
    }
};

struct CPUTimer {
    timeval beg, end;
    CPUTimer() {}
    ~CPUTimer() {}
    void start() {gettimeofday(&beg, NULL);}
    double stop() {gettimeofday(&end, NULL); return end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;}
};

// CPU functions
void readInput(int& n, int& m, std::vector<Edge>& edges) {
    scanf("%d %d", &n, &m);
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].w);
        edges[i].id = i;
        edges[i].inMST = false;
    }
}

void initDSU(int* parent, int* sz, int n) {
    for (int i = 0; i <= n; i++) {
        parent[i] = i;
        sz[i] = 1;
    }
}

int find(int x, int* parent) {
    if (parent[x] == x) return x;
    return parent[x] = find(parent[x], parent);
}

bool unite(int u, int v, int* parent, int* sz) {
    u = find(u, parent);
    v = find(v, parent);
    if (u == v) return false;
    if (sz[u] < sz[v]) std::swap(u, v);
    parent[v] = u;
    sz[u] += sz[v];
    return true;
}

ll buildMST(std::vector<Edge>& edges, int n, std::vector<std::pair<int, int>>* adj) {
    std::sort(edges.begin(), edges.end());
    
    int* parent = new int[n + 1];
    int* sz = new int[n + 1];
    initDSU(parent, sz, n);
    
    ll MST_weight = 0;
    for (auto& e : edges) {
        if (unite(e.u, e.v, parent, sz)) {
            e.inMST = true;
            MST_weight += e.w;
            adj[e.u].push_back({e.v, e.w});
            adj[e.v].push_back({e.u, e.w});
        }
    }
    
    delete[] parent;
    delete[] sz;
    return MST_weight;
}

void dfs(int u, int p, int w, int* up, int* maxEdge, int* depth, std::vector<std::pair<int, int>>* adj) {
    up[u * LOG + 0] = p;
    maxEdge[u * LOG + 0] = w;
    
    for (int i = 1; i < LOG; i++) {
        int anc = up[u * LOG + (i-1)];
        up[u * LOG + i] = up[anc * LOG + (i-1)];
        maxEdge[u * LOG + i] = std::max(maxEdge[u * LOG + (i-1)], maxEdge[anc * LOG + (i-1)]);
    }
    
    for (auto it : adj[u]) {
        int v = it.first;
        int wt = it.second;
        if (v != p) {
            depth[v] = depth[u] + 1;
            dfs(v, u, wt, up, maxEdge, depth, adj);
        }
    }
}

int getMaxEdge(int u, int v, int* up, int* maxEdge, int* depth) {
    if (depth[u] < depth[v]) std::swap(u, v);
    int maxE = 0;
    
    for (int i = LOG - 1; i >= 0; i--) {
        if (depth[u] - (1 << i) >= depth[v]) {
            maxE = std::max(maxE, maxEdge[u * LOG + i]);
            u = up[u * LOG + i];
        }
    }
    
    if (u == v) return maxE;
    
    for (int i = LOG - 1; i >= 0; i--) {
        if (up[u * LOG + i] != up[v * LOG + i]) {
            maxE = std::max(maxE, std::max(maxEdge[u * LOG + i], maxEdge[v * LOG + i]));
            u = up[u * LOG + i];
            v = up[v * LOG + i];
        }
    }
    
    return std::max(maxE, std::max(maxEdge[u * LOG + 0], maxEdge[v * LOG + 0]));
}

// CUDA kernels and device functions
__device__ int d_getMaxEdge(int u, int v, const int* __restrict__ up, const int* __restrict__ maxEdge, const int* __restrict__ depth) {
    if (depth[u] < depth[v]) {
        int temp = u;
        u = v;
        v = temp;
    }
    int maxE = 0;
    
    for (int i = LOG - 1; i >= 0; i--) {
        if (depth[u] - (1 << i) >= depth[v]) {
            maxE = max(maxE, maxEdge[u * LOG + i]);
            u = up[u * LOG + i];
        }
    }
    
    if (u == v) return maxE;
    
    for (int i = LOG - 1; i >= 0; i--) {
        if (up[u * LOG + i] != up[v * LOG + i]) {
            maxE = max(maxE, max(maxEdge[u * LOG + i], maxEdge[v * LOG + i]));
            u = up[u * LOG + i];
            v = up[v * LOG + i];
        }
    }
    
    return max(maxE, max(maxEdge[u * LOG + 0], maxEdge[v * LOG + 0]));
}

// Structure to store replacement edge information
struct ReplacementInfo {
    int replacementEdgeIdx;
    ll newMSTWeight;
};

// Kernel to compute the weight of MST if each edge in MST is replaced
__global__ void computeMSTEdgeReplacementKernel(
    const int* __restrict__ edgeU,
    const int* __restrict__ edgeV,
    const int* __restrict__ edgeW,
    const bool* __restrict__ inMST,
    const int* __restrict__ up,
    const int* __restrict__ maxEdge,
    const int* __restrict__ depth,
    ReplacementInfo* __restrict__ results,
    int m,
    ll originalMST
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m) return;
    
    if (!inMST[idx]) {
        // This edge is not in MST - check if it can improve MST
        int u = edgeU[idx];
        int v = edgeV[idx];
        int w = edgeW[idx];
        
        int maxE = d_getMaxEdge(u, v, up, maxEdge, depth);
        if (maxE > w) {
            // We can improve MST
            results[idx].newMSTWeight = originalMST - maxE + w;
            results[idx].replacementEdgeIdx = idx;
        } else {
            // No improvement
            results[idx].newMSTWeight = originalMST;
            results[idx].replacementEdgeIdx = -1;
        }
    }
}

// Kernel to find replacements for edges in MST
__global__ void findMSTEdgeReplacementsKernel(
    const int* __restrict__ mstEdgeIndices,
    int numMSTEdges,
    const int* __restrict__ edgeU,
    const int* __restrict__ edgeV,
    const int* __restrict__ edgeW,
    const bool* __restrict__ inMST,
    const int* __restrict__ up,
    const int* __restrict__ maxEdge,
    const int* __restrict__ depth,
    ReplacementInfo* __restrict__ results,
    int m,
    ll originalMST
) {
    int mstIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (mstIdx >= numMSTEdges) return;
    
    int edgeIdx = mstEdgeIndices[mstIdx];
    int mstU = edgeU[edgeIdx];
    int mstV = edgeV[edgeIdx];
    int mstW = edgeW[edgeIdx];
    
    ll bestWeight = LLONG_MAX;
    int bestEdgeIdx = -1;
    
    // Each thread examines all non-MST edges to find a replacement
    for (int i = 0; i < m; i++) {
        if (!inMST[i]) {
            int u = edgeU[i];
            int v = edgeV[i];
            int w = edgeW[i];
            
            // Check if this edge reconnects the components
            int maxE = d_getMaxEdge(u, v, up, maxEdge, depth);
            if (maxE == mstW) {
                ll newWeight = originalMST - mstW + w;
                if (newWeight < bestWeight) {
                    bestWeight = newWeight;
                    bestEdgeIdx = i;
                }
            }
        }
    }
    
    results[edgeIdx].newMSTWeight = (bestWeight == LLONG_MAX) ? -1 : bestWeight;
    results[edgeIdx].replacementEdgeIdx = bestEdgeIdx;
}

static void CheckCuda(const int line) {
    cudaError_t e;
    cudaDeviceSynchronize();
    if (cudaSuccess != (e = cudaGetLastError())) {
        fprintf(stderr, "CUDA error %d on line %d: %s\n", e, line, cudaGetErrorString(e));
        exit(-1);
    }
}

// Function to describe the MST by listing included edges
void describeMST(const std::vector<Edge>& edges, ll totalWeight) {
    printf("MST Edges: ");
    for (const auto& e : edges) {
        if (e.inMST) {
            printf("(%d-%d, w=%d) ", e.u, e.v, e.w);
        }
    }
    printf("\nTotal MST Weight: %lld\n", totalWeight);
}

// Function to describe new MST after edge replacement
void describeNewMST(const std::vector<Edge>& edges, int removedEdgeIdx, int addedEdgeIdx, ll newWeight) {
    printf("New MST Edges: ");
    for (int i = 0; i < edges.size(); i++) {
        if ((edges[i].inMST && i != removedEdgeIdx) || i == addedEdgeIdx) {
            printf("(%d-%d, w=%d) ", edges[i].u, edges[i].v, edges[i].w);
        }
    }
    printf("\nNew MST Weight: %lld\n", newWeight);
}

int main() {
    // Read input
    int n, m;
    std::vector<Edge> edges;
    readInput(n, m, edges);
    
    // Build MST on CPU
    std::vector<std::pair<int, int>> adj[MAXN];
    ll MST_weight = buildMST(edges, n, adj);
    printf("Original MST weight: %lld\n", MST_weight);
    
    // Describe original MST
    describeMST(edges, MST_weight);
    
    // Prepare binary lifting structures
    int* up = new int[n * LOG + 1]();
    int* maxEdge = new int[n * LOG + 1]();
    int* depth = new int[n + 1]();
    
    dfs(1, 1, 0, up, maxEdge, depth, adj);
    
    // Prepare CUDA memory
    int *d_edgeU, *d_edgeV, *d_edgeW;
    bool *d_inMST;
    int *d_up, *d_maxEdge, *d_depth;
    ReplacementInfo *d_results;
    
    // Allocate device memory
    cudaMalloc((void**)&d_edgeU, m * sizeof(int));
    cudaMalloc((void**)&d_edgeV, m * sizeof(int));
    cudaMalloc((void**)&d_edgeW, m * sizeof(int));
    cudaMalloc((void**)&d_inMST, m * sizeof(bool));
    cudaMalloc((void**)&d_up, (n+1) * LOG * sizeof(int));
    cudaMalloc((void**)&d_maxEdge, (n+1) * LOG * sizeof(int));
    cudaMalloc((void**)&d_depth, (n+1) * sizeof(int));
    cudaMalloc((void**)&d_results, m * sizeof(ReplacementInfo));
    
    // Prepare data
    std::vector<int> edgeU(m), edgeV(m), edgeW(m);
    std::vector<char> inMST(m);  // Using char instead of bool for contiguous memory
    std::vector<int> mstEdgeIndices;
    
    for (int i = 0; i < m; i++) {
        edgeU[i] = edges[i].u;
        edgeV[i] = edges[i].v;
        edgeW[i] = edges[i].w;
        inMST[i] = edges[i].inMST ? 1 : 0;
        
        if (edges[i].inMST) {
            mstEdgeIndices.push_back(i);
        }
    }
    
    // Copy data to device
    cudaMemcpy(d_edgeU, edgeU.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeV, edgeV.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeW, edgeW.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inMST, inMST.data(), m * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up, (n+1) * LOG * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxEdge, maxEdge, (n+1) * LOG * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth, depth, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
    
    // Initialize results
    std::vector<ReplacementInfo> results(m);
    for (int i = 0; i < m; i++) {
        results[i].newMSTWeight = 0;
        results[i].replacementEdgeIdx = -1;
    }
    cudaMemcpy(d_results, results.data(), m * sizeof(ReplacementInfo), cudaMemcpyHostToDevice);
    
    // Launch kernels
    CPUTimer timer;
    timer.start();
    
    // First handle non-MST edges
    int blocksNonMST = (m + ThreadsPerBlock - 1) / ThreadsPerBlock;
    computeMSTEdgeReplacementKernel<<<blocksNonMST, ThreadsPerBlock>>>(
        d_edgeU, d_edgeV, d_edgeW, d_inMST, d_up, d_maxEdge, d_depth,
        d_results, m, MST_weight
    );
    CheckCuda(__LINE__);
    
    // Then handle MST edges
    int numMSTEdges = mstEdgeIndices.size();
    int *d_mstEdgeIndices;
    cudaMalloc((void**)&d_mstEdgeIndices, numMSTEdges * sizeof(int));
    cudaMemcpy(d_mstEdgeIndices, mstEdgeIndices.data(), numMSTEdges * sizeof(int), cudaMemcpyHostToDevice);
    
    int blocksMST = (numMSTEdges + ThreadsPerBlock - 1) / ThreadsPerBlock;
    findMSTEdgeReplacementsKernel<<<blocksMST, ThreadsPerBlock>>>(
        d_mstEdgeIndices, numMSTEdges, d_edgeU, d_edgeV, d_edgeW, d_inMST,
        d_up, d_maxEdge, d_depth, d_results, m, MST_weight
    );
    CheckCuda(__LINE__);
    
    cudaDeviceSynchronize();
    double runtime = timer.stop();
    printf("CUDA execution time: %f seconds\n", runtime);
    
    // Get results back
    cudaMemcpy(results.data(), d_results, m * sizeof(ReplacementInfo), cudaMemcpyDeviceToHost);
    
    // Print detailed results
    printf("\n----- DETAILED MST EDGE ANALYSIS -----\n");
    for (int i = 0; i < m; i++) {
        if (inMST[i]) {
            printf("Edge %d-%d (in MST, weight=%d): ", edgeU[i], edgeV[i], edgeW[i]);
            if (results[i].newMSTWeight == -1) {
                printf("MST not possible if removed\n");
                printf("  This edge is critical - removing it disconnects the graph\n");
            } else {
                printf("New MST weight = %lld\n", results[i].newMSTWeight);
                int replaceEdge = results[i].replacementEdgeIdx;
                printf("  If removed, can be replaced by edge %d-%d (weight=%d)\n", 
                       edgeU[replaceEdge], edgeV[replaceEdge], edgeW[replaceEdge]);
                describeNewMST(edges, i, replaceEdge, results[i].newMSTWeight);
            }
        } else {
            printf("Edge %d-%d (not in MST, weight=%d): ", edgeU[i], edgeV[i], edgeW[i]);
            if (results[i].newMSTWeight == MST_weight || results[i].replacementEdgeIdx == -1) {
                printf("No improvement possible\n");
                printf("  Adding this edge would create a cycle with no benefit\n");
            } else {
                printf("New MST weight = %lld\n", results[i].newMSTWeight);
                int heavyEdge = results[i].replacementEdgeIdx;
                printf("  If added, would replace edge %d-%d (weight=%d)\n", 
                       edgeU[heavyEdge], edgeV[heavyEdge], edgeW[heavyEdge]);
                describeNewMST(edges, heavyEdge, i, results[i].newMSTWeight);
            }
        }
        printf("\n");
    }
    
    // Clean up
    cudaFree(d_edgeU);
    cudaFree(d_edgeV);
    cudaFree(d_edgeW);
    cudaFree(d_inMST);
    cudaFree(d_up);
    cudaFree(d_maxEdge);
    cudaFree(d_depth);
    cudaFree(d_results);
    cudaFree(d_mstEdgeIndices);
    
    delete[] up;
    delete[] maxEdge;
    delete[] depth;
    
    return 0;
}
