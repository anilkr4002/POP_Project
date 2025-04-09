#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <vector>
#include <limits>
#include <sys/time.h>

const int MAXN = 100005;
const int LOG = 17;  // Adjust based on max tree depth (2^LOG >= n)
const int BLOCK_SIZE = 256;
const int SHARED_MEM_SIZE = 1024;  // Adjust based on GPU capabilities

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

// Structure to store replacement edge information
struct ReplacementInfo {
    int replacementEdgeIdx;
    ll newMSTWeight;
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

// Optimized DFS to build binary lifting arrays
void dfs(int u, int p, int w, int* up, int* maxEdge, int* depth, std::vector<std::pair<int, int>>* adj) {
    up[u * LOG + 0] = p;
    maxEdge[u * LOG + 0] = w;
    
    // Build binary lifting arrays
    for (int i = 1; i < LOG; i++) {
        int anc = up[u * LOG + (i-1)];
        if (anc == p && i > 1) {  // Optimization: skip unnecessary computations
            up[u * LOG + i] = p;
            maxEdge[u * LOG + i] = maxEdge[u * LOG + (i-1)];
        } else {
            up[u * LOG + i] = up[anc * LOG + (i-1)];
            maxEdge[u * LOG + i] = std::max(maxEdge[u * LOG + (i-1)], maxEdge[anc * LOG + (i-1)]);
        }
    }
    
    // Process adjacent nodes
    for (const auto& it : adj[u]) {
        int v = it.first;
        int wt = it.second;
        if (v != p) {
            depth[v] = depth[u] + 1;
            dfs(v, u, wt, up, maxEdge, depth, adj);
        }
    }
}

// Optimized CPU version of getMaxEdge
int getMaxEdge(int u, int v, const int* up, const int* maxEdge, const int* depth) {
    if (depth[u] < depth[v]) std::swap(u, v);
    int maxE = 0;
    
    // Bring nodes to same depth - bit manipulation optimization
    int diff = depth[u] - depth[v];
    for (int i = 0; diff > 0; i++) {
        if (diff & 1) {
            maxE = std::max(maxE, maxEdge[u * LOG + i]);
            u = up[u * LOG + i];
        }
        diff >>= 1;
    }
    
    if (u == v) return maxE;
    
    // Find LCA
    for (int i = LOG - 1; i >= 0; i--) {
        if (up[u * LOG + i] != up[v * LOG + i]) {
            maxE = std::max(maxE, std::max(maxEdge[u * LOG + i], maxEdge[v * LOG + i]));
            u = up[u * LOG + i];
            v = up[v * LOG + i];
        }
    }
    
    return std::max(maxE, std::max(maxEdge[u * LOG + 0], maxEdge[v * LOG + 0]));
}

// GPU device function with optimized bit manipulation
__device__ int d_getMaxEdge(int u, int v, const int* __restrict__ up, 
                           const int* __restrict__ maxEdge, 
                           const int* __restrict__ depth) {
    if (depth[u] < depth[v]) {
        int temp = u;
        u = v;
        v = temp;
    }
    int maxE = 0;
    
    // Bring nodes to same depth - bit manipulation optimization
    int diff = depth[u] - depth[v];
    for (int i = 0; diff > 0; i++) {
        if (diff & 1) {
            maxE = max(maxE, maxEdge[u * LOG + i]);
            u = up[u * LOG + i];
        }
        diff >>= 1;
    }
    
    if (u == v) return maxE;
    
    // Find LCA with binary lifting
    for (int i = LOG - 1; i >= 0; i--) {
        if (up[u * LOG + i] != up[v * LOG + i]) {
            maxE = max(maxE, max(maxEdge[u * LOG + i], maxEdge[v * LOG + i]));
            u = up[u * LOG + i];
            v = up[v * LOG + i];
        }
    }
    
    return max(maxE, max(maxEdge[u * LOG + 0], maxEdge[v * LOG + 0]));
}

// Improved kernel with better memory access patterns
__global__ void computeNonMSTEdgeReplacementKernel(
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
    if (idx >= m || inMST[idx]) return;
    
    // Process only non-MST edges
    int u = edgeU[idx];
    int v = edgeV[idx];
    int w = edgeW[idx];
    
    int maxE = d_getMaxEdge(u, v, up, maxEdge, depth);
    
    // Store result if we can improve MST
    if (maxE > w) {
        results[idx].newMSTWeight = originalMST - maxE + w;
        results[idx].replacementEdgeIdx = idx;
    } else {
        results[idx].newMSTWeight = originalMST;
        results[idx].replacementEdgeIdx = -1;
    }
}

// Optimized kernel with shared memory for MST edge replacements
__global__ void findMSTEdgeReplacementsKernel(
    const int* __restrict__ mstEdgeIndices,
    int numMSTEdges,
    const int* __restrict__ edgeU,
    const int* __restrict__ edgeV,
    const int* __restrict__ edgeW,
    const bool* __restrict__ inMST,
    const int* __restrict__ nonMSTIndices,
    int numNonMSTEdges,
    const int* __restrict__ up,
    const int* __restrict__ maxEdge,
    const int* __restrict__ depth,
    ReplacementInfo* __restrict__ results,
    ll originalMST
) {
    __shared__ int s_nonMSTU[SHARED_MEM_SIZE];
    __shared__ int s_nonMSTV[SHARED_MEM_SIZE];
    __shared__ int s_nonMSTW[SHARED_MEM_SIZE];
    __shared__ int s_nonMSTIdx[SHARED_MEM_SIZE];
    
    int mstIdx = blockIdx.x;
    int localThreadIdx = threadIdx.x;
    
    if (mstIdx >= numMSTEdges) return;
    
    int edgeIdx = mstEdgeIndices[mstIdx];
    int mstU = edgeU[edgeIdx];
    int mstV = edgeV[edgeIdx];
    int mstW = edgeW[edgeIdx];
    
    ll bestWeight = LLONG_MAX;
    int bestEdgeIdx = -1;
    
    // Process non-MST edges in chunks using shared memory
    for (int offset = 0; offset < numNonMSTEdges; offset += SHARED_MEM_SIZE) {
        int remaining = min(SHARED_MEM_SIZE, numNonMSTEdges - offset);
        
        // Cooperatively load non-MST edges into shared memory
        if (localThreadIdx < remaining) {
            int nonMSTIdx = nonMSTIndices[offset + localThreadIdx];
            s_nonMSTU[localThreadIdx] = edgeU[nonMSTIdx];
            s_nonMSTV[localThreadIdx] = edgeV[nonMSTIdx];
            s_nonMSTW[localThreadIdx] = edgeW[nonMSTIdx];
            s_nonMSTIdx[localThreadIdx] = nonMSTIdx;
        }
        
        __syncthreads();
        
        // Each thread processes some edges from shared memory
        for (int i = localThreadIdx; i < remaining; i += blockDim.x) {
            int u = s_nonMSTU[i];
            int v = s_nonMSTV[i];
            int w = s_nonMSTW[i];
            int nonMSTIdx = s_nonMSTIdx[i];
            
            // Check if this edge reconnects the MST components when mstEdge is removed
            int maxE = d_getMaxEdge(u, v, up, maxEdge, depth);
            if (maxE == mstW) {
                ll newWeight = originalMST - mstW + w;
                if (newWeight < bestWeight) {
                    bestWeight = newWeight;
                    bestEdgeIdx = nonMSTIdx;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Use warp-level reduction to find the best edge across threads
    #pragma unroll
    for (int mask = blockDim.x/2; mask > 0; mask >>= 1) {
        ll peerWeight = __shfl_down_sync(0xffffffff, bestWeight, mask);
        int peerIdx = __shfl_down_sync(0xffffffff, bestEdgeIdx, mask);
        
        if (peerWeight < bestWeight) {
            bestWeight = peerWeight;
            bestEdgeIdx = peerIdx;
        }
    }
    
    // Thread 0 writes the result
    if (localThreadIdx == 0) {
        results[edgeIdx].newMSTWeight = (bestWeight == LLONG_MAX) ? -1 : bestWeight;
        results[edgeIdx].replacementEdgeIdx = bestEdgeIdx;
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
    for (size_t i = 0; i < edges.size(); i++) {
        if ((edges[i].inMST && static_cast<int>(i) != removedEdgeIdx) || static_cast<int>(i) == addedEdgeIdx) {
            printf("(%d-%d, w=%d) ", edges[i].u, edges[i].v, edges[i].w);
        }
    }
    printf("\nNew MST Weight: %lld\n", newWeight);
}

// Error checking function
void CheckCuda(const char* msg, const int line) {
    cudaError_t e = cudaGetLastError();
    if (cudaSuccess != e) {
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n", e, msg, line, cudaGetErrorString(e));
        exit(-1);
    }
}

// Macro for easier error checking
#define CHECK_CUDA(msg) CheckCuda(msg, __LINE__)

int main() {
    // Read input
    int n, m;
    std::vector<Edge> edges;
    readInput(n, m, edges);
    
    CPUTimer overallTimer;
    overallTimer.start();
    
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
    
    // Compute binary lifting arrays with DFS
    CPUTimer preprocessTimer;
    preprocessTimer.start();
    dfs(1, 1, 0, up, maxEdge, depth, adj);
    double preprocessTime = preprocessTimer.stop();
    printf("Preprocessing time: %f seconds\n", preprocessTime);
    
    // Prepare data vectors for GPU
    std::vector<int> edgeU(m), edgeV(m), edgeW(m);
    std::vector<int> inMST(m);
    std::vector<int> mstEdgeIndices;
    std::vector<int> nonMSTEdgeIndices;
    
    for (int i = 0; i < m; i++) {
        edgeU[i] = edges[i].u;
        edgeV[i] = edges[i].v;
        edgeW[i] = edges[i].w;
        inMST[i] = edges[i].inMST;
        
        if (edges[i].inMST) {
            mstEdgeIndices.push_back(i);
        } else {
            nonMSTEdgeIndices.push_back(i);
        }
    }
    
    int numMSTEdges = mstEdgeIndices.size();
    int numNonMSTEdges = nonMSTEdgeIndices.size();
    
    // Allocate device memory
    int *d_edgeU, *d_edgeV, *d_edgeW;
    bool *d_inMST;
    int *d_up, *d_maxEdge, *d_depth;
    int *d_mstEdgeIndices, *d_nonMSTIndices;
    ReplacementInfo *d_results;
    
    cudaMalloc((void**)&d_edgeU, m * sizeof(int));
    cudaMalloc((void**)&d_edgeV, m * sizeof(int));
    cudaMalloc((void**)&d_edgeW, m * sizeof(int));
    cudaMalloc((void**)&d_inMST, m * sizeof(bool));
    cudaMalloc((void**)&d_up, (n+1) * LOG * sizeof(int));
    cudaMalloc((void**)&d_maxEdge, (n+1) * LOG * sizeof(int));
    cudaMalloc((void**)&d_depth, (n+1) * sizeof(int));
    cudaMalloc((void**)&d_results, m * sizeof(ReplacementInfo));
    cudaMalloc((void**)&d_mstEdgeIndices, numMSTEdges * sizeof(int));
    cudaMalloc((void**)&d_nonMSTIndices, numNonMSTEdges * sizeof(int));
    
    CHECK_CUDA("Memory allocation");
    
    // Copy data to device
    cudaMemcpy(d_edgeU, edgeU.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeV, edgeV.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeW, edgeW.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inMST, inMST.data(), m * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, up, (n+1) * LOG * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxEdge, maxEdge, (n+1) * LOG * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth, depth, (n+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mstEdgeIndices, mstEdgeIndices.data(), numMSTEdges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonMSTIndices, nonMSTEdgeIndices.data(), numNonMSTEdges * sizeof(int), cudaMemcpyHostToDevice);
    
    CHECK_CUDA("Memory copy to device");
    
    // Initialize results
    std::vector<ReplacementInfo> results(m);
    for (int i = 0; i < m; i++) {
        results[i].newMSTWeight = 0;
        results[i].replacementEdgeIdx = -1;
    }
    cudaMemcpy(d_results, results.data(), m * sizeof(ReplacementInfo), cudaMemcpyHostToDevice);
    
    // Launch kernels
    CPUTimer kernelTimer;
    kernelTimer.start();
    
    // Process non-MST edges
    int nonMSTBlocks = (numNonMSTEdges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeNonMSTEdgeReplacementKernel<<<nonMSTBlocks, BLOCK_SIZE>>>(
        d_edgeU, d_edgeV, d_edgeW, d_inMST, d_up, d_maxEdge, d_depth,
        d_results, m, MST_weight
    );
    CHECK_CUDA("Non-MST kernel");
    
    // Process MST edges - one block per MST edge with shared memory optimization
    findMSTEdgeReplacementsKernel<<<numMSTEdges, BLOCK_SIZE>>>(
        d_mstEdgeIndices, numMSTEdges, 
        d_edgeU, d_edgeV, d_edgeW, 
        d_inMST, d_nonMSTIndices, numNonMSTEdges,
        d_up, d_maxEdge, d_depth, 
        d_results, MST_weight
    );
    CHECK_CUDA("MST kernel");
    
    cudaDeviceSynchronize();
    double kernelTime = kernelTimer.stop();
    printf("CUDA kernel execution time: %f seconds\n", kernelTime);
    
    // Copy results back to host
    cudaMemcpy(results.data(), d_results, m * sizeof(ReplacementInfo), cudaMemcpyDeviceToHost);
    CHECK_CUDA("Memory copy from device");
    
    // Total execution time
    double totalTime = overallTimer.stop();
    printf("Total execution time: %f seconds\n", totalTime);
    
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
                if (replaceEdge >= 0) {
                    printf("  If removed, can be replaced by edge %d-%d (weight=%d)\n", 
                           edgeU[replaceEdge], edgeV[replaceEdge], edgeW[replaceEdge]);
                    describeNewMST(edges, i, replaceEdge, results[i].newMSTWeight);
                } else {
                    printf("  No valid replacement found\n");
                }
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
    cudaFree(d_nonMSTIndices);
    
    delete[] up;
    delete[] maxEdge;
    delete[] depth;
    
    return 0;
}
