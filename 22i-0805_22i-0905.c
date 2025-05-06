#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <metis.h>

#define INFINITY 1e9
#define MAX_CHANGES 100
#define ASYNC_LEVEL 2 


typedef struct {
    int u, v, w;
} Edge;


typedef struct {
    int V, E;
    int *xadj;
    int *adjncy;
    int *adjwgt;
    int *part; 
} Graph;

// Priority queue node (for initial Dijkstra)
typedef struct {
    int vertex, dist;
} PQNode;

// Priority queue (heap-based for efficiency)
typedef struct {
    PQNode *nodes;
    int size, capacity;
} PriorityQueue;

// SSSP tree structure
typedef struct {
    int *dist;    
    int *parent;  
    char *affected; 
    char *affected_del;
} SSSPTree;

void pq_init(PriorityQueue *pq, int capacity) {
    pq->nodes = (PQNode *)malloc(sizeof(PQNode) * (capacity + 1));
    pq->size = 0;
    pq->capacity = capacity;
}

void pq_swap(PQNode *a, PQNode *b) {
    PQNode temp = *a;
    *a = *b;
    *b = temp;
}

void pq_heapify_up(PriorityQueue *pq, int idx) {
    while (idx > 1 && pq->nodes[idx].dist < pq->nodes[idx / 2].dist) {
        pq_swap(&pq->nodes[idx], &pq->nodes[idx / 2]);
        idx /= 2;
    }
}

void pq_heapify_down(PriorityQueue *pq, int idx) {
    int min_idx = idx;
    int left = 2 * idx;
    int right = 2 * idx + 1;
    if (left <= pq->size && pq->nodes[left].dist < pq->nodes[min_idx].dist)
        min_idx = left;
    if (right <= pq->size && pq->nodes[right].dist < pq->nodes[min_idx].dist)
        min_idx = right;
    if (min_idx != idx) {
        pq_swap(&pq->nodes[idx], &pq->nodes[min_idx]);
        pq_heapify_down(pq, min_idx);
    }
}

void pq_push(PriorityQueue *pq, int vertex, int dist) {
    if (pq->size >= pq->capacity) return;
    pq->size++;
    pq->nodes[pq->size] = (PQNode){vertex, dist};
    pq_heapify_up(pq, pq->size);
}

int pq_pop(PriorityQueue *pq, int *vertex, int *dist) {
    if (pq->size == 0) return 0;
    *vertex = pq->nodes[1].vertex;
    *dist = pq->nodes[1].dist;
    pq->nodes[1] = pq->nodes[pq->size--];
    pq_heapify_down(pq, 1);
    return 1;
}

void free_graph(Graph *g) {
    free(g->xadj);
    free(g->adjncy);
    free(g->adjwgt);
    free(g->part);
}

void free_sssp_tree(SSSPTree *t, int V) {
    free(t->dist);
    free(t->parent);
    free(t->affected);
    free(t->affected_del);
}

Graph load_graph(const char *filename, int rank, int size) {
    Graph g = {0};
    FILE *f = NULL;
    int edge_count = 0;

    // Read graph size on rank 0
    if (rank == 0) {
        f = fopen(filename, "r");
        if (!f) {
            perror("Error opening file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (fscanf(f, "%d %d", &g.V, &g.E) != 2) {
            printf("Error reading graph size.\n");
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (g.V <= 0 || g.E <= 0) {
            printf("Invalid graph size: V=%d, E=%d\n", g.V, g.E);
            fclose(f);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        edge_count = 2 * g.E; // Undirected graph
    }

    // Broadcast graph size
    MPI_Bcast(&g.V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g.E, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&edge_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate edges array
    Edge *edges = malloc(sizeof(Edge) * edge_count);
    if (!edges) {
        printf("Memory allocation failed for edges.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read edges on rank 0
    int *degree = calloc(g.V, sizeof(int));
    if (rank == 0) {
        for (int i = 0; i < g.E; i++) {
            int u, v, w;
            if (fscanf(f, "%d %d %d", &u, &v, &w) != 3) {
                printf("Invalid edge format at line %d.\n", i + 2);
                fclose(f);
                free(degree);
                free(edges);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (u < 0 || u >= g.V || v < 0 || v >= g.V || w <= 0) {
                printf("Invalid edge data: u=%d, v=%d, w=%d\n", u, v, w);
                fclose(f);
                free(degree);
                free(edges);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            edges[2 * i] = (Edge){u, v, w};
            edges[2 * i + 1] = (Edge){v, u, w};
            degree[u]++;
            degree[v]++;
        }
        fclose(f);

        // Debug: Print edges
        printf("Graph edges:\n");
        for (int i = 0; i < edge_count; i++) {
            printf("Edge %d -> %d, weight=%d\n", edges[i].u, edges[i].v, edges[i].w);
        }
    }

    // Broadcast edges
    MPI_Bcast(edges, edge_count * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Build CSR
    g.xadj = malloc(sizeof(int) * (g.V + 1));
    if (!g.xadj) {
        printf("Memory allocation failed for xadj.\n");
        free(degree);
        free(edges);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    g.xadj[0] = 0;
    for (int i = 0; i < g.V; i++)
        g.xadj[i + 1] = g.xadj[i] + degree[i];

    int *offset = malloc(sizeof(int) * (g.V + 1));
    if (!offset) {
        printf("Memory allocation failed for offset.\n");
        free(degree);
        free(edges);
        free(g.xadj);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memcpy(offset, g.xadj, sizeof(int) * (g.V + 1));

    g.adjncy = malloc(sizeof(int) * edge_count);
    g.adjwgt = malloc(sizeof(int) * edge_count);
    if (!g.adjncy || !g.adjwgt) {
        printf("Memory allocation failed for adjncy/adjwgt.\n");
        free(degree);
        free(edges);
        free(g.xadj);
        free(offset);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < edge_count; i++) {
        int u = edges[i].u;
        int idx = offset[u]++;
        g.adjncy[idx] = edges[i].v;
        g.adjwgt[idx] = edges[i].w;
    }

    g.E = edge_count / 2;

    // Check connectivity from vertex 0
    if (rank == 0 && degree[0] == 0) {
        printf("Warning: Source vertex 0 has no outgoing edges. Graph may be disconnected.\n");
    }

    // Partition graph using METIS
    g.part = malloc(sizeof(int) * g.V);
    if (!g.part) {
        printf("Memory allocation failed for part.\n");
        free(degree);
        free(edges);
        free(g.xadj);
        free(offset);
        free(g.adjncy);
        free(g.adjwgt);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    idx_t nparts = size;
    idx_t ncon = 1;
    idx_t objval;
    idx_t *vwgt = NULL, *vsize = NULL;
    idx_t *adjwgt = g.adjwgt;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_NCUTS] = 1;
    options[METIS_OPTION_NITER] = 10;

    if (rank == 0) {
        int ret = METIS_PartGraphKway(&g.V, &ncon, g.xadj, g.adjncy, vwgt, vsize, adjwgt,
                                      &nparts, NULL, NULL, options, &objval, g.part);
        if (ret != METIS_OK) {
            printf("METIS partitioning failed.\n");
            free(degree);
            free(edges);
            free(g.xadj);
            free(offset);
            free(g.adjncy);
            free(g.adjwgt);
            free(g.part);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("METIS partitioning completed with edge-cut: %d\n", (int)objval);
    }
    MPI_Bcast(g.part, g.V, MPI_INT, 0, MPI_COMM_WORLD);

    free(degree);
    free(offset);
    free(edges);
    return g;
}

void dijkstra(Graph *g, int src, SSSPTree *t, int rank, int size) {
    PriorityQueue pq;
    pq_init(&pq, g->V);
    char *visited = calloc(g->V, sizeof(char));

    // Initialize SSSP tree
    for (int i = 0; i < g->V; i++) {
        t->dist[i] = INFINITY;
        t->parent[i] = -1;
        t->affected[i] = 0;
        t->affected_del[i] = 0;
    }
    t->dist[src] = 0;
    pq_push(&pq, src, 0);

    // Process vertices
    while (pq.size > 0) {
        int u, d;
        if (!pq_pop(&pq, &u, &d)) break;
        if (visited[u]) continue;
        visited[u] = 1;

        // Debug: Print processed vertex
        if (rank == 0) printf("Processing vertex %d, dist=%d\n", u, d);

        // Process all vertices (fallback if partitioning fails)
        // Comment out the following line to use partitioning
        // if (g->part[u] != rank) continue;

        #pragma omp parallel for schedule(dynamic)
        for (int i = g->xadj[u]; i < g->xadj[u + 1]; i++) {
            int v = g->adjncy[i], w = g->adjwgt[i];
            if (!visited[v] && t->dist[u] + w < t->dist[v]) {
                #pragma omp critical
                {
                    if (t->dist[u] + w < t->dist[v]) {
                        t->dist[v] = t->dist[u] + w;
                        t->parent[v] = u;
                        pq_push(&pq, v, t->dist[v]);
                    }
                }
            }
        }
    }

    // Synchronize distances and parents
    MPI_Allreduce(MPI_IN_PLACE, t->dist, g->V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    for (int i = 0; i < g->V; i++) {
        if (t->dist[i] == INFINITY) t->parent[i] = -1;
    }

    // Check if any vertices were reached
    int reachable = 0;
    for (int i = 0; i < g->V; i++) {
        if (t->dist[i] < INFINITY) reachable++;
    }
    if (rank == 0 && reachable <= 1) {
        printf("Warning: Only %d vertex reachable from source. Check graph connectivity.\n", reachable);
    }

    free(visited);
    free(pq.nodes);
}

// Process edge changes (Algorithm 2 from the paper)
void process_changes(Graph *g, SSSPTree *t, Edge *changes, int num_changes, int rank) {
    // Initialize affected flags
    #pragma omp parallel for
    for (int i = 0; i < g->V; i++) {
        t->affected[i] = 0;
        t->affected_del[i] = 0;
    }

    // Process deletions and insertions in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_changes; i++) {
        int u = changes[i].u, v = changes[i].v, w = changes[i].w;
        if (w < 0) { // Deletion
            if ((t->parent[v] == u && t->dist[v] == t->dist[u] + w) ||
                (t->parent[u] == v && t->dist[u] == t->dist[v] + w)) {
                int y = t->dist[u] > t->dist[v] ? u : v;
                #pragma omp critical
                {
                    t->dist[y] = INFINITY;
                    t->parent[y] = -1;
                    t->affected_del[y] = 1;
                    t->affected[y] = 1;
                }
            }
        } else { // Insertion
            int x = t->dist[u] < t->dist[v] ? u : v;
            int y = t->dist[u] < t->dist[v] ? v : u;
            if (t->dist[y] > t->dist[x] + w) {
                #pragma omp critical
                {
                    t->dist[y] = t->dist[x] + w;
                    t->parent[y] = x;
                    t->affected[y] = 1;
                }
            }
        }
    }
}

// Asynchronous update of SSSP tree (Algorithm 4 from the paper)
void async_update(Graph *g, SSSPTree *t, int rank) {
    char change = 1;
    while (change) {
        change = 0;
        // Process deletion-affected vertices
        #pragma omp parallel for schedule(dynamic) reduction(|:change)
        for (int v = 0; v < g->V; v++) {
            if (g->part[v] != rank || !t->affected_del[v]) continue;
            int level = 0;
            int *queue = malloc(sizeof(int) * g->V);
            int q_size = 0, q_start = 0;
            queue[q_size++] = v;
            while (q_start < q_size && level < ASYNC_LEVEL) {
                int x = queue[q_start++];
                for (int i = g->xadj[x]; i < g->xadj[x + 1]; i++) {
                    int c = g->adjncy[i];
                    if (t->parent[c] == x) {
                        t->dist[c] = INFINITY;
                        t->parent[c] = -1;
                        t->affected[c] = 1;
                        t->affected_del[c] = 1;
                        change |= 1;
                        if (level < ASYNC_LEVEL - 1)
                            queue[q_size++] = c;
                    }
                }
                level++;
            }
            free(queue);
        }

        // Synchronize affected flags and distances
        MPI_Allreduce(MPI_IN_PLACE, t->affected_del, g->V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, t->affected, g->V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, t->dist, g->V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Process affected vertices
        #pragma omp parallel for schedule(dynamic) reduction(|:change)
        for (int v = 0; v < g->V; v++) {
            if (g->part[v] != rank || !t->affected[v]) continue;
            t->affected[v] = 0;
            int level = 0;
            int *queue = malloc(sizeof(int) * g->V);
            int q_size = 0, q_start = 0;
            queue[q_size++] = v;
            while (q_start < q_size && level < ASYNC_LEVEL) {
                int x = queue[q_start++];
                for (int i = g->xadj[x]; i < g->xadj[x + 1]; i++) {
                    int n = g->adjncy[i];
                    int w = g->adjwgt[i];
                    if (t->dist[x] > t->dist[n] + w) {
                        t->dist[x] = t->dist[n] + w;
                        t->parent[x] = n;
                        t->affected[x] = 1;
                        change |= 1;
                        if (level < ASYNC_LEVEL - 1)
                            queue[q_size++] = x;
                    }
                    if (t->dist[n] > t->dist[x] + w) {
                        t->dist[n] = t->dist[x] + w;
                        t->parent[n] = x;
                        t->affected[n] = 1;
                        change |= 1;
                        if (level < ASYNC_LEVEL - 1)
                            queue[q_size++] = n;
                    }
                }
                level++;
            }
            free(queue);
        }

        // Synchronize changes
        MPI_Allreduce(MPI_IN_PLACE, t->affected, g->V, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, t->dist, g->V, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &change, 1, MPI_CHAR, MPI_LOR, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <graph_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    double start_time = MPI_Wtime();
    Graph g = load_graph(argv[1], rank, size);
    SSSPTree t;
    t.dist = malloc(sizeof(int) * g.V);
    t.parent = malloc(sizeof(int) * g.V);
    t.affected = calloc(g.V, sizeof(char));
    t.affected_del = calloc(g.V, sizeof(char));

    // Initial SSSP
    if (rank == 0) printf("Computing initial SSSP from node 0...\n");
    dijkstra(&g, 0, &t, rank, size);
    if (rank == 0) {
        for (int i = 0; i < g.V; i++)
            printf("Vertex %d: Dist=%d, Parent=%d\n", i, t.dist[i], t.parent[i]);
    }

    // Define edge changes
    Edge changes[MAX_CHANGES] = {
        {0, 2, 5},  // Insertion
        {1, 2, 7},  // Insertion
        {2, 3, -1}, // Deletion
        {3, 4, 3}   // Insertion
    };
    int num_changes = 4;

    // Broadcast changes
    MPI_Bcast(changes, num_changes * sizeof(Edge), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Update SSSP
    if (rank == 0) printf("Updating SSSP...\n");
    process_changes(&g, &t, changes, num_changes, rank);
    async_update(&g, &t, rank);
    if (rank == 0) {
        for (int i = 0; i < g.V; i++)
            printf("Vertex %d: Dist=%d, Parent=%d\n", i, t.dist[i], t.parent[i]);
    }

    // Scalability analysis
    double end_time = MPI_Wtime();
    if (rank == 0) {
        printf("Total execution time: %f seconds\n", end_time - start_time);
    }

    free_sssp_tree(&t, g.V);
    free_graph(&g);
    MPI_Finalize();
    return 0;
}