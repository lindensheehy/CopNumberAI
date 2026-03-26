#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <system_error>
#include <cstring>

// ==========================================
// DECLARATIONS (Header Equivalents)
// ==========================================

// --- File I/O ---
// Reads from fileName and returns a new heap allocated buffer containing the contents
uint8_t* readFile(const char* fileName, std::uintmax_t* fileLengthOut);
// Returns the length of fileName
bool getFileLength(const char* fileName, std::uintmax_t* fileLengthOut);

// --- Graph ---
class Graph {
    /*
        General purpose data structure for storing undirected graphs
        Uses an internal adjacency matrix for edge states
    */
public: 
    int nodeCount;
    int edgeCount;

    Graph();
    Graph(const char* fileName);
    ~Graph();

    // Returns true if an edge exists between the two passed nodes
    bool getEdge(int node1, int node2) const;

private:
    bool* g; // Adjacency matrix
};

// --- Adjacency List ---
class AdjacencyList {
    /*
        General purpose Adjacency List
        This uses a flat contiguous array with a constant stride based on the max degree within the graph
        Intended for large, sparse graphs
    */
public:
    int nodeCount;
    int maxDegree;

    AdjacencyList(Graph* g);
    AdjacencyList(int nodeCount, int maxDegree);
    ~AdjacencyList();

    // Returns a pointer to list of edges connected to node. This list will have length at most this->maxDegree. 
    // The value 255 serves as a terminator of the data (no more edges are connected), even if the index has not reached maxDegree-1
    uint8_t* getEdges(int node) const;

    // Adds the edge (u, v) to the internal array   
    void addEdge(uint8_t u, uint8_t v);

private:
    uint8_t* edges;
};


// ==========================================
// IMPLEMENTATIONS (Source Equivalents)
// ==========================================

// --- File I/O Implementation ---

uint8_t* readFile(const char* fileName, std::uintmax_t* fileLengthOut) {
    std::uintmax_t fileLength;
    bool fileLengthErr = getFileLength(fileName, &fileLength);
    
    if (fileLengthErr) return nullptr;
    if (fileLength == 0) return nullptr;
    if (fileLengthOut == nullptr) return nullptr;
    
    *fileLengthOut = fileLength;

    std::FILE* file = std::fopen(fileName, "rb");
    if (!file) return nullptr;

    uint8_t* buf = new uint8_t[fileLength];
    size_t n = std::fread(buf, 1, fileLength, file);
    std::fclose(file);

    if (n != fileLength || n == 0) {
        delete[] buf;
        return nullptr;
    }

    return buf;
}

bool getFileLength(const char* fileName, std::uintmax_t* size) {
    std::error_code ec;
    std::uintmax_t temp = std::filesystem::file_size(fileName, ec);
    if (ec) return true;
    
    *size = temp;
    return false;
}

// --- Graph Implementation ---

Graph::Graph() {
    this->g = nullptr;
    this->nodeCount = 0;
    this->edgeCount = 0;
}

Graph::Graph(const char* fileName) {
    this->g = nullptr;
    this->nodeCount = 0;
    this->edgeCount = 0;

    std::uintmax_t fileLength = 0;
    uint8_t* buf = readFile(fileName, &fileLength);

    if (!buf) return;

    // 1. Determine nodeCount by scanning until the first newline or '-'
    uint32_t cols = 0;
    while (cols < fileLength && buf[cols] != '\n' && buf[cols] != '\r' && buf[cols] != '-') {
        cols++;
    }

    // Protect against empty or heavily malformed files
    if (cols == 0) {
        delete[] buf;
        return; 
    }

    this->nodeCount = cols;

    // 2. Allocate the flat 1D Adjacency Matrix
    this->g = new bool[this->nodeCount * this->nodeCount]{false};

    // 3. Parse the matrix
    int row = 0;
    int col = 0;
    int totalOnes = 0;

    for (size_t i = 0; i < fileLength; ++i) {
        char c = static_cast<char>(buf[i]);

        if (c == '-') break; // End marker hit

        if (c == '0' || c == '1') {
            if (row < this->nodeCount && col < this->nodeCount) {
                bool isEdge = (c == '1');
                
                // MATH: Convert 2D coordinates to 1D index
                this->g[row * this->nodeCount + col] = isEdge;
                
                if (isEdge) totalOnes++;
            }
            col++;
        } else if (c == '\n') {
            if (col > 0) { 
                row++;
                col = 0;
            }
        }
    }

    // 4. Calculate edge count (every undirected edge is listed twice)
    this->edgeCount = totalOnes / 2;
    delete[] buf;
}

Graph::~Graph() {
    delete[] this->g;
}

bool Graph::getEdge(int node1, int node2) const {
    if (!this->g) return false;

    // Bounds check
    if (node1 < 0 || node1 >= this->nodeCount || node2 < 0 || node2 >= this->nodeCount) {
        return false;
    }

    return this->g[node1 * this->nodeCount + node2];
}

// --- AdjacencyList Implementation ---

AdjacencyList::AdjacencyList(Graph* g) {
    nodeCount = g->nodeCount;

    // Step 1: Determine maxDegree
    maxDegree = 0;
    for (int i = 0; i < nodeCount; ++i) {
        int currentDegree = 0;
        for (int j = 0; j < nodeCount; ++j) {
            if (g->getEdge(i, j)) {
                currentDegree++;
            }
        }
        if (currentDegree > maxDegree) {
            maxDegree = currentDegree;
        }
    }

    maxDegree++;

    // Step 2: Allocate memory and initialize terminators
    int totalSize = nodeCount * maxDegree;
    edges = new uint8_t[totalSize];
    std::memset(edges, 255, totalSize);

    // Step 3: Populate the flat array directly
    for (int i = 0; i < nodeCount; ++i) {
        int offset = i * maxDegree;
        int edgeIndex = 0;
        for (int j = 0; j < nodeCount; ++j) {
            if (g->getEdge(i, j)) {
                edges[offset + edgeIndex] = (uint8_t)j;
                edgeIndex++;
            }
        }
    }
}

AdjacencyList::AdjacencyList(int nodeCount, int maxDegree) : nodeCount(nodeCount), maxDegree(maxDegree) {
    int totalSize = nodeCount * maxDegree;
    this->edges = new uint8_t[totalSize];
    std::memset(this->edges, 255, totalSize);
}

AdjacencyList::~AdjacencyList() {
    delete[] this->edges;
}

uint8_t* AdjacencyList::getEdges(int node) const {
    return &(this->edges[node * maxDegree]);
}

void AdjacencyList::addEdge(uint8_t u, uint8_t v) {
    int offset = u * maxDegree;
    for (int i = 0; i < maxDegree; ++i) {
        if (this->edges[offset + i] == 255) {
            this->edges[offset + i] = v;
            return;
        }
    }
}