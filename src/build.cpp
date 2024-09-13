#include "utils.h"
#include "io.h"
#include <iostream>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

int main(int argc, char** argv) {
    std::string filename = argv[1];
    std::string foldername = argv[2];
    const int d = atoi(argv[3]);
    const int nc = atoi(argv[4]);
    const int MAX_NODE_NUM = atoi(argv[5]);
    const int K_NEIGHBORS = atoi(argv[6]);
    int nb, na, bitLength;
    std::vector<std::vector<std::string>> attributes;
    float* data = loadDatasetForIndex(filename, d, nb, na, attributes);

    //allocating bits
    std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>> bitMap = allocateBits(attributes, na, bitLength);

    //clustering using FAISS
    faiss::IndexFlatL2 index(d);
    faiss::Clustering clus(d, nc);
    clus.train(nb, data, index);

    //getting data indices for each cluster
    std::vector<faiss::idx_t> assign(nb);
    index.assign(nb, data, assign.data());

    //outputting cluster centers
    outputClusterCenters(clus.centroids, d, foldername + "/ClusterCenters.txt", nc);

    //building bit reference tables for each cluster
    std::vector<std::unordered_map<std::string, std::vector<int>>> clusterReferenceTables = buildClusterReferenceTable(nb, na, nc, attributes, assign, bitLength, bitMap);

    //outputting cluster reference tables
    outputClusterReferenceTable(clusterReferenceTables, foldername + "/ClusterReferenceTables.txt", nc);

    //outputting kNN graphs
    outputKnnGraph(foldername + "/kNNGraphs.txt", data, clusterReferenceTables, d, nc, MAX_NODE_NUM, K_NEIGHBORS);

    //outputting bitMap
    outputBitMap(bitMap,foldername +"/bitMap.txt", bitLength);

    return 0;
}
