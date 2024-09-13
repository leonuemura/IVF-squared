#include "utils.h"
#include "io.h"
#include <iostream>
#include <random>
#include <chrono>


int main(int argc, char *argv[]) {
    std::string datafilename = argv[1];  //dataset file
    std::string queryfilename = argv[2];  //query file
    std::string foldername = argv[3]; //index folder
    std::string linearfilename = argv[4]; //groundtruth file
    int d = atoi(argv[5]); //dimension
    int nc = atoi(argv[6]); //number of clusters
    int K_SEARCH = atoi(argv[7]); //number of search
    int search_limit = atoi(argv[8]); //number of clusters to search
    int nb, na, nq, bitLength; //number of data, number of query, bit length

    //Loading dataset
    float* data = loadDatasetForQuery(datafilename, nb, d, na);

    //Loading queryset
    std::vector<std::vector<std::string>> queryattributes;
    float*  query = loadQueryset(queryfilename, nq, d, queryattributes);
    
    //Loading groundtruthset
    std::vector<std::vector<int>> groundtruth = loadGroundtruthset(linearfilename, K_SEARCH);
    std::vector<std::vector<int>> query_result(nq); //vector to store query results
    std::vector<std::chrono::duration<float>> query_time; //vector to store query time

    //Loading bitMap
    std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>> bitMap = loadBitMap(foldername, bitLength);

    //Loading cluster centers
    float* cluster = loadClusterCenters(foldername, d, nc);


    //Loading cluster reference tables
    std::unordered_map<int, std::unordered_map<std::string, std::vector<int>>> clusterReferenceTables = loadClusterReferenceTable(foldername, nc);


    //Loading kNN graphs
    std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>> knnGraphs = loadKnnGraphs(foldername, nc);


    //Querying
    for (int i=0; i < nq; i++){
        auto start = std::chrono::high_resolution_clock::now();
        std::string bitKey = getBitKey(queryattributes[i], bitMap, na, bitLength);
        std::vector<int> closestClusters = findClosestClusters(query + i * d, cluster, search_limit, d, nc);
        findClosestNodes(query + i * d, data, bitKey, clusterReferenceTables, knnGraphs, closestClusters, K_SEARCH, d, query_result[i]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        query_time.push_back(duration);
    }

    //Calculate average time
    calculateAverageTime(query_time);

    //Calculate recall
    calculateRecall(groundtruth, query_result);


    return 0;
}