#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>
#include <string>
#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>
#include <x86intrin.h>
#include <cassert>
#include <chrono>


std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>> allocateBits(const std::vector<std::vector<std::string>>& attributes, int& na, int& bitLength);


std::string getBitKey(const std::vector<std::string>& attributes, std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>>& bitMap, int& na, int& bitLength);


std::vector<std::unordered_map<std::string, std::vector<int>>> buildClusterReferenceTable(int& nb, int& na, const int& nc, std::vector<std::vector<std::string>>& attributes, const std::vector<faiss::idx_t>& assign, int& bitLength, std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>>& bitMap);

void buildKnnGraph(float* data, const std::vector<int>& indices, std::ofstream& outFile, const int& d, const int& K_NEIGHBORS);

static inline __m128 masked_read(int d, const float *x);

float calculateL2Distance(const float *x, const float *y, size_t d);

void greedySearch(float* query, float* data, const std::unordered_map<int, std::vector<int>>& knnGraph, int startNodeId, const int& K_SEARCH, int& d, std::multimap<float, int>& nodeDistances);

std::vector<int> findClosestClusters(float* query, float* cluster, int& search_limit, int& d, int& nc) ;

void findClosestNodes(float* query, float* data, const std::string& bitKey, std::unordered_map<int, std::unordered_map<std::string, std::vector<int>>>& clusterReferenceTables, std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>>& knnGraphs, const std::vector<int>& closestClusters, const int& K_SEARCH, int& d, std::vector<int>& matchingNodes) ;

void calculateAverageTime(std::vector<std::chrono::duration<float>> query_time);

void calculateRecall(const std::vector<std::vector<int>>& linear, const std::vector<std::vector<int>>& graph);


#endif // UTILS_H
