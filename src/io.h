#ifndef IO_H
#define IO_H

#include <string>
#include <vector>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>


float* loadDatasetForIndex(const std::string& filename, const int& d, int& nb, int& na, std::vector<std::vector<std::string>>& attributes);

float* loadDatasetForQuery(const std::string& filename, int& numData, int& dimension, int& na);

float* loadQueryset(const std::string& filename, int& numQuery, int& dimension, std::vector<std::vector<std::string>>& queryattributes);

std::vector<std::vector<int>> loadGroundtruthset(const std::string& filepath, const int dim);

std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>> loadBitMap(const std::string& foldername, int& bitLength);

float* loadClusterCenters(const std::string& foldername, int& dimension, int& nc);

std::unordered_map<int, std::unordered_map<std::string, std::vector<int>>> loadClusterReferenceTable(const std::string& foldername, int& nc);

std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>> loadKnnGraphs(const std::string& foldername, int& nc);

void outputBitMap(const std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>>& bitMap, const std::string& filename, int& bitLength);

void outputClusterCenters(const std::vector<float>& centroids, int dimension, const std::string& filename, const int& NUM_CLUSTERS);

void outputClusterReferenceTable(const std::vector<std::unordered_map<std::string, std::vector<int>>>& clusterReferenceTables, const std::string& filename, const int& NUM_CLUSTERS);

void outputKnnGraph(const std::string& filename, float* data, const std::vector<std::unordered_map<std::string, std::vector<int>>>& clusterReferenceTables, const int& d, const int& NUM_CLUSTERS, const int& MAX_NODE_NUM, const int& K_NEIGHBORS);




#endif // IO_H
