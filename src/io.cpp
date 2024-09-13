#include "io.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <sstream>

float* loadDatasetForIndex(const std::string& filename, const int& d, int& nb, int& na, std::vector<std::vector<std::string>>& attributes) {
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);
    nb = std::stoi(line); 


    std::getline(file, line);
    std::string token;
    na = std::stoi(line); 

    float* node = new float[nb * d];
    int count = 0;


    while (std::getline(file, line)) {
        std::istringstream dataStream(line);
        std::vector<std::string> dataTokens;
        while (std::getline(dataStream, token, ' ')) {
            dataTokens.push_back(token);
        }


        std::vector<std::string> attribute(na);
        for (int i = 0; i < na; ++i) {
            attribute[i] = dataTokens[i];
        }

        attributes.push_back(attribute);


        for (size_t i = 0; i < d; ++i) {
            node[count * d + i] = std::stof(dataTokens[i + na]);
        }
        count++;
    }

    return node;
}


float* loadDatasetForQuery(const std::string& filename, int& numData, int& dimension, int& na) {
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);
    numData = std::stoi(line); 

    std::getline(file, line);
    na = std::stoi(line); 
    std::string token;

    float* data = new float[numData * dimension];

    int count = 0;

    while (std::getline(file, line)) {
        std::istringstream dataStream(line);
        std::vector<std::string> dataTokens;
        while (std::getline(dataStream, token, ' ')) {
            dataTokens.push_back(token);
        }
        for (size_t i = 0; i < dimension; ++i) {
            data[count * dimension + i] = std::stof(dataTokens[i + na]);
        }
        count++;
    }
    return data;
}


float* loadQueryset(const std::string& filename, int& numQuery, int& dimension, std::vector<std::vector<std::string>>& queryattributes) {
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line);
    numQuery = std::stoi(line); 

    std::getline(file, line);
    std::string token;
    int numAttributes = std::stoi(line); 

    float *query = new float[numQuery * dimension];

    int count = 0;

    while (std::getline(file, line)) {
        std::istringstream dataStream(line);
        std::vector<std::string> dataTokens;
        std::vector<std::string> attributes;
        int attcount = 0;
        while (std::getline(dataStream, token, ' ')) {
            dataTokens.push_back(token);
        }

        for (int i = 0; i < numAttributes; ++i) {
            attributes.push_back(dataTokens[i]);
        }
        queryattributes.push_back(attributes);

        for (size_t i = 0; i < dimension; ++i) {
            query[count * dimension + i] = std::stof(dataTokens[i + numAttributes]);
        }
        count++;
    }

    return query;
}

std::vector<std::vector<int>> loadGroundtruthset(const std::string& filepath, const int dim) {
    std::vector<std::vector<int>> data; 

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open the file." << std::endl;
        return data; 
    }

    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<int> row; 
        std::string token;
        int count = 0;
        while (std::getline(ss, token, ' ') && count < dim) {
            int value = std::stoi(token); 
            row.push_back(value); 
            count++;
        }

        data.push_back(row); 
    }

    file.close();
    return data;
}


std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>> loadBitMap(const std::string& foldername, int& bitLength) {
    std::ifstream inFile(foldername + "/bitMap.txt");
    std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>> bitMap;
    if (inFile.is_open()) {
        inFile >> bitLength;
        inFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 

        std::string line;
        while (std::getline(inFile, line)) {
            std::istringstream iss(line);
            int attributeIndex;
            std::string value;
            std::string bitsetStr;

            iss >> attributeIndex >> value >> bitsetStr;

            boost::dynamic_bitset<> bits;
            bits.resize(bitLength); 
            for (size_t i = 0; i < bitsetStr.size(); ++i) {
                if (bitsetStr[i] == '1') {
                    bits.set(bitsetStr.size() - 1 - i);
                }
            }

            bitMap[attributeIndex][value] = bits;
        }
        inFile.close();
    } else {
        std::cerr << "Could not open the file : " << foldername + "/bitMap.txt" << std::endl;
    }
    return bitMap;
}

float* loadClusterCenters(const std::string& foldername, int& dimension, int& nc) {
    std::ifstream inFile(foldername + "/ClusterCenters.txt");
    float* clusterCenters = new float[nc * dimension];

    if (inFile.is_open()) {
        for (int i = 0; i < nc; ++i) {
            for (int j = 0; j < dimension; ++j) {
                inFile >> clusterCenters[dimension * i + j];
            }
        }
        inFile.close();
    } else {
        std::cerr << "Could not open the file : " << (foldername + "/ClusterCenters.txt").c_str() << std::endl;
    }

    return clusterCenters;
}

std::unordered_map<int, std::unordered_map<std::string, std::vector<int>>> loadClusterReferenceTable(const std::string& foldername, int& nc) {
    std::ifstream inFile(foldername + "/ClusterReferenceTables.txt");
    std::vector<std::unordered_map<std::string, std::vector<int>>> clusterReferenceTables_vector(nc);
    std::unordered_map<int, std::unordered_map<std::string, std::vector<int>>> clusterReferenceTables;
    

    if (inFile.is_open()) {
        std::string line;
        int currentClusterId = -1;

        while (std::getline(inFile, line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] == 'c') {
                currentClusterId = std::stoi(line.substr(1));
            } else if (currentClusterId >= 0) {
                std::istringstream iss(line);
                std::string bitKey;
                iss >> bitKey;  
                std::vector<int> indices;
                int index;
                while (iss >> index) {
                    indices.push_back(index);
                }
                clusterReferenceTables_vector[currentClusterId][bitKey] = indices;
            }
        }
        inFile.close();
    } else {
        std::cerr << "Could not open the file : " << (foldername + "/ClusterReferenceTables.txt").c_str() << std::endl;
    }

    for(int i=0; i < nc; i++){
        clusterReferenceTables[i] = clusterReferenceTables_vector[i];
    }

    return clusterReferenceTables;
}


std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>> loadKnnGraphs(const std::string& foldername, int& nc) {
    std::ifstream inFile(foldername + "/kNNGraphs.txt");
    std::vector<std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>> knnGraphs_vector(nc);
    std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>> knnGraphs;

    if (inFile.is_open()) {
        std::string line;
        int currentClusterId = -1;
        std::string currentBitKey;

        while (std::getline(inFile, line)) {
            if (line[0] == 'c') {
                currentClusterId = std::stoi(line.substr(1));
            } else if (line.find_first_not_of("0123456789") == std::string::npos) {
                currentBitKey = line;
            } else {
                std::istringstream iss(line);
                int nodeId;
                iss >> nodeId;  // ノードIDを取得
                std::vector<int> neighbors;
                int neighborId;
                while (iss >> neighborId) {
                    neighbors.push_back(neighborId);
                }
                knnGraphs_vector[currentClusterId][currentBitKey][nodeId] = neighbors;
            }
        }
        inFile.close();
    } else {
        std::cerr << "Could not open the file : " << (foldername + "/kNNGraphs.txt").c_str() << std::endl;
    }

    for(int i=0; i < nc; i++){
        knnGraphs[i] = knnGraphs_vector[i];
    }

    return knnGraphs;
}

void outputBitMap(const std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>>& bitMap, const std::string& filename, int& bitLength) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << bitLength << std::endl;
        for (const auto& attribute : bitMap) {
            int attributeIndex = attribute.first;
            for (const auto& value : attribute.second) {
                outFile << attributeIndex << " " << value.first << " " << value.second << "\n";
            }
        }
        outFile.close();
    } else {
        std::cerr << "Could not open the file : " << filename << std::endl;
    }
}

void outputKnnGraph(const std::string& filename, float* data, const std::vector<std::unordered_map<std::string, std::vector<int>>>& clusterReferenceTables, const int& d, const int& NUM_CLUSTERS, const int& MAX_NODE_NUM, const int& K_NEIGHBORS) {
    std::ofstream knnOutFile(filename);
    if (knnOutFile.is_open()) {
        for (int clusterId = 0; clusterId < NUM_CLUSTERS; ++clusterId) {
            for (const auto& entry : clusterReferenceTables[clusterId]) {
                if (entry.second.size() > MAX_NODE_NUM) {
                    knnOutFile << "c" << clusterId << std::endl;
                    knnOutFile << entry.first << std::endl;
                    buildKnnGraph(data, entry.second, knnOutFile, d, K_NEIGHBORS);
                }
            }
        }
        knnOutFile.close();
    } else {
        std::cerr << "Could not open the file : kNNGraphs.txt" << std::endl;
    }
}


void outputClusterCenters(const std::vector<float>& centroids, int dimension, const std::string& filename, const int& NUM_CLUSTERS) {
    // Output to file
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        for (int i = 0; i < NUM_CLUSTERS; ++i) {
            for (int j = 0; j < dimension; ++j) {
                outFile << centroids[dimension * i + j] << " ";
            }
            outFile << std::endl;
        }
        outFile.close();
    } else {
        std::cerr << "Could not open the file : " << filename << std::endl;
    }
}


void outputClusterReferenceTable(const std::vector<std::unordered_map<std::string, std::vector<int>>>& clusterReferenceTables, const std::string& filename, const int& NUM_CLUSTERS) {
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        for (int clusterId = 0; clusterId < NUM_CLUSTERS; ++clusterId) {
            outFile << "c" << clusterId << std::endl;
            for (const auto& entry : clusterReferenceTables[clusterId]) {
                outFile << entry.first << " ";
                for (int idx : entry.second) {
                    outFile << idx << " ";
                }
                outFile << std::endl;
            }
            outFile << std::endl;
        }
        outFile.close();
    } else {
        std::cerr << "Could not open the file : " << filename << std::endl;
    }
}