#include "utils.h"
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <boost/algorithm/string.hpp>


std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>> allocateBits(const std::vector<std::vector<std::string>>& attributes, int& na, int& bitLength) {
    int attribute_num = 0;
    std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>> bitMap;
    std::vector<std::unordered_set<std::string>> uniqueAttributes(na);


    for (size_t i = 0; i < attributes.size(); i++) {
        for (int j = 0; j < na; j++) {
            uniqueAttributes[j].insert(attributes[i][j]);
        }
    }
    for(const auto& attribute : uniqueAttributes){
        attribute_num += attribute.size();
    }

    int bitPosition = 0;
    for(int i = 0; i < na; i++){
        for (const auto& value : uniqueAttributes[i]) {
            boost::dynamic_bitset<> bits(attribute_num);
            bits.set(bitPosition);
            bitMap[i][value] = bits;
            bitPosition++;
        }
    }

    bitLength = attribute_num;  
    return bitMap;
}


std::string getBitKey(const std::vector<std::string>& attributes, std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>>& bitMap, int& na, int& bitLength) {
    boost::dynamic_bitset<> bitKey(bitLength); 

    for (int i = 0; i < na; i++) {
        auto it = bitMap[i].find(attributes[i]);
        if (it != bitMap[i].end()) {
            bitKey |= it->second; 
        }
    }

    std::string bitKeyString;
    boost::to_string(bitKey, bitKeyString);
    return bitKeyString;
}



std::vector<std::unordered_map<std::string, std::vector<int>>> buildClusterReferenceTable(int& nb, int& na, const int& nc, std::vector<std::vector<std::string>>& attributes, const std::vector<long>& assign, int& bitLength, std::unordered_map<int, std::unordered_map<std::string, boost::dynamic_bitset<>>>& bitMap) {
    std::vector<std::unordered_map<std::string, std::vector<int>>> clusterReferenceTables(nc);
    for (size_t i = 0; i < nb; ++i) {
        int clusterId = assign[i];
        std::string bitKey = getBitKey(attributes[i], bitMap, na, bitLength);
        clusterReferenceTables[clusterId][bitKey].push_back(i);
    }
    return clusterReferenceTables;
}



void buildKnnGraph(float* data, const std::vector<int>& indices, std::ofstream& outFile, const int& d, const int& K_NEIGHBORS) {
    size_t nb = indices.size();
    std::vector<float> xb(d * nb);
    for (size_t i = 0; i < nb; ++i) {
        for (size_t j = 0; j < d; ++j) {
            xb[d * i + j] = data[indices[i] * d + j];
        }
    }

    faiss::IndexFlatL2 index(d);
    index.add(nb, xb.data());

    std::vector<faiss::idx_t> I(nb * (K_NEIGHBORS + 1));
    std::vector<float> D(nb * (K_NEIGHBORS + 1));

    index.search(nb, xb.data(), K_NEIGHBORS + 1, D.data(), I.data());

    for (size_t i = 0; i < nb; ++i) {
        outFile << indices[i] << " ";
        int neighborsFound = 0;
        for (size_t j = 0; j < K_NEIGHBORS + 1; ++j) {
            if (I[i * (K_NEIGHBORS + 1) + j] != i) { 
                outFile << indices[I[i * (K_NEIGHBORS + 1) + j]] << " ";
                neighborsFound++;
                if (neighborsFound == K_NEIGHBORS) {
                    break;
                }
            }
        }
        outFile << std::endl;
    }
}

static inline __m128 masked_read (int d, const float *x)
{
    assert (0 <= d && d < 4);
    __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
    switch (d) {
      case 3:
        buf[2] = x[2];
      case 2:
        buf[1] = x[1];
      case 1:
        buf[0] = x[0];
    }
    return _mm_load_ps (buf);
    // cannot use AVX2 _mm_mask_set1_epi32
}


float calculateL2Distance (const float *x, const float *y, size_t d){
    __m512 msum1 = _mm512_setzero_ps();

    while (d >= 16) {
        __m512 mx = _mm512_loadu_ps (x); x += 16;
        __m512 my = _mm512_loadu_ps (y); y += 16;
        const __m512 a_m_b1 = mx - my;
        msum1 += a_m_b1 * a_m_b1;
        d -= 16;
    }

    __m256 msum2 = _mm512_extractf32x8_ps(msum1, 1);
    msum2 +=       _mm512_extractf32x8_ps(msum1, 0);

    while (d >= 8) {
        __m256 mx = _mm256_loadu_ps (x); x += 8;
        __m256 my = _mm256_loadu_ps (y); y += 8;
        const __m256 a_m_b1 = mx - my;
        msum2 += a_m_b1 * a_m_b1;
        d -= 8;
    }

    __m128 msum3 = _mm256_extractf128_ps(msum2, 1);
    msum3 +=       _mm256_extractf128_ps(msum2, 0);

    if (d >= 4) {
        __m128 mx = _mm_loadu_ps (x); x += 4;
        __m128 my = _mm_loadu_ps (y); y += 4;
        const __m128 a_m_b1 = mx - my;
        msum3 += a_m_b1 * a_m_b1;
        d -= 4;
    }

    if (d > 0) {
        __m128 mx = masked_read (d, x);
        __m128 my = masked_read (d, y);
        __m128 a_m_b1 = mx - my;
        msum3 += a_m_b1 * a_m_b1;
    }

    msum3 = _mm_hadd_ps (msum3, msum3);
    msum3 = _mm_hadd_ps (msum3, msum3);
    return  _mm_cvtss_f32 (msum3);
}

void greedySearch(float* query, float* data, const std::unordered_map<int, std::vector<int>>& knnGraph, int startNodeId, const int& K_SEARCH, int& d, std::multimap<float, int>& nodeDistances) {
    std::multimap<float, int> Candidate;
    std::multimap<float, int> Candidate_copy;
    std::vector<int> result;
    std::unordered_map<int, bool> visitedMap;
    float tau = std::numeric_limits<float>::infinity();
    float l = calculateL2Distance(query, data + startNodeId * d, d);
    Candidate.emplace(l, startNodeId);
    Candidate.emplace(l, startNodeId);
    visitedMap[startNodeId] = true;

    while (!Candidate_copy.empty()) {
        std::multimap<float, int>::iterator it = Candidate_copy.begin();
        int currentNodeId = it->second;
        Candidate_copy.erase(it);
        const auto& neighbors = knnGraph.at(currentNodeId);
        for (int neighborId : neighbors) {
            if (visitedMap[neighborId]) {
                continue;
            }
            visitedMap[neighborId] = true;
            if(Candidate.size() == K_SEARCH){
                std::multimap<float, int>::reverse_iterator i = Candidate.rbegin();
                tau = i -> first;
            }
            float neighborDistance = calculateL2Distance(query, data + neighborId * d, d);
            if (neighborDistance < tau) {
                Candidate.emplace(neighborDistance, neighborId);
                if(Candidate.size() >  K_SEARCH){
                    std::multimap<float, int>::reverse_iterator ri = Candidate.rbegin();
                    Candidate.erase((++ri).base());
                }
                Candidate_copy.emplace(neighborDistance, neighborId);
            }
        }
    }
    for(const auto& c : Candidate){
        nodeDistances.emplace(c.first, c.second);
    }
}

std::vector<int> findClosestClusters(float* query, float* cluster, int& search_limit, int& d, int& nc) {
    std::multimap<float, int> clusterDistances;
    for (int i = 0; i < nc; i++) {
        float distance = calculateL2Distance(query, cluster + i * d, d);
        clusterDistances.emplace(distance, i);
    }
    auto it = clusterDistances.begin();
    std::vector<int> closestClusters;
    for (int i = 0; i < search_limit; i++) {
        closestClusters.push_back(it->second);
        ++it;
    }
    return closestClusters;
}

void findClosestNodes(float* query, float* data, const std::string& bitKey, std::unordered_map<int, std::unordered_map<std::string, std::vector<int>>>& clusterReferenceTables, std::unordered_map<int, std::unordered_map<std::string, std::unordered_map<int, std::vector<int>>>>& knnGraphs, const std::vector<int>& closestClusters, const int& K_SEARCH, int& d, std::vector<int>& matchingNodes) {
    std::multimap<float, int> nodeDistances;
    for (int clusterId : closestClusters) {
        auto it = clusterReferenceTables[clusterId].find(bitKey);
        if (it != clusterReferenceTables[clusterId].end()) {
            for (const int& node : it->second) {
                auto knnGraphIt = knnGraphs[clusterId].find(bitKey);
                if (knnGraphIt != knnGraphs[clusterId].end()) {
                    greedySearch(query, data, knnGraphs[clusterId][bitKey], node, K_SEARCH, d, nodeDistances);
                } else {
                    float distance = calculateL2Distance(query, data + node * d, d);
                    nodeDistances.emplace(distance, node);
                }
            }
        }
    }

    for(const auto& c : nodeDistances){
        if(matchingNodes.size() >= K_SEARCH){
            break;
        }
        matchingNodes.push_back(c.second);
    }
}


void calculateAverageTime(std::vector<std::chrono::duration<float>> query_time){
    std::chrono::duration<float> total_time(0);
    for(auto& time : query_time){
        total_time += time;
    }
    float average_time_ms = static_cast<float>((total_time / query_time.size()).count() * 1000);
    std::cout << average_time_ms << " ";
}

void calculateRecall(const std::vector<std::vector<int>>& linear, const std::vector<std::vector<int>>& graph){
    std::vector<float> result(graph.size());
    int count = 0;
    for(int i=0; i < graph.size(); i++){
        if(linear[i].size() == 0){
            result[i] = 10000;
        } else {
            for(const auto& row1 : linear[i]){
                for(const auto& row2 : graph[i]){
                    if(row1==row2){
                        count += 1;
                    }
                }
            }
            if(count!=0){
                result[i] = static_cast<float>(count) / linear[i].size();
            } else {
                result[i] = 0;
            }
        }
        count = 0;
    }

    float sum = 0.0;
    float ave = 0.0;
    int count_num = 0;

    for(int i=0; i < result.size(); i++){
        if(result[i] != 10000){
            sum += result[i];
            count_num++;
        }

    }
    ave = sum / count_num;

    std::cout << ave << std::endl;
}

