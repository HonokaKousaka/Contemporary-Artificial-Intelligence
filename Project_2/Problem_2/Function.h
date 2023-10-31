#ifndef PROBLEM_2_FUNCTION_H
#define PROBLEM_2_FUNCTION_H

#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
using namespace std;

struct path {
    int target;
    int cost;
};

struct Node
{
    int fn;
    int gn;
    int hn;
    int start_point;
};

struct CompareNode
{
    bool operator()(const Node* lhs, const Node* rhs) const {
        return lhs->fn > rhs->fn;
    }
};

vector<int> astar(map<int, vector<path>>& graph, int start, int end);
vector<int> findResult(int N, int M, vector<vector<int>>& connections);

#endif //PROBLEM_2_FUNCTION_H
