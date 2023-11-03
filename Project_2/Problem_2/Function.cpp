#include "Function.h"

// Astar algorithm
vector<int> astar(map<int, vector<path>>& graph, int start, int end) {
    vector<int> result;
    priority_queue<Node *, std::vector<Node *>, CompareNode> pq;
    // The shortest path to the next is the heuristic
    int first_fn = 256;
    for (auto &i: graph[0])
    {
        int cost = i.cost;
        if (cost < first_fn)
            first_fn = cost;
    }
    Node node_start = {first_fn, 0, first_fn, 0};
    pq.push(&node_start);

    while (!pq.empty()) {
        Node *currentNode = pq.top();
        pq.pop();
        int fn_cur = currentNode->fn;
        int gn_cur = currentNode->gn;
        int hn_cur = currentNode->hn;
        int room_cur = currentNode->start_point;
        if (room_cur == end) {
            result.push_back(gn_cur);
            continue;
        }
        for (auto &i: graph[room_cur])
        {
            int neighbour = i.target;
            int heuristic = 256;
            // The shortest path to the next is the heuristic
            for (auto &j: graph[neighbour])
            {
                int path_cost = j.cost;
                if (path_cost < heuristic)
                    heuristic = path_cost;
            }
            int cost = i.cost;
            int new_cost = gn_cur + cost;
            int new_hn = heuristic;
            Node* node_next = new Node{new_cost + new_hn, new_cost, new_hn, neighbour};
            pq.push(node_next);
        }

    }
    return result;
}

vector<int> findResult(int N, int M, vector<vector<int>>& connections)
{
    map<int, vector<path>> graph;

    for (auto & connection : connections)
    {
        int x = connection[0];
        int y = connection[1];
        int d = connection[2];
        graph[x].push_back({y,d});
    }

    vector<int> result = astar(graph, 0, N-1);

    return result;
}