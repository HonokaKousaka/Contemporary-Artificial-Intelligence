#ifndef PROJECT_2_ASTAR_H
#define PROJECT_2_ASTAR_H

#include <vector>
#include <map>
#include <queue>
using namespace std;

// Defines a state
struct IceState
{
    vector<vector<int>> Ice;
    // fn = gn + hn, where gn stands for the depth, hn stands for the prediction
    // and fn stands for the total
    int fn;
    int gn;
    // The position of empty
    int x;
    int y;
    int number;
};

// Will be used in A-star's priority queue
struct Compare
{
    bool operator()(const IceState& s1, const IceState& s2)
    {
        return s1.fn > s2.fn;
    }
};

std::vector<std::vector<int>> input_number(std::string input);
int Manhattan(std::vector<std::vector<int>>& current, std::vector<std::vector<int>>& target);
bool isSafe(int x, int y);
IceState move(std::vector<std::vector<int>>& Ice, int x, int y, int new_x, int new_y, int gn, int number);
int Astar(std::vector<std::vector<int>>& start, std::vector<std::vector<int>>& target);

#endif //PROJECT_2_ASTAR_H
