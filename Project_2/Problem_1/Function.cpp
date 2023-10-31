#include "Astar.h"

// Change input into a matrix
vector<vector<int>> input_number(string input)
{
    vector<vector<int>> matrix(3, vector<int>(3));
    for (int i = 0; i < 9; i++)
    {
        int row = i / 3;
        int col = i % 3;
        matrix[row][col] = input[i] - '0';
    }
    return matrix;
}

// Calculate the Manhattan Distance of all nodes
int Manhattan(vector<vector<int>>& current, vector<vector<int>>& target)
{
    int distance = 0;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    if (current[i][j] == target[k][l] && current[i][j] != 0)
                    {
                        distance = distance + abs(k - i) + abs(l - j);
                    }
                }
            }
        }
    }
    return distance;
}

// Make sure the state is okay
bool isSafe(int x, int y)
{
    return (x >= 0 && y >= 0 && x <= 2 && y <= 2);
}

// Change the state
IceState move(vector<vector<int>>& Ice, int x, int y, int new_x, int new_y, int gn, int number)
{
    vector<vector<int>> newIce = Ice;
    swap(newIce[x][y], newIce[new_x][new_y]);

    IceState newIceState = {newIce, Manhattan(newIce, Ice) + gn, gn + 1, new_x, new_y, number + 1};
    return newIceState;
}

// A-star Algorithm
int Astar(vector<vector<int>>& start, vector<vector<int>>& target)
{
    // Record visited states, namely Close
    map<vector<vector<int>>, bool> visitedStates;
    priority_queue<IceState, vector<IceState>, Compare> pq;
    // Find where 0 lies in
    int zero_x, zero_y;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            if (start[i][j] == 0)
            {
                zero_x = i;
                zero_y = j;
                break;
            }
        }
    }
    // Initialize
    IceState startState = {start, Manhattan(start, target), 0, zero_x, zero_y, 0};
    pq.push(startState);
    // Open
    while (!pq.empty())
    {
        IceState current = pq.top();
        pq.pop();
        if (current.Ice == target)
            return current.number;
        visitedStates[current.Ice] = true;

        int x = current.x;
        int y = current.y;
        int new_x, new_y;
        // left, right, down, up
        int direct_x[] = {-1, 1, 0, 0};
        int direct_y[] = {0, 0, -1, 1};

        for (int i = 0; i < 4; i++)
        {
            new_x = x + direct_x[i];
            new_y = y + direct_y[i];

            if (isSafe(new_x, new_y))
            {
                IceState newState = move(current.Ice, x, y, new_x, new_y, current.gn, current.number);
                if (!visitedStates[newState.Ice])
                {
                    pq.push(newState);
                }
            }
        }
    }
    return -1;

}
