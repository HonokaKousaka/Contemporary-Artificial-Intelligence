#include <iostream>
#include "Function.h"

int main()
{
    // Input the first line
    int room_num, path_num, path_go;
    cin >> room_num >> path_num >> path_go;

    vector<vector<int>> paths;

    for (int i = 0; i < path_num; i++) {
        vector<int> connection(3);
        for (int j = 0; j < 3; j++) {
            cin >> connection[j];
            if (j != 2)
                connection[j]--;
        }
        if (connection[0] < connection[1])
            paths.push_back(connection);
    }
    vector<int> result = findResult(room_num, path_num, paths);
    sort(result.begin(), result.end());
    int diff = path_go - int(result.size());
    for (int i = 0; i < result.size(); i++)
    {
        if (i < path_go - 1)
            cout << result[i] << endl;
        else
        {
            cout << result[i];
            break;
        }
    }
    for (int i = 0; i < diff; i++)
    {
        if (i < path_go - result.size() - 1)
            cout << -1 << endl;
        else
            cout << -1;
    }
    return 0;
}