#include <iostream>
#include "Function.h"

int main()
{
    cout << "Please input:" <<endl;
    // Input the first line
    int room_num, path_num, path_go;
    cin >> room_num >> path_num >> path_go;

    vector<vector<int>> paths;

    // Input the paths
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
    cout << "---------------------" << endl;
    // In order to print as the problem asks
    // However, looks ugly when in .exe file.

//    for (int i = 0; i < result.size(); i++)
//    {
//        if (i < path_go - 1)
//            cout << result[i] << endl;
//        else
//        {
//            cout << result[i];
//            break;
//        }
//    }
//    for (int i = 0; i < diff; i++)
//    {
//        if (i < path_go - result.size() - 1)
//            cout << -1 << endl;
//        else
//            cout << -1;
//    }
    for (int i : result)
    {
        cout << i << endl;
    }
    for (int i = 0; i < diff; i++)
    {
        cout << -1 << endl;
    }


    system("pause");
    return 0;
}