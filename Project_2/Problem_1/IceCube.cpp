#include <iostream>
#include "AStar.h"
using namespace std;

int main()
{
//    int i, j;
    char input[9];
    cout<<"Please print a nine-digit number:";
    cin.getline(input, 10);
    vector<vector<int>> start = input_number(input);
    vector<vector<int>> target = {{1, 3, 5}, {7, 0, 2}, {6, 8, 4}};
//    // Check the input.
//    for (i=0;i<3;i++)
//    {
//        for (j=0;j<3;j++)
//        {
//            cout<<start[i][j];
//        }
//    }
//    // Check the answer.
//    for (i=0;i<3;i++)
//    {
//        for (j=0;j<3;j++)
//        {
//            cout<<target[i][j];
//        }
//    }
    int solution = Astar(start, target);
    cout<<solution<<endl;
    
    return 0;
}