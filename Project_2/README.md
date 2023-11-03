# Project 2: A* Algorithm
This is a task about solving 2 problems by using A* algorithm.

&nbsp;

## Requirements

If you are willing to run the codes, you have to make preparations as follows.

### Environment: C++

C++ is compulsory for the codes. 

Although other IDE and text editors are acceptable, **CLion** is strongly recommended when running the codes. The codes consist of some .cpp files and .h files, so it is necessary to link these files. CLion is quite useful when using the linker, and the executable files are created by CLion.

&nbsp;

## Work

### Step 1: 

Create a priority queue OpenList storing those nodes which will be expanded later. Create a CloseList as well in order to confirm whether a node has been expanded. The starting node is stored in the OpenList at the beginning, while the CloseList is empty at first.

### Step 2: 

Choose the node whose $f$ is the smallest, named $best$, and set it into CloseList. If the node is the finishing node, return $g$ as the final answer to the problem. If it is not the finishing node, then get all its successors as a set $subs$.

### Step 3:

Get each node in $subs$, and calculate the cost on reaching the node, which is $g'(sub)=g(best)+c(best, sub)$. 

**If** $sub$ **is already in OpenList, then compare** $g'(sub)$ **with** $g(sub)$. 

If is this situation, $g'(sub)$ is smaller, then substitute the old $g(sub)$ with $g'(sub)$. If in this situation, $g'(sub)$ is not smaller, then get another $sub$ in $subs$, repeat Step 3.

**If** $sub$ **is not in OpenList, then check whether it is in CloseList.**

If it is in CloseList, then compare $g'(sub)$ with $g(sub)$. If is this situation, $g'(sub)$ is smaller, then substitute the old $g(sub)$ with $g'(sub)$. If in this situation, $g'(sub)$ is not smaller, then get another $sub$ in $subs$, repeat this step.

If it is not in CloseList, then add $sub$ into OpenList.

Each time a $sub$ is added into OpenList, calculate $f(sub)=g'(sub)+h(sub)$. Repeat Step 3 until $subs$ is empty.

&nbsp;

## Results

A* algorithm is a path search algorithm, which can make sure the shortest paths will be found and reduce the time spent on finding the shortest paths. However, a good heuristic function is needed, which is not simple. We need to consider more when we are looking for a heuristic function.

&nbsp;

## Files

In **Project_1**:

- Astar.h: Define necessary struct and claim some functions which is necessary for Problem 1.
- Function.cpp: Details of the functions mentioned in Astar.h for Problem 1.
- IceCute.cpp: Define the standard answer, input and output of Problem 1.
- Problem_1.exe: The executable file of Problem 1.

In **Project_2**:

- Function.h: Define necessary struct and claim some functions which is necessary for Problem 2.
- Function.cpp: Details of the functions mentioned in Function.h for Problem 2.
- Jack.cpp: Define the input and output of Problem 2.
- Problem_2.exe: The executable file of Problem 2.
