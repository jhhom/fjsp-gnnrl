Action: Dispatch Job 1, Operation 2 on Machine 2 (Duration = 2)

Jobs
```
1 3 2    0 0 2    2 1 0    
0 2 1    2 2 1    1 0 1    
0 0 3    0 0 0    3 0 0
```

Job Makespans
```
2 2 2    0 0 2    2 1 0    
0 2 1    2 2 1    1 0 1    
0 0 3    0 0 0    3 0 0
```

Result: Operation end times
```
2 2 2    0 0 0    0 0 0    
4 4 4    0 0 0    0 0 0    
0 0 0    0 0 0    0 0 0
```

Result: Job Makespans (after)
```
0 0 0    0 0 2    2 1 0    
4 4 4    2 2 1    1 0 1    
0 0 3    0 0 0    3 0 0
```

Result: Operation lower bounds
```
2 2 2    0 0 2    2 1 0    
4 4 4    4 4 3    2 0 2
0 0 7    0 0 0    5 0 0
```



* Line 1: Find the last non-zero:
    * job = [0]
    * operation [1]
  
    * This means the last non-zero for the first matrix is the second operation (second row of first matrix)

* Line 2: 
    ```
    0 0 0    0 0 2    2 1 0    
    0 0 0    2 2 1    1 0 1    
    0 0 3    0 0 0    3 0 0
    ```

    * In Job Makespan matrix, Set all the numbers before and in the last non-zero to zeroes

* Line 3:
    ```
    0 0 0    0 0 2    2 1 0    
    4 4 4    2 2 1    1 0 1    
    0 0 3    0 0 0    3 0 0
    ```

    * Set the last non-zeroes to their operation end times


* Line 4:
    ```
    9999 9999 9999    9999 9999    2       2    1 9999    
    4    4    4       2    2    1          1 9999    1    
    9999 9999    3    9999 9999 9999       3 9999 9999
    ```

    * Set all the zeroes in Job makespans to 9999
    * The purpose is to exclude 0 from finding the minimum of every row in each matrix

* Line 5: Take the minimum along every row of every matrix
    ```
    9999    4    3
       2    1 9999
       1    1    3
    ```

    * Note: Matrix is collapsed to row. Row is collapsed to min element.

* Line 6: Revert all 9999 to 0.
    ```
    0    4    3
    2    1    0
    1    1    3
    ```

* Line 7: Perform cumulative sum along every row
    ```
    0    4    7    
    2    3    3    
    1    2    5
    ```

* Line 8: Shift the sum by one row
    ```
    5    0    4    
    7    2    3    
    3    1    2
    ```

* Line 9: Set first column to 0
    ```
    0    0    4    
    0    2    3    
    0    1    2
    ```

* Line 10: Repeat the shifted sum for every matrix
    ```
    0    0    0       0    0    0       0    0    0    
    0    0    0       2    2    2       1    1    1    
    4    4    4       3    3    3       2    2    2
    ```

* Line 11: Add the job makespans with the shifted sum

    ```
    0 0 0    0 0 2    2 1 0    
    4 4 4    2 2 1    1 0 1    
    0 0 3    0 0 0    3 0 0

    +

    0 0 0    0 0 0    0 0 0    
    0 0 0    2 2 2    1 1 1    
    4 4 4    3 3 3    2 2 2

    =

    0 0 0    0 0 2    2 1 0    
    4 4 4    4 4 3    2 1 2
    4 4 7    3 3 3    5 2 2
    ```


* Line 12: Set the non-eligible alternative to 0
    ```
    0    0    0       0    0    2       2    1    0    
    0    4    4       4    4    3       2    0    2    
    0    0    7       0    0    0       5    0    0
    ```

* Line 13: Set the indexes where operation end times are not zero to its corresponding element in operation end times
    * Operation end times
        ```
        2    2    2       0    0    0       0    0    0    
        4    4    4       0    0    0       0    0    0    
        0    0    0       0    0    0       0    0    0
        ```

    * Resulting operation lower bound
        ```
        2    2    2       0    0    2       2    1    0    
        4    4    4       4    4    3       2    0    2    
        0    0    7       0    0    0       5    0    0
        ```



