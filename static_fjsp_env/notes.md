# Calculate end time lowerbound
Durations
```
1 3 2   0 0 2   2 1 0
0 2 1   2 2 1   1 0 1
0 0 3   0 0 0   3 0 0
```

Operation End Times
```
2 2 2   0 0 0   0 0 0
4 4 4   0 0 0   0 0 0
0 0 0   0 0 0   0 0 0
```

Job Makespans
```
2 2 2   0 2 1   0 0 3
0 0 2   2 2 1   0 0 0
2 1 0   1 0 1   3 0 0
```

End Time lowerbound
```
2 2 2   0 0 2   2 1 0
4 4 4   4 4 3   2 0 2
0 0 7   0 0 0   5 0 0
```


You pick the O11 on Machine 3. (Duration = 2)


# Question: How to calculate lowerbound?
Let's say we have following FJSP problem.

```
Job 1   Job 2   Job 3

1 3 2   0 0 2   2 1 0
0 2 1   2 2 1   1 0 1
0 0 3   0 0 0   3 0 0
```

For instance, $x_{001} = 3$ .
The first matrix, first row, second column has a value of 3.
This means: Job 1 (index 0), Operation 1 (index 0), processed on Machine 2 (index 1) has a duration of 3.


What is the lowerbound for each operation?

We define lowerbound, LB as:

LB of Operation 1 = LB of Operation 0 + Minimum duration out of all alternative machines

If Operation 0 has been dispatched, its LB is its end time.
Otherwise, its LB is its previous LB + Minimum duration out of all alternative machines.

In these case that Operation 0 is the first operation, its LB is simply its minimum duration of all alternatives.

Let's say that none of any operations have been dispatched.

Thus we have the following LB:
```
1 2 1
2 3 2
5 0 5
```

Note that we use 0 as a placeholder for non-existent operations.

Calculation is as follows:

```
LB of Op 1 = Min duration = 1
LB of Op 2 = LB of Op 1 + Min duration = 1 + 1 = 2
LB of Op 3 = LB of Op 2 + Min duration = 2 + 3 = 5

so on... until Op 9
```

## Step 1: Dispatch Operation 11 on Machine 2
Let's say we just dispatched Operation 21. What's the updated lowerbound? How to calculate them?


