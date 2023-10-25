"""
Suppose you must detennine whether a given number between 3 and 100.
inclusive. is a prime. Recall that an integer N 2 is prime if its only factors
are I and itself. So 17 and 23 are prime whereas 33 is not. because it is the
product of 3 and 11. Assume that you must solve this problem without benefit
of a computer or pocket calculator. Your first attempt at a solution, using the
generate-and-test approach. might look like the following pseudocode:
"""

def is_prime_to_100(n):
  for num in range(2, n):
    if n % num == 0:
      return False
  return True

for num in range(3,101):
  print(num, is_prime_to_100(num))

'''
Ouput:
3 True
4 False
5 True
6 False
7 True
8 False
9 False
10 False
11 True
12 False
13 True
14 False
15 False
16 False
17 True
18 False
19 True
20 False
21 False
22 False
23 True
24 False
25 False
26 False
27 False
28 False
29 True
30 False
31 True
32 False
33 False
34 False
35 False
36 False
37 True
38 False
39 False
40 False
41 True
42 False
43 True
44 False
45 False
46 False
47 True
48 False
49 False
50 False
51 False
52 False
53 True
54 False
55 False
56 False
57 False
58 False
59 True
60 False
61 True
62 False
63 False
64 False
65 False
66 False
67 True
68 False
69 False
70 False
71 True
72 False
73 True
74 False
75 False
76 False
77 False
78 False
79 True
80 False
81 False
82 False
83 True
84 False
85 False
86 False
87 False
88 False
89 True
90 False
91 False
92 False
93 False
94 False
95 False
96 False
97 True
98 False
99 False
100 False
'''