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

