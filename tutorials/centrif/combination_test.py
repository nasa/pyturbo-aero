from itertools import combinations, combinations_with_replacement

A = ['A','B','C','D'] # stuff to combine
N = 10 # Number of placements
total_combinations = []
for i in range(3):
    combos = list(combinations(A,i))
    temp = [cc for c in combos for cc in c]
    total_combinations.extend(temp)
