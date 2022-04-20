def nonDivisibleSubset(k, s):
    multiplicities = {number: 0 for number in range(k)}
    for num in s:
        multiplicities[num % k] += 1

    longest_set = 0 if multiplicities[0] == 0 else 1
    for i in range(1, (k - 1) // 2 + 1):
        longest_set += max(multiplicities[i], multiplicities[k - i])
    if k % 2 == 0:
        longest_set += 0 if multiplicities[k // 2] == 0 else 1
    return longest_set


# def biggerIsGreater(w):
#     prev = chr(0)
#     for i, letter in enumerate(w[::-1]):
#         if letter < prev:
#             break
#         elif i == len(w) - 1:
#             return "no answer"
#         prev = letter

#     split_index = len(w) - i - 1
#     new_word = w[:split_index]
#     new_word += w[split_index + 1] + w[split_index]
#     if len(new_word) < len(w):
#         new_word += ''.join(sorted(w[split_index + 2:]))
#     return new_word

def swap_letters(w, i, j):
    assert i <= j
    new_word = w[:i] + w[j] + w[i + 1:j] + w[i]
    if len(new_word) < len(w):
        new_word += w[j + 1:]
    return new_word


def get_smallest_greater_than(w, c):
    smallest = chr(255)
    for letter in w:
        if letter > c:
            smallest = min(letter, smallest)
    return smallest


def biggerIsGreater(w):
    prev = w[-1]
    word_size = len(w)
    for i, letter in enumerate(w[::-1]):
        if letter < prev:
            smallest_greater_than = get_smallest_greater_than(w[word_size - i:], letter)
            idx_smallest = word_size - i + w[word_size - i:].index(smallest_greater_than)
            break
        elif i == len(w) - 1:
            return "no answer"
        prev = letter

    idx_split = len(w) - i - 1
    w_tmp = swap_letters(w, idx_split, idx_smallest)
    new_word = w_tmp[:idx_split + 1]
    if len(new_word) < word_size:
        new_word += ''.join(sorted(w_tmp[idx_split + 1:]))
    return new_word


# words = ["abcdc", "abcedcba", "abc", "cba", "aaaaa", "zsdfm", "asdas", "asdasf", "asdasdzxc", "asdwkm"]
# for word in words:
#     print(word, biggerIsGreater(word))

def isKaprekar(num):
    d = len(str(num))
    if d == 1:
        return num == 1
    square_string = str(num**2)
    summed = int(square_string[:d]) + int(square_string[d:])
    return summed == num


def kaprekarNumbers(p, q):
    kaprekars = []
    for number in range(p, q + 1):
        if isKaprekar(number):
            kaprekars.append(number)
    if len(kaprekars) == 0:
        return "INVALID RANGE"
    else:
        return kaprekars


#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'queensAttack' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER k
#  3. INTEGER r_q
#  4. INTEGER c_q
#  5. 2D_INTEGER_ARRAY obstacles
#


def counter(x, k):
    houses = sorted(x)
    houses_diff = [houses[i + 1] - houses[i] for i in range(len(houses) - 1)]
    cumsum = 0
    counter = 0
    for diff in houses_diff:
        if diff > k:
            counter += 1
        cumsum += diff
        if cumsum >= 2 * k:
            counter += 1
            cumsum = 0

    return houses_diff, counter


def traverse(indexes, query):
    traversal_order = []

    def next_node(node, d=0):
        d += 1
        left_node, right_node = indexes[node - 1]
        if d % query == 0:
            if right_node != -1:
                next_node(right_node, d)
            indexes[node - 1] = [right_node, left_node]
            traversal_order.append(node)
            if left_node != -1:
                next_node(left_node, d)
        else:
            if left_node != -1:
                next_node(left_node, d)
            traversal_order.append(node)
            if right_node != -1:
                next_node(right_node, d)
    next_node(1)
    return traversal_order


def swapDepth(indexes, depth):
    node = 2**(depth - 1) - 1
    while node < 2**depth - 1 and node < len(indexes):
        children = indexes[node]
        indexes[node] = children[::-1]
        node += 1


def swapDepthRecursive(indexes, depth):
    def go_deeper(node, d=1):
        # if node == 1:
        #     breakpoint()
        d += 1
        left, right = indexes[node - 1]
        if d % depth == 0:
            indexes[node - 1] = [right, left]
        if left != -1:
            go_deeper(left, d)
        if right != -1:
            go_deeper(right, d)
    go_deeper(1, 0)


def swapNodes(indexes, queries):
    trees = []
    for query in queries:
        # for depth in range(query, len(indexes), query):
        # swapDepthRecursive(indexes, query)
        trees.append(traverse(indexes, query))
    return trees


indexes = [
    [2, 3],
    [4, -1],
    [5, -1],
    [6, -1],
    [7, 8],
    [-1, 9],
    [-1, -1],
    [10, 11],
    [-1, -1],
    [-1, -1],
    [-1, -1]]
queries = [2, 4]

# print(traverse(indexes))
print(swapNodes(indexes, queries))
