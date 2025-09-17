import torch
import random

from softer.data_structures.softarray import SoftArray
from softer.control.ifelse import IfElse
from softer.comparison.softgt import SoftGt

k = 8

def bubble_sort(arr):
    # Define control flow and condition
    ifelse = IfElse()
    gt = SoftGt(k=k)

    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            # if arr[j] > arr[j + 1], then swap them
            condition = gt(arr[j], arr[j + 1])

            arr[j] = ifelse(condition, arr[j] + arr[j + 1], arr[j])
            arr[j + 1] = ifelse(condition, arr[j] - arr[j + 1], arr[j + 1])
            arr[j] = ifelse(condition, arr[j] - arr[j + 1], arr[j])

    return arr

if __name__ == "__main__":
    # Shuffle some numbers
    nums = list(range(-100, 100, 10))
    random.shuffle(nums)
    data = torch.tensor(nums, dtype=torch.float64)

    # Make a SoftArray
    arr = SoftArray(n=len(data), data=data, k=k)

    # Sort
    print(bubble_sort(arr))
