class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:

        n = len(nums)
        ans = []
        path = []

        def backtrack(index):
            ans.append(path.copy())

            for i in range(index, n):
                path.append(nums[i])
                backtrack(i+1)
                path.pop()

        backtrack(0)
        return ans