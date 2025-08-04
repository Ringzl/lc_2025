class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []

        used = [False for _ in range(len(nums))]

        def backtrack(nums, used):

            if len(path) == len(nums):
                res.append(path.copy())
                return

            for i in range(len(nums)):
                if used[i] == True:
                    continue
                used[i] = True
                path.append(nums[i])
                backtrack(nums, used)
                path.pop()
                used[i] = False
            
        backtrack(nums, used)
        return res