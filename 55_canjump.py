class Solution:
    def canJump(self, nums: List[int]) -> bool:

        max_pos = 0
        n = len(nums)
        for i in range(n):

            if i <= max_pos:
                max_pos = max(max_pos, i + nums[i])

                if max_pos >= n-1:
                    return True

        return False