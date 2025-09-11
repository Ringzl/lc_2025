class Solution:
    def jump(self, nums: List[int]) -> int:
        max_pos = 0
        n = len(nums)

        ans = 0
        end = 0
        for i in range(n-1):
            if i <= max_pos:
                max_pos = max(max_pos, i + nums[i])

            if i == end:
                end = max_pos
                ans += 1
        return ans