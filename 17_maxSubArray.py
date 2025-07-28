class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # pre = max(pre + nums[i], nums[i])

        presum = 0
        ans = nums[0]
        for num in nums:
            presum = max(presum + num, num)
            ans = max(presum, ans)
        return ans