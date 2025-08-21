class Solution:
    def singleNumber(self, nums: List[int]) -> int:

        # num_cnt = {}
        # n = len(nums)
        # for i in range(n):
        #     num_cnt[nums[i]] = num_cnt.get(nums[i], 0) + 1
        
        # for num in num_cnt:
        #     if num_cnt[num] == 1:
        #         return num
        ans = 0
        for num in nums:
            ans ^= num
        return ans