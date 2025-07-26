class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        cnt_dct = {}
        n = len(nums)
        max_cnt = 0 
        ans = 0
        for i in range(n):
            cnt_dct[nums[i]] = cnt_dct.get(nums[i], 0) + 1
            if cnt_dct[nums[i]] > max_cnt:
                max_cnt = cnt_dct[nums[i]]
                ans = nums[i]
        return ans 