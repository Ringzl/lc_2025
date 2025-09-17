class Solution:
    def findDuplicate(self, nums: List[int]) -> int:

        num_cnt = {}
        for n in nums:
            num_cnt[n] = num_cnt.get(n, 0) + 1
            if num_cnt[n] > 1:
                return n