class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:

        # 前缀和
        n = len(nums)
        presum_dct = {0:1}

        presum = 0
        ans = 0
        for i in range(n):
            presum += nums[i]

            if presum - k in presum_dct:
                ans += presum_dct[presum- k]
            presum_dct[presum] = presum_dct.get(presum, 0) + 1
        
        return ans 