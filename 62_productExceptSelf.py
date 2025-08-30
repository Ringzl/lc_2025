class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        n = len(nums)

        # ans = []
        # for i in range(n):
        #     tmp = 1
        #     for j in range(n):
        #         if j != i:
        #             tmp *= nums[j]
        #     ans.append(tmp)
        # return ans

        L = [1 for _ in range(n)] # 左侧乘积
        for i in range(1, n):
            L[i] = nums[i-1] * L[i-1]

        R = [1 for _ in range(n)] # 右侧乘积
        for i in range(n-2, -1, -1):
            R[i] = nums[i+1] * R[i+1]
        
        ans = []
        for i in range(n):
            ans.append(L[i] * R[i])
        
        return ans