class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        # n = len(nums)

        # ans = [0 for _ in range(n)]
        # for i in range(n):
        #     ans[(i+k)%n] = nums[i]
        
        # for i in range(n):
        #     nums[i] = ans[i]

        # 翻转
        n = len(nums)
        k = k % n
        if k != 0:
            tmp = nums[-k:]
            nums[k-n:] = nums[0:n-k]
            nums[0:k] = tmp
        