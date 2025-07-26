class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:

        nums_set = set(nums)
        ans = 0
        for num in nums_set:
            if num-1 not in nums_set:
                cur_num = num
                cnt = 1

                while cur_num + 1 in nums_set:
                    cur_num += 1
                    cnt += 1
                ans = max(ans, cnt)
        
        return ans   