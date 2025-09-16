class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # color_cnt = dict()
        # for c in nums:
        #     color_cnt[c] = color_cnt.get(c, 0) + 1
        
        # start = 0
        # for i in [0,1,2]:
        #     if i not in color_cnt:
        #         continue
        #     nums[start: start+color_cnt[i]] = [i] * color_cnt[i]
        #     start = start+color_cnt[i]
        idx0, idx1, idx2 = 0, 0, 0
        for n in nums:

            if n == 0:
                nums[idx2] = 2
                idx2 += 1
                nums[idx1] = 1
                idx1 += 1
                nums[idx0] = 0
                idx0 += 1
                
            elif n == 1:
                nums[idx2] = 2
                idx2 += 1
                nums[idx1] = 1
                idx1 += 1
            else:
                nums[idx2] = 2
                idx2 += 1
        return nums