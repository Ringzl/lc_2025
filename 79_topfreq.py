class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        
        n = len(nums)
        freq_dct = {}
        for i in range(n):
            freq_dct[nums[i]] = freq_dct.get(nums[i], 0) + 1
        
        # freq_set = sorted(freq_dct.items(), key = lambda x:x[1], reverse= True)
        # ans = []
        # for i in range(k):
        #     ans.append(freq_set[i][0])

        # 用大小为k的最小堆保存
        heap_k = []
        for key in freq_dct:
            heapq.heappush(heap_k, [cnt_dct[key], key])

            if len(heap_k) > k:
                heapq.heappop(heap_k)
        ans = []
        while heap_k:
            ans.append(heapq.heappop(heap_k)[1])

    
        return ans