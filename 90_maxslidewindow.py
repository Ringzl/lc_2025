class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        window_heap = []
        for i in range(k):
            heapq.heappush(window_heap, (-nums[i], i))
        ans = [-window_heap[0][0]]
        for i in range(k,n):
            heapq.heappush(window_heap, (-nums[i], i))
            while window_heap[0][1] <= i-k:
                heapq.heappop(window_heap)

            ans.append(-window_heap[0][0])

        return ans