class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:

        intervals.sort(key=lambda x:x[0])

        n = len(intervals)
        ans = [intervals[0]]

        for i in range(1, n):
            if ans[-1][1] < intervals[i][0]:
                ans.append(intervals[i])
            else:
                # 合区间
                ans[-1][1] = max(intervals[i][1], ans[-1][1])
        return ans