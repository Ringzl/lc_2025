class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0 for _ in range(n)]
        st = []
        for i in range(n):
            while st and temperatures[i] > temperatures[st[-1]]:
                ans[st[-1]] = i - st[-1]
                st.pop()
            st.append(i)
        return ans 