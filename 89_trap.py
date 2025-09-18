class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)

        res = 0
        st = []
        st.append(0)

        for i in range(1, n):
            while(len(st) > 0 and height[i] > height[st[-1]]):
                mid = st.pop()
                if len(st) > 0:
                    h = min(height[st[-1]], height[i]) - height[mid]
                    w = i - st[-1] -1
                    res += h * w
            st.append(i)
        
        return res