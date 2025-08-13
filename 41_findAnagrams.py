class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:

        p = sorted(p)
        n = len(s)
        np = len(p)

        ans = []
        for i in range(n-np+1):
            tmp = sorted(s[i:i+np])
            if tmp == p:
                ans.append(i)
        return ans