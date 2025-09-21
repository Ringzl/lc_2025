class Solution:
    def minWindow(self, s: str, t: str) -> str:


        need = {}
        for c in t:
            need[c] = need.get(c, 0) + 1

        
        missing = len(t)
        start = 0 
        sub_len = len(s) + 1
        l = 0 # 窗口左端

        for r in range(len(s)):
            c = s[r]
            need[c] = need.get(c, 0)
            if need[c] > 0:
                missing -= 1
            need[c] -= 1

            if not missing:
                while need[s[l]] < 0: # 当前窗口左端字符冗余
                    need[s[l]] += 1
                    l += 1

                if r - l + 1 < sub_len:
                    sub_len = r - l + 1
                    start = l
        
        return "" if sub_len == len(s) + 1 else s[start:start+sub_len]