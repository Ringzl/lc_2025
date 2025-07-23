class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        # # 保存位置
        # char_dct = {}
        # n = len(s)
        # ans = 0
        # for i in range(n):

        #     # 存在重复，找到重复位置，并从字典中去除该位置前的字符
        #     if s[i] in char_dct:
        #         pos = char_dct[s[i]]
        #         for c in list(char_dct.keys()):
        #             if c in char_dct and char_dct[c] <= pos:
        #                 char_dct.pop(c)
            
        #     char_dct[s[i]] = i

        #     ans = max(ans, len(char_dct))

        # return ans
            
        left = 0
        right = 0
        n = len(s)

        char_set = set()
        ans = 0
        cnt = 0
        while right < n:
            if s[right] not in char_set:
                char_set.add(s[right])
                right += 1
                cnt += 1
            else:
                char_set.remove(s[left])
                left += 1
                cnt -= 1
            ans = max(ans, cnt)

        return ans

        