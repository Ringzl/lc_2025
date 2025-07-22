class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dct = {}
        for s in strs:
            sorted_s = ''.join(sorted(s))
            dct.setdefault(sorted_s, []).append(s)
        
        ans = []
        for key in dct:
            ans.append(dct[key])

        return ans        