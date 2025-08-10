class Solution:
    def letterCombinations(self, digits: str) -> List[str]:

        nc_dct = {
            '2': 'abc',
            '3': 'def', 
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv', 
            '9': 'wxyz'
        }

        if digits == "":
            return []
        
        ans = []
        self.path = ""
        n = len(digits)
        
        def backtrack(index):
            if index == n:
                ans.append(self.path)
            else:
                for c in nc_dct[digits[index]]:
                    self.path += c
                    backtrack(index+1)
                    self.path = self.path[:-1]

        backtrack(0)
        return ans