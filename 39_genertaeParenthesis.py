class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        
        ans = []
        self.path = ""
        def backtrack(left, right):
            if (left == n and right == n):
                ans.append(self.path)
                return
            
            if left < right or left > n:
                return 
            
            self.path += '('
            backtrack(left + 1, right)
            self.path = self.path[:-1]

            self.path += ')'
            backtrack(left, right+1)
            self.path = self.path[:-1]

        backtrack(0,0)

        return ans