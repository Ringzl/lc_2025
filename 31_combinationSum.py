class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        ans = []

        path = []
        n = len(candidates)
        def backtrack(index):
            if sum(path) >= target:
                if sum(path) == target:
                    ans.append(path.copy())
                return 
            
            for i in range(index,n):
                path.append(candidates[i])
                backtrack(i)
                path.pop()

        backtrack(0)

        return ans