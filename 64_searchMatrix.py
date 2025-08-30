class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        m, n = len(matrix), len(matrix[0])

        # for i in range(m):

        #     l, r = 0, n-1
        #     while l <= r:
        #         mid = l + (r-l)//2

        #         if matrix[i][mid] == target:
        #             return True
        #         elif matrix[i][mid] > target:
        #             r = mid - 1
        #         else:
        #             l = mid + 1

        # return False

        r, c = 0, n-1
        while r < m and c >= 0:

            if matrix[r][c] == target:
                return True
            elif matrix[r][c] < target:
                r += 1
            else:
                c -= 1
        
        return False