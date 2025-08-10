class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        m, n = len(matrix), len(matrix[0])
        left = 0 
        right = m * n -1

        while left <= right:

            mid = left + (right - left) // 2
            x = mid // n
            y = mid % n

            if matrix[x][y] == target:
                return True

            elif matrix[x][y] < target:
                left = mid + 1
            else:
                right = mid - 1

        return False