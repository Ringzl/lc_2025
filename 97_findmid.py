class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m = len(nums1)
        n = len(nums2)

        i,j = 0, 0
        med1, med2 = 0,0
        k = 0
        ans = []
        while i < m and j < n:
            if nums1[i] < nums2[j]:
                tmp = nums1[i]
                i += 1
            else:
                tmp = nums2[j]
                j += 1

            if k == int((m+n)//2-1):
                med1 = tmp
            elif k == (m+n)//2:
                med2 = tmp

            k += 1
    
        
        while i < m:
            tmp = nums1[i]
            
            if k == int((m+n)//2-1):
                med1 = tmp
            elif k == (m+n)//2:
                med2 = tmp
            k += 1
            i += 1

        
        while j < n:
            tmp = nums2[j]
            
            if k == int((m+n)//2-1):
                med1 = tmp
            elif k == (m+n)//2:
                med2 = tmp
            k += 1
            j += 1
        print(i,j,k,med1, med2)
        return (med1 + med2) / 2 if (m + n) % 2 == 0 else med2