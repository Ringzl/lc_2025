class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        p = head
        nums = []
        while p:
            nums.append(p.val)
            p = p.next
        
        nums.sort()

        p = head
        i = 0
        while p:
            p.val = nums[i]
            p = p.next
            i += 1

        return head 