class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        prev = None
        cur = head
        while cur != None:
            tmp = cur.next
            cur.next = prev
            prev = cur
            cur = tmp
        
        return prev