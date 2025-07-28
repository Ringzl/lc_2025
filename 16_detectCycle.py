class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:

        slow = head
        fast = head
        while fast != None:
            slow = slow.next

            if fast.next == None:
                return None
            fast = fast.next.next

            if slow == fast:
                cur = head
                while cur != slow:
                    cur = cur.next
                    slow = slow.next
                return cur
        return None