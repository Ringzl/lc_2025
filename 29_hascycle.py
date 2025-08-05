class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:

        slow = head
        fast = head

        while fast != None:
            slow = slow.next

            if fast.next == None:
                return False

            fast = fast.next.next

            if slow == fast:
                return True