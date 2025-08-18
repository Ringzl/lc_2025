class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        left, right = head, head

        for i in range(n):
            right = right.next

        pre = None
        while right != None:
            pre = left
            left = left.next
            right = right.next

        if pre != None:
            pre.next = left.next
            return head
        elif left != None:
            return left.next
        else:
            return None