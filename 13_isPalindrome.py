class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:

        st = []
        cur = head
        while cur != None:
            st.append(cur.val)
            cur = cur.next
        
        cur = head
        while cur != None:
            if cur.val != st[-1]:
                return False
            st.pop()
            cur = cur.next

        return True