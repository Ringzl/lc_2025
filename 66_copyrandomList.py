class Solution:
    def __init__(self):
        self.cacheNode = dict()
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':

        if head == None:
            return 

        if head not in self.cacheNode:
            headNew = Node(head.val)
            self.cacheNode[head] = headNew

            headNew.next = self.copyRandomList(head.next)
            headNew.random = self.copyRandomList(head.random)

        return self.cacheNode[head]