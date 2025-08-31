class DLinkedNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = dict()

        self.head = DLinkedNode()
        self.tail = DLinkedNode()

        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0
        
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        
        # key 存在，移到头部
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key not in self.cache:
            # key 不存在，创建一个新节点
            node = DLinkedNode(key, value)

            # 添进hash表,添加到表头
            self.cache[key] = node
            self.addToHead(node)
            self.size += 1

            if self.size > self.capacity:
                # 删除表尾
                removed = self.removeTail()
                self.cache.pop(removed.key)
                self.size -= 1
        else:
            # key 存在，修改value值， 移到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
    
    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node