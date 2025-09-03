class Node:
    def __init__(self):
        self.children = [None] * 26 
        self.isEnd = False 

class Trie:

    def __init__(self):
        self.root = Node()
        
    def insert(self, word: str) -> None:
        # 从根节点开始构造word对应的路径节点
        node = self.root

        for c in word:
            char_id = ord(c) - ord('a')
            if not node.children[char_id]:
                node.children[char_id] = Node()
            node = node.children[char_id]
        node.isEnd = True
        

    def search(self, word: str) -> bool:
        node = self.root
        for c in word:
            node = node.children[ord(c) - ord('a')]
            if not node:
                return False
        return node.isEnd

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for c in prefix:
            node = node.children[ord(c) - ord('a')]
            if not node:
                return False
        return True