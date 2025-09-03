class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        
        # 建图
        edges = [[] for _ in range(numCourses)]
        indegree = [0 for _ in range(numCourses)]

        for info in prerequisites:
            edges[info[1]].append(info[0])
            indegree[info[0]] += 1

        q = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                q.appendleft(i)
        
        visited = 0
        while q:
            visited += 1
            u = q.pop()

            for v in edges[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    q.appendleft(v)
        
        return visited == numCourses