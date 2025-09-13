class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # 最后一次出现位置
        last_pos = [0 for _ in range(26)]

        for i,c in enumerate(s):
            last_pos[ord(c) - ord('a')] = i

        start = 0
        end = 0
        partition = []
        for i,c in enumerate(s):
            end = max(end, last_pos[ord(c) - ord('a')])

            if i == end:
                partition.append(end - start + 1)
                start = end + 1
        return partition