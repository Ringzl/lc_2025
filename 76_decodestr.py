class Solution:
    def decodeString(self, s: str) -> str:
        st = []

        i = 0
        while i < len(s):

            cur = s[i]

            if cur.isdigit():
                # 获取数字
                num = ''
                while s[i].isdigit():
                    num += s[i]
                    i += 1
                st.append(num)
            elif cur.isalpha() or cur == '[':
                st.append(s[i])
                i += 1
            else:
                i += 1
                sub = []
                while st[-1] != '[':
                    sub.append(st[-1])
                    st.pop()
                sub = sub[::-1]
                sub = ''.join(sub)
                
                # 左括号出栈
                st.pop()

                # 栈顶为次数
                rep = int(st[-1])
                st.pop()

                t = rep * sub
                st.append(t)
                
                
        return ''.join(st)