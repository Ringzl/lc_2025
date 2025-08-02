class Solution:
    def isValid(self, s: str) -> bool:

        st = []

        dct = {
            '(' : ')',
            '[' : ']',
            '{' : '}'
        }

        for char in s:
            if char in dct:
                st.append(char)  
            else:
                if len(st) > 0:
                    if char == dct[st[-1]]:
                        st.pop()
                    else:
                        return False
                else:
                    return False

        return False if len(st) > 0 else True