class MinStack:

    def __init__(self):
        self.st = []
        self.min_st = [math.inf]

    def push(self, val: int) -> None:
        self.st.append(val)
        self.min_st.append(min(self.min_st[-1], val))

    def pop(self) -> None:
        self.st.pop()
        self.min_st.pop()
        

    def top(self) -> int:
        return self.st[-1]
        

    def getMin(self) -> int:
        return self.min_st[-1]