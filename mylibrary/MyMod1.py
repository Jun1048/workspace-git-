# 함수를 포함하는 모듈
def plus(a, b):
    return a + b

def minus(a, b):
    return a - b

# 이 파일을 직접 실행했을 때만 작동하는 테스트 코드
if __name__ == "__main__":
    print(plus(10, 20))
    print(minus(10, 20))