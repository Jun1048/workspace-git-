# 두 가지 기능을 수행하는 클래스
class Calc:
    def plus(self, x, y):
        return x + y
    
    def minus(self, x, y):
        return x - y
    
# 모듈안에 클래스의 기능을 부여받은 객체 생성.
my_calc = Calc()

# 테스트 코드
if __name__ == "__main__":
    print(my_calc.plus(10, 20), my_calc.minus(10, 20));