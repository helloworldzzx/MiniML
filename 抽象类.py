from abc import ABC, abstractmethod

# 抽象父类
class Animal(ABC):
    # 抽象方法：无实现，强制子类重写
    @abstractmethod
    def speak(self):
        pass

# 子类必须实现speak
class Dog(Animal):
    def speak(self):
        print("汪汪汪")

class Cat(Animal):
    def speak(self):
        print("喵喵喵")

# 测试
d = Dog()
d.speak()  # 汪汪汪
c = Cat()
c.speak()  # 喵喵喵

# 报错：抽象类不能实例化
# a = Animal()