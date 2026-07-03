from abc import ABC, abstractmethod
#ABC：抽象基类父类，自定义抽象类必须继承它
#abstractmethod：装饰器，标记方法为抽象方法，只有声明，没有实现逻辑
#子类继承并必须实现全部抽象方法
class Module(ABC):
    """
    所有网络层的父类。
    每一个层（Linear、Conv、ReLU……）
    都必须继承 Module。

    子类需要实现：
        forward()
        backward()

    如果该层有参数（例如 Linear、Conv），
    还需要返回参数列表。
    """

    @abstractmethod
    def forward(self, x):
        """
        前向传播
        输入：x（ndarray）
        输出：当前层的输出
        """
        pass

    @abstractmethod
    def backward(self, grad_output):
        """
        反向传播

        输入：grad_output（loss对当前层输出的梯度）
        输出：loss对当前层输入的梯度
        """
        pass

    def parameters(self):
        """
        返回需要训练的参数。

        比如：ReLU没有参数，默认返回空列表。
             Linear、Conv需要重写这个函数。
        """
        return []