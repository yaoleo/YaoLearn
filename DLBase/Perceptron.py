from functools import reduce


class VectorOp(object):
    """
    calcuate vector
    """

    @staticmethod
    def dot(x, y):
        """
        计算两个向量x和y的内积
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]按元素相乘
        # 变成[x1*y1, x2*y2, x3*y3]
        # 然后利用reduce求和
        return reduce(lambda a, b: a + b, VectorOp.element_multiply(x, y), 0.0)

    @staticmethod
    def element_multiply(x, y):
        """
        将两个向量x和y按元素相乘
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        # 变成[(x1,y1),(x2,y2),(x3,y3),...]
        # 然后利用map函数计算[x1*y1, x2*y2, x3*y3]
        return list(map(lambda x_y: x_y[0] * x_y[1], zip(x, y)))

    @staticmethod
    def element_add(x, y):
        """
        将两个向量x和y按元素相加
        """
        # 首先把x[x1,x2,x3...]和y[y1,y2,y3,...]打包在一起
        # 变成[(x1,y1),(x2,y2),(x3,y3),...]
        # 然后利用map函数计算[x1+y1, x2+y2, x3+y3]
        return list(map(lambda x_y: x_y[0] + x_y[1], zip(x, y)))

    @staticmethod
    def scala_multiply(v, s):
        """
        将向量v中的每个元素和标量s相乘
        """
        return map(lambda e: e * s, v)


class Perceptron(object):

    def __init__(self, input_num, activator):
        """
        init perception
        :param input_num: number of the input paragram
        :param activator: activate function double->double
        """
        self.activator = activator
        self.weights = [0.0] * input_num
        self.bias = 0.0

    def predict(self, input_vec):
        """
        predict the result with input_vec
        :param input_vec:
        :return:
        """
        return self.activator(VectorOp.dot(input_vec, self.weights) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        """
        :param input_vecs: 训练数据向量
        :param labels: 对应label
        :param iteration: 训练轮数
        :param rate: 学习率
        :return:
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        """
        one iteration to run all data
        :param input_vecs:
        :param labels:
        :param rate:
        :return:
        """
        samples = zip(input_vecs, labels)     # 一个样本是一个(input_vec, label)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)  # 激素按感知器在当前权重下的输出
            self._update_weights(input_vec, output, label, rate) # 更新权重

    def _update_weights(self,input_vec, output, label, rate):
        """

        :param input_vec:
        :param output:
        :param label:
        :param rate:
        :return:
        """
        delta = label - output
        # 更新weights 把input_vec[x1,x2,x3,...]向量中的每个值乘上delta，得到每个权重更新
        self.weights = VectorOp.element_add(self.weights,
                                            VectorOp.scala_multiply(input_vec, rate * delta))
        self.bias += rate * delta # 更新bias

    def __str__(self):
        """
        打印权重和偏置项
        :return:
        """
        return 'weights\t:%s\n bias\t:%f\n' % (self.weights,self.bias)

def f(x):
    """
    定义激活函数f
    """
    return 1 if x > 0 else 0

def get_training_dataset():
    """
    基于and真值表构建训练数据
    """
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perception(iteration):
    """
    使用and真值表训练感知器
    """
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, iteration, 0.1)
    # 返回训练好的感知器
    return p
    pass

if __name__ == '__main__':
    # 训练轮数
    for iteration in range(10):
        and_perception = train_and_perception(iteration) # 训练
        print(and_perception) # 打印训练的权重
    # 测试
        print('1 and 1 = %d' % and_perception.predict([1, 1]))
        print('0 and 0 = %d' % and_perception.predict([0, 0]))
        print('1 and 0 = %d' % and_perception.predict([1, 0]))
        print('0 and 1 = %d' % and_perception.predict([0, 1]))