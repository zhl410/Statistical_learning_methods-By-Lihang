import numpy as np
from heapq import heappush, heappop

def minkowski_distance_p(x, y, p=2):
    """
    Parameters:
    ------------
    x: (M, K) array_like 
    y: (M, K) array_like
    p:  float, 1<= p <= infinity
    ------------
    计算M个K维向量的距离，但是该距离没有开p次根号
    """
    #把输入的array_like类型的数据转换成numpy中的ndarray
    x = np.asarray(x)
    y = np.asarray(y)
    ##axis=-1沿最后一个坐标轴，0，1沿着第一，第二个坐标轴
    if p == np.inf:
        return np.max(np.abs(x-y), axis=-1)
    else:
        return np.sum(np.abs(x-y)**p, axis=-1)
    
def minkowski_distance(x, y, p=2):
    if p==np.inf:
        return minkowski_distance_p(x, y, np.inf)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)
		
	
"""
关于kd树划分的几点说明：
1. 切分的粒度不用那么细，叶子结点上可以保留多个值，规定叶子结点的大小就好了。
2. 对划分坐标轴的选取可以采用维度最大间隔的方法,使用第d个坐标，该坐标点的最大值和最小值差最大。
3. 使用平均值而不是采用中位数作为切分点
"""
class KDTree(object):
    def __init__(self, data, leafsize=10):
        """
        
        """
        self.data= np.asarray(data)
        self.n, self.m = self.data.shape
        self.leafsize = leafsize
        if self.leafsize < 1:
            raise ValueError("leafsize must be at least 1")
        self.maxes = np.max(self.data, axis=0)
        self.mins = np.min(self.data, axis=0)
        self.tree = self.__build(np.arange(self.n), self.maxes, self.mins)
    
    """
    定义结点类，作为叶子结点和内部结点的父类
    是必须重写比较运算符吗？？？
    """
    class node(object):
        def __it__(self, other):
            return id(self) < id(other)
        def __gt__(self, other):
            return id(self) > id(other)
        def __le__(self, other):
            return id(self) <= id(other)
        def __ge__(self, other):
            return id(self) >= id(other)
        def __eq__(self, other):
            return id(self) == id(other)
        
    """
    定义叶子结点
    """
    class leafnode(node):
        def __init__(self, idx):
            self.idx = idx
            self.children = len(self.idx)
    """
    定义内部结点
    """
    class innernode(node):
        def __init__(self, split_dim, split, less, greater):
            """
            split_dim: 在某个维度上进行的划分
            split：在该维度上的划分点
            """
            self.split_dim = split_dim
            self.split = split
            self.less = less
            self.greater = greater
            self.children = less.children+greater.children
            
    """
    仅开头带双下划线__的命名 用于对象的数据封装，以此命名的属性或者方法为类的私有属性或者私有方法。
    如果在外部直接访问私有属性或者方法,是不可行的，这就起到了隐藏数据的作用。
    但是这种实现机制并不是很严格，机制是通过自动"变形"实现的，类中所有以双下划线开头的名称__name都会自动变为"_类名__name"的新名称。
    使用"_类名__name"就可以访问了，如._KDTree__build()。同时机制可以阻止继承类重新定义或者更改方法的实现。
    
    """
    def __build(self, idx, maxes, mins):
        if len(idx) <= self.leafsize:
            return KDTree.leafnode(idx)
        else:
            #在第d维上进行划分，选自第d维的依据是该维度的间隔最大
            d = np.argmax(maxes-mins)
            #第d维上的数据
            data = self.data[idx][d]
            #第d维上的区间端点
            maxval, minval = maxes[d], mins[d]
        if maxval == minval:
            #所有的点值都相同
            return KDTree.leafnode(idx)
        """
         Splitting Methods
        sliding midpoint rule;
        see Maneewongvatana and Mount 1999"""
        split = (maxval + minval) / 2
        #分别返回小于等于，大于split值的元素的索引
        less_idx = np.nonzero(data <= split)[0] 
        greater_idx = np.nonzero(data>split)[0]
        #对于极端的划分情况进行调整
        if len(less_idx) == 0:
                split = np.min(data)
                less_idx = np.nonzero(data <= split)[0]
                greater_idx = np.nonzero(data > split)[0]
        if len(greater_idx) == 0:
                split = np.max(data)
                less_idx = np.nonzero(data < split)[0]
                greater_idx = np.nonzero(data >= split)[0]
        if len(less_idx) == 0:
                # _still_ zero? all must have the same value
            if not np.all(data == data[0]):
                raise ValueError("Troublesome data array: %s" % data)
            split = data[0]
            less_idx = np.arange(len(data)-1)
            greater_idx = np.array([len(data)-1])
        #递归调用左边和右边
        lessmaxes = np.copy(maxes)
        lessmaxes[d] = split
        greatermins = np.copy(mins)
        greatermins[d] = split
        return KDTree.innernode(d, split,
                    self.__build(idx[less_idx],lessmaxes,mins),
                    self.__build(idx[greater_idx],maxes,greatermins))
    
    def query(self, x, k=1, p=2, distance_upper_bound=np.inf):
        x = np.asarray(x)
        #距离下界，形象化的思考一下哈，点x出现在mins和maxes的位置分三种情况
        side_distances = np.maximum(0,np.maximum(x-self.maxes,self.mins-x))
        if p != np.inf:
            side_distances **= p
            min_distance = np.sum(side_distances)
        else:
            min_distance = np.amax(side_distances)
            
        if p != np.inf and distance_upper_bound != np.inf:
            distance_upper_bound = distance_upper_bound**p
    
        q, neighbors = [(min_distance, tuple(side_distances), self.tree)], []
        """
        q: 维护搜索的优先队列 
        # entries are:
        (minimum distance between the cell and the target, 
        distances between the nearest side of the cell and the target, 
        the head node of the cell)
        neighbors: priority queue for the nearest neighbors
        用于保存k近邻结果的优先队列，heapq默认是最小堆，为了立即能够得到队列中点的最大距离来更新距离上界upper bound，可以存储距离的相反数。
        #entries are (-distance**p, index)
        
        """

        while q:
            min_distance, side_distances, node = heappop(q)
            if isinstance(node, KDTree.leafnode):
                # 对于叶子结点，就一个个暴力排除
                data = self.data[node.idx]
                # 把x沿x-轴扩充，然后和叶子结点上的点比较大小
                ds = minkowski_distance_p(data,x[np.newaxis,:],p)
                
                for i in range(len(ds)):
                    if ds[i] < distance_upper_bound:
                        if len(neighbors) == k:
                            heappop(neighbors)
                        heappush(neighbors, (-ds[i], node.idx[i]))
                        if len(neighbors) == k: #更新距离上界
                            distance_upper_bound = -neighbors[0][0]
            else:
                # we don't push cells that are too far onto the queue at all,
                # but since the distance_upper_bound decreases, we might get
                # here even if the cell's too far
                if min_distance > distance_upper_bound:
                    # since this is the nearest cell, we're done, bail out
                    break
                # compute minimum distances to the children and push them on
                if x[node.split_dim] < node.split:
                    near, far = node.less, node.greater
                else:
                    near, far = node.greater, node.less
                # near child is at the same distance as the current node
                heappush(q,(min_distance, side_distances, near))
                
                # 对于far child需要进行距离判断，用新的距离替代原来的距离，然后和距离上界比较 
                sd = list(side_distances)
                if p == np.inf:
                    min_distance = max(min_distance, abs(node.split-x[node.split_dim]))
                elif p == 1:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])
                    min_distance = (min_distance - side_distances[node.split_dim]) + sd[node.split_dim]
                else:
                    sd[node.split_dim] = np.abs(node.split-x[node.split_dim])**p
                    min_distance = (min_distance - side_distances[node.split_dim]) + sd[node.split_dim]

                if min_distance <= distance_upper_bound:
                    heappush(q,(min_distance, tuple(sd), far))

        if p == np.inf:
            return sorted([(-d,i) for (d,i) in neighbors])
        else:
            return sorted([((-d)**(1./p),i) for (d,i) in neighbors])