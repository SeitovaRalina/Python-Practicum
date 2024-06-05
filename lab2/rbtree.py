from collections import deque
import pygraphviz as pgv


BLACK = "BLACK"
RED = "RED"


class Node:
    def __init__(self):
        self.value = None
        self.left = None
        self.right = None
        self.parent = None
        self.color = BLACK


class RedBlackTree:

    def __init__(self):
        self.NIL = Node()
        self.root = self.NIL

    def insert(self, value):
        """
        Вставляет новый узел с указанным значением в красно-черное дерево.
        """
        def create_node(_value):
            """
            Создает новый узел красного цвета для вставки в красно-черное дерево
            """
            node = Node()
            node.left = self.NIL
            node.right = self.NIL
            node.value = _value
            node.color = RED
            return node
        current_node = self.root
        parent = self.NIL
        while current_node != self.NIL:
            parent = current_node
            if value < current_node.value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        new_node = create_node(value)
        new_node.parent = parent

        if parent == self.NIL:
            self.root = new_node
        elif value < parent.value:
            parent.left = new_node
        else:
            parent.right = new_node

        self._fix_insert(new_node)

    def _fix_insert(self, node):
        """
        Корректирует дерево после вставки нового узла, чтобы поддерживать свойства красно-черного дерева.
        """
        # папаша - красный
        while node.parent.color == RED:
            # отец — левый ребенок деда
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == RED:
                    node.parent.color = BLACK
                    uncle.color = BLACK
                    node.parent.parent.color = RED
                    node = node.parent.parent
                else:
                    # новый узел - правый сын
                    if node == node.parent.right:
                        node = node.parent
                        self.__rotate_left(node)
                    node.parent.color = BLACK
                    node.parent.parent.color = RED
                    self.__rotate_right(node.parent.parent)
            # отец — правый ребенок деда
            else:
                uncle = node.parent.parent.left
                if uncle.color == RED:
                    node.parent.color = BLACK
                    uncle.color = BLACK
                    node.parent.parent.color = RED
                    node = node.parent.parent
                else:
                    # новый узел - левый сын
                    if node == node.parent.left:
                        node = node.parent
                        self.__rotate_right(node)
                    node.parent.color = BLACK
                    node.parent.parent.color = RED
                    self.__rotate_left(node.parent.parent)
        self.root.color = BLACK

    def __rotate_left(self, node):
        """
        Выполняет левый поворот относительно указанного узла.
        """
        right_child = node.right
        node.right = right_child.left
        if right_child.left != self.NIL:
            right_child.left.parent = node
        right_child.parent = node.parent
        if node.parent == self.NIL:
            self.root = right_child
        elif node == node.parent.left:
            node.parent.left = right_child
        else:
            node.parent.right = right_child
        right_child.left = node
        node.parent = right_child

    def __rotate_right(self, node):
        """
        Выполняет правый поворот относительно указанного узла.
        """
        left_child = node.left
        node.left = left_child.right
        if left_child.right != self.NIL:
            left_child.right.parent = node
        left_child.parent = node.parent
        if node.parent == self.NIL:
            self.root = left_child
        elif node == node.parent.right:
            node.parent.right = left_child
        else:
            node.parent.left = left_child
        left_child.right = node
        node.parent = left_child

    def delete(self, value):
        '''
        Удаляет узел с указанным значением из красно-черного дерева
        '''
        def transplant(u, v):
            if u.parent == self.NIL:
                self.root = v
            elif u == u.parent.left:
                u.parent.left = v
            else:
                u.parent.right = v
            v.parent = u.parent

        def minimum(node):
            if node.left == self.NIL:
                return node
            return minimum(node.left)

        # поиск узла для удаления p
        p = self.root
        while p.value != value:
            if p.value < value:
                p = p.right
            else:
                p = p.left

        y = p
        y_original_color = y.color
        if p.left == self.NIL:
            x = p.right
            transplant(p, x)
        elif p.right == self.NIL:
            x = p.left
            transplant(p, x)
        else:
            y = minimum(p.right)
            y_original_color = y.color
            x = y.right
            if y.parent == p:
                x.parent = y
            else:
                # следы от элемента, который встанет на место удаляемого, тоже нужно удалить
                transplant(y, x)
                y.right = p.right
                y.right.parent = y
            transplant(p, y)
            y.left = p.left
            y.left.parent = y
            y.color = p.color

        if y_original_color == BLACK:
            self._fix_delete(x)

    def _fix_delete(self, node):
        '''
        Корректирует дерево после вставки черного узла, чтобы поддерживать свойства красно-черного дерева
        '''
        while node != self.root and node.color == BLACK:
            if node == node.parent.left:
                s = node.parent.right
                if s.color == RED:
                    s.color = BLACK
                    node.parent.color = RED
                    self.__rotate_left(node.parent)
                    s = node.parent.right
                if s.left.color == BLACK and s.right.color == BLACK:
                    s.color = RED
                    node = node.parent
                else:
                    if s.right.color == BLACK:
                        s.left.color = BLACK
                        s.color = RED
                        self.__rotate_right(s)
                        s = node.parent.right
                    s.color = node.parent.color
                    node.parent.color = BLACK
                    s.right.color = BLACK
                    self.__rotate_left(node.parent)
                    node = self.root
            else:
                s = node.parent.left
                if s.color == RED:
                    s.color = BLACK
                    node.parent.color = RED
                    self.__rotate_right(node.parent)
                    s = node.parent.left
                if s.right.color == BLACK and s.left.color == BLACK:
                    s.color = RED
                    node = node.parent
                else:
                    if s.left.color == BLACK:
                        s.right.color = BLACK
                        s.color = RED
                        self.__rotate_left(s)
                        s = node.parent.left
                    s.color = node.parent.color
                    node.parent.color = BLACK
                    s.left.color = BLACK
                    self.__rotate_right(node.parent)
                    node = self.root
        node.color = BLACK

    def draw_img(self, img_name='Red_Black_Tree.png'):
        """
        Визуализация красно-черного дерева с помощью библиотеки pyGraphViz
        """
        if self.root is None:
            return

        tree = pgv.AGraph(directed=True, strict=True)

        queue = deque([self.root])
        num = 0
        while queue:
            e = queue.popleft()
            if e != self.NIL:
                tree.add_node(e.value, color=e.color, fontcolor="white", style="filled",
                              fontname="Microsoft YaHei", shape="circle", margin=0)
                for c in [e.left, e.right]:
                    queue.append(c)
                    if c != self.NIL:
                        tree.add_edge(e.value, c.value, color="b")
                    else:
                        num += 1
                        tree.add_node("nil%s" % num, label="Nil", color="darkgray", fontcolor="white", style="filled",
                                      fontname="Microsoft YaHei", shape="circle", margin=0)
                        tree.add_edge(e.value, "nil%s" % num, color="darkgray")

        tree.graph_attr['epsilon'] = '0.01'
        tree.layout('dot')
        tree.draw(img_name)
