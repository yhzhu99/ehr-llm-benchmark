class Node(object):
    def __init__(self, value='root'):
        self.value = value
        self.children = []


def get_path_length(root, path, k):
    # base case handling
    if root is None:
        return False
    path.append(root.value)
    if root.value == k:
        return True
    for child in root.children:
        if child.children is not None and get_path_length(child, path, k):
            return True
    # 如果当前结点的值并不是k
    path.pop()
    return False


def find_distance(root, n1, n2):
    if root:
        # 获取第一个结点的路径（存储跟结点到i）
        path1 = []
        get_path_length(root, path1, n1)
        # 获取第二个结点的路径
        path2 = []
        get_path_length(root, path2, n2)
        # 找到它们的公共祖先
        i = 0
        while i < len(path1) and i < len(path2):
            if path1[i] != path2[i]:
                break
            i = i + 1
        # 减去重复计算的跟结点到lca部分即为结果
        return len(path1) + len(path2) - 2 * i
    else:
        return 0


def is_children(root, k):
    for node in root.children:
        if node.value == k:
            return node
    return None


def build_tree_fun(file_path):
    root = Node(value='root')

    with open(file_path, 'r') as f:
        content = f.readline()
        while content:
            contents = content.split(sep=' ', maxsplit=1)
            code = contents[0].strip()

            # 一级节点
            first_node = is_children(root, code[0])
            if first_node is None:
                first_node = Node(value=code[0])
                root.children.append(first_node)

            # 一级节点
            second_node = is_children(first_node, code[:3])
            if second_node is None:
                second_node = Node(value=code[:3])
                first_node.children.append(second_node)

            parent_node = second_node
            for index in range(3, len(code)):
                children_node = is_children(parent_node, code[:index + 1])
                if children_node is None:
                    children_node = Node(value=code[:index + 1])
                    parent_node.children.append(children_node)
                parent_node = children_node

            content = f.readline()
    return root