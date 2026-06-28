import numpy as np
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class Stream:
    """Класс для представления потока в сети."""
    def __init__(self, name, flow_value, std_dev):
        self.name = name
        self.value = flow_value
        self.std_dev = std_dev
        self.variance = std_dev ** 2
        self.is_measured = True 

class Node:
    """Класс для представления узла (точки пересечения потоков)."""
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

class Network:
    """Класс для управления топологией сети и потоками."""
    def __init__(self):
        self.streams = {}
        self.nodes = {}
        self.stream_order = [] 
        self.node_order = []

    def add_node(self, name):
        """Добавление узла в сеть."""
        if name not in self.nodes:
            self.nodes[name] = Node(name)
            self.node_order.append(name)

    def add_stream(self, name, from_node, to_node, flow_value, std_dev):
        """Добавление потока между узлами."""
        stream = Stream(name, flow_value, std_dev)
        self.streams[name] = stream
        self.stream_order.append(name)
        
        if from_node:
            self.add_node(from_node)
            self.nodes[from_node].outputs.append(name)
        
        if to_node:
            self.add_node(to_node)
            self.nodes[to_node].inputs.append(name)

    def get_incidence_matrix(self):
        """Формирование матрицы инцидентности A."""
        n_nodes = len(self.node_order)
        n_streams = len(self.stream_order)
        A = np.zeros((n_nodes, n_streams))

        for i, node_name in enumerate(self.node_order):
            node = self.nodes[node_name]
            for stream_name in node.inputs:
                j = self.stream_order.index(stream_name)
                A[i, j] = 1.0 
            for stream_name in node.outputs:
                j = self.stream_order.index(stream_name)
                A[i, j] = -1.0 
        
        return A

    def get_measurements_vector(self):
        """Возвращает вектор измеренных значений потоков."""
        return np.array([self.streams[name].value for name in self.stream_order])

    def get_covariance_matrix(self):
        """Возвращает матрицу ковариации ошибок измерений."""
        variances = [self.streams[name].variance for name in self.stream_order]
        return np.diag(variances)
    
    def find_loops(self, candidate_streams_indices):
        """Проверка на наличие циклов (петель) среди подозреваемых потоков."""
        if not candidate_streams_indices:
            return False
            
        candidate_names = [self.stream_order[i] for i in candidate_streams_indices]
        edges = []
        involved_nodes = set()
        
        for name in candidate_names:
            u, v = None, None
            for n_name, node in self.nodes.items():
                if name in node.inputs:
                    v = n_name
                if name in node.outputs:
                    u = n_name
            
            if u is None: u = "ENV"
            if v is None: v = "ENV"
            
            edges.append((u, v))
            involved_nodes.add(u)
            involved_nodes.add(v)
            
        seen_pairs = set()
        for u, v in edges:
            pair = tuple(sorted((u, v)))
            if pair in seen_pairs:
                return True 
            seen_pairs.add(pair)
            
        adj = {node: [] for node in involved_nodes}
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        visited = set()
        def has_cycle(curr, parent):
            visited.add(curr)
            for neighbor in adj[curr]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    return True
                if has_cycle(neighbor, curr):
                    return True
            return False
            
        for node in involved_nodes:
            if node not in visited:
                if has_cycle(node, None):
                    return True
        return False

    def find_undirected_cycle(self, target_stream_name):
        """
        Ищет простой цикл, содержащий указанный поток.
        Возвращает набор имен потоков, образующих цикл (эквивалентный набор).
        """
        # 1. Определение начала и конца целевого потока
        u, v = None, None
        stream_idx = self.stream_order.index(target_stream_name)
        
        # Поиск связей во всех узлах
        for n_name, node in self.nodes.items():
            if target_stream_name in node.outputs: # u -> поток -> v
                u = n_name
            if target_stream_name in node.inputs:
                v = n_name
        
        if u is None: u = "ENV" # Окружающая среда (вход)
        if v is None: v = "ENV" # Окружающая среда (выход)
        
        if u == v: # Петля на один узел
            return {target_stream_name}

        # 2. Построение графа смежности (неориентированного) без учета целевого потока
        adj = {}
        # Убеждаемся, что все узлы и ENV присутствуют в списке смежности
        nodes = list(self.nodes.keys()) + ["ENV"]
        for n in nodes: adj[n] = []
        
        for s_name in self.stream_order:
            if s_name == target_stream_name:
                continue
                
            n1, n2 = None, None
            for n_name, node in self.nodes.items():
                if s_name in node.outputs: n1 = n_name
                if s_name in node.inputs: n2 = n_name
            
            if n1 is None: n1 = "ENV"
            if n2 is None: n2 = "ENV"
            
            # Неориентированные ребра
            adj[n1].append((n2, s_name))
            adj[n2].append((n1, s_name))

        # 3. BFS для поиска пути от v к u
        queue = [(v, [])] # Текущий узел, путь [(имя_потока), ...]
        visited = {v}
        
        while queue:
            curr, path = queue.pop(0)
            if curr == u:
                # Путь найден! Цикл = целевой поток + найденный путь
                return set([target_stream_name] + path)
            
            for neighbor, s_via in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [s_via]))
                    
        return {target_stream_name} # Цикл не найден


class SICCSolver:
    """Класс для решения задачи SICC (сведение баланса и поиск грубых ошибок)."""
    def __init__(self, network):
        self.network = network
        self.A = network.get_incidence_matrix()
        self.x_measured = network.get_measurements_vector()
        self.Q = network.get_covariance_matrix()
        self.Q_inv = np.linalg.inv(self.Q)
        
        try:
            self.V = self.A @ self.Q @ self.A.T 
            self.V_inv = np.linalg.inv(self.V)
            self.S = self.Q @ self.A.T @ self.V_inv @ self.A
        except np.linalg.LinAlgError:
            logger.error("Сингулярная матрица. Избыточная или некорректно поставленная задача.")
            raise

    def reconcile_base(self):
        """Базовое сведение баланса (Projected Measurements)."""
        I = np.eye(len(self.x_measured))
        x_tilde = (I - self.S) @ self.x_measured
        return x_tilde

    def perform_measurement_test(self, x_meas, x_rec, threshold=1.96):
        """Проведение теста невязок (Z-тест) для выявления подозрительных измерений."""
        r = x_meas - x_rec
        # Корректная ковариация: S * Q * S.T (Распространение ошибки)
        try:
            Cov_r = self.S @ self.Q @ self.S.T
        except:
             Cov_r = self.S @ self.Q # Запасной вариант при несовпадении размерностей
        
        flags = []
        z_scores = []
        
        for i in range(len(r)):
            variance = Cov_r[i, i]
            if variance < 1e-10:
                z = 0.0
            else:
                std_dev_r = np.sqrt(variance)
                z = r[i] / std_dev_r
            
            z_scores.append(z)
            if abs(z) > threshold:
                flags.append(i)
                
        return flags, z_scores

    def identify_equivalent_sets(self, lcge):
        """
        Идентификация эквивалентных наборов грубых ошибок.
        Для каждого подозреваемого проверяется, входит ли он в цикл.
        Все потоки в таком цикле являются равнозначными кандидатами.
        """
        equiv_sets = []
        for idx in lcge:
            stream_name = self.network.stream_order[idx]
            # Поиск цикла, содержащего этот поток
            cycle_streams = self.network.find_undirected_cycle(stream_name)
            equiv_sets.append(cycle_streams)
            
        return equiv_sets

    def solve_sicc(self):
        """Основной метод алгоритма SICC для идентификации грубых ошибок."""
        logger.info("Запуск алгоритма SICC...")
        
        x_tilde = self.reconcile_base()
        flags, z_scores = self.perform_measurement_test(self.x_measured, x_tilde)
        
        logger.info(f"Начальные флаги: {flags}")
        if not flags:
            return [], [], x_tilde, x_tilde, {}, "Грубых ошибок не обнаружено"
            
        lc = list(flags)
        lcge = [] 
        
        lc.sort(key=lambda idx: abs(z_scores[idx]), reverse=True)
        final_lc = []
        for cand in lc:
            if not self.network.find_loops(final_lc + [cand]):
                final_lc.append(cand)
        lc = final_lc
        
        max_iterations = 10
        iter_count = 0
        
        while lc and iter_count < max_iterations:
            iter_count += 1
            best_candidate = None
            min_obj_func = float('inf')
            
            for candidate in lc:
                current_set = lcge + [candidate]
                if self.network.find_loops(current_set):
                    continue
                
                n_meas = len(self.x_measured)
                L = np.zeros((n_meas, len(current_set)))
                for col_idx, s_idx in enumerate(current_set):
                    L[s_idx, col_idx] = 1.0
                    
                SL = self.S @ L
                
                try:
                    term_matrix = SL.T @ self.Q_inv @ SL
                    r_base = self.x_measured - x_tilde
                    rhs = SL.T @ self.Q_inv @ r_base
                    delta_star = np.linalg.solve(term_matrix, rhs)
                except np.linalg.LinAlgError:
                    continue
                
                x_trial = x_tilde + SL @ delta_star
                resid = x_trial - self.x_measured
                obj_func = resid.T @ self.Q_inv @ resid
                
                if obj_func < min_obj_func:
                    min_obj_func = obj_func
                    best_candidate = candidate
            
            if best_candidate is not None:
                logger.info(f"Итерация {iter_count}: Добавлен {best_candidate} (Obj: {min_obj_func:.4f})")
                lcge.append(best_candidate)
                
                current_set = lcge
                L = np.zeros((len(self.x_measured), len(current_set)))
                for col_idx, s_idx in enumerate(current_set):
                    L[s_idx, col_idx] = 1.0
                SL = self.S @ L
                
                try:
                    term_matrix = SL.T @ self.Q_inv @ SL
                    r_base = self.x_measured - x_tilde
                    rhs = SL.T @ self.Q_inv @ r_base
                    delta_stars = np.linalg.solve(term_matrix, rhs)
                except:
                    break

                x_rec = x_tilde + SL @ delta_stars
                flags_new, z_new = self.perform_measurement_test(self.x_measured, x_rec)
                
                lc = []
                flags_new.sort(key=lambda idx: abs(z_new[idx]), reverse=True)
                
                for f in flags_new:
                    if f not in lcge:
                        lc.append(f)
                
                filtered_lc = []
                for c in lc:
                    if not self.network.find_loops(lcge + [c]):
                        filtered_lc.append(c)
                lc = filtered_lc
            else:
                break
        
        # Шаг 6: Эквивалентные наборы
        equiv_sets = self.identify_equivalent_sets(lcge)

        if not lcge:
             return [], [], x_tilde, x_tilde, {}, "Грубых ошибок не подтверждено"
             
        L = np.zeros((len(self.x_measured), len(lcge)))
        for col_idx, s_idx in enumerate(lcge):
            L[s_idx, col_idx] = 1.0
        SL = self.S @ L
        term_matrix = SL.T @ self.Q_inv @ SL
        r_base = self.x_measured - x_tilde
        rhs = SL.T @ self.Q_inv @ r_base
        final_biases = np.linalg.solve(term_matrix, rhs)
        
        # Расчет итоговых нескорректированных потоков (x_hat)
        # x_hat = x_tilde + h * delta
        # h = (S - I)L
        I = np.eye(len(self.x_measured))
        h = (self.S - I) @ L
        x_hat = x_tilde + h @ final_biases
        
        bias_map = {}
        for idx, val in zip(lcge, final_biases):
            stream_name = self.network.stream_order[idx]
            bias_map[stream_name] = val
            
        return lcge, equiv_sets, x_tilde, x_hat, bias_map, "Обнаружены подозреваемые грубые ошибки"
