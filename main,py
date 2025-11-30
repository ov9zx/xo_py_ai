import math
from copy import deepcopy
import time

BOARD_SIZE = 4
WIN_LENGTH = 3
HUMAN = "X"  # Maximizer
AI = "O"  # Minimizer
EMPTY = "."  # Пустая ячейка


# ============================================================
#               БАЗОВЫЕ ИГРОВЫЕ ФУНКЦИИ
# ============================================================

def create_board():
    """Создание пустого поля 4x4."""
    return [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def print_board(board):
    """Вывод в консоль игрового поля."""
    print("\n  " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for i, row in enumerate(board):
        print(i, " ".join(row))
    print()


def is_win(board, player):
    """Проверка победы: 3 одинаковых символа подряд."""
    # Проверка горизонталей, вертикалей, основных и обратных диагоналей

    # 1. Горизонтали
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - WIN_LENGTH + 1):
            if all(board[r][c + i] == player for i in range(WIN_LENGTH)):
                return True

    # 2. Вертикали
    for c in range(BOARD_SIZE):
        for r in range(BOARD_SIZE - WIN_LENGTH + 1):
            if all(board[r + i][c] == player for i in range(WIN_LENGTH)):
                return True

    # 3. Основные диагонали (\)
    for r in range(BOARD_SIZE - WIN_LENGTH + 1):
        for c in range(BOARD_SIZE - WIN_LENGTH + 1):
            if all(board[r + i][c + i] == player for i in range(WIN_LENGTH)):
                return True

    # 4. Обратные диагонали (/)
    for r in range(BOARD_SIZE - WIN_LENGTH + 1):
        for c in range(WIN_LENGTH - 1, BOARD_SIZE):
            if all(board[r + i][c - i] == player for i in range(WIN_LENGTH)):
                return True

    return False


def is_draw(board):
    """Проверка ничьей."""
    return all(board[r][c] != EMPTY for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))


def get_valid_moves(board):
    """Генерация всех возможных ходов."""
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == EMPTY:
                moves.append((r, c))
    return moves


# ============================================================
#               ЭВРИСТИЧЕСКАЯ ФУНКЦИЯ
# ============================================================

def evaluate_segment(segment):
    """Оценивает один отрезок из 3 клеток согласно требованиям задания."""
    # +10/-10 за двойки, +1/-1 за возможности

    count_x = segment.count(HUMAN)
    count_o = segment.count(AI)
    score = 0

    # Оценка для X (Maximizer)
    if count_o == 0:  # Линия потенциально выигрышная для X
        if count_x == 2:
            score += 10  # +10 за каждую "двойку" X
        elif count_x == 1:
            score += 1  # +1 за каждую возможную линию с одним X

    # Оценка для O (Minimizer)
    if count_x == 0:  # Линия потенциально выигрышная для O
        if count_o == 2:
            score -= 10  # -10 за каждую "двойку" O
        elif count_o == 1:
            score -= 1  # -1 за аналогичную линию для O

    return score


def heuristic(board):
    """Основная эвристическая функция, суммирует оценки по всем 24 отрезкам."""
    total_score = 0

    # 1. Горизонтали
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE - WIN_LENGTH + 1):
            segment = [board[r][c + i] for i in range(WIN_LENGTH)]
            total_score += evaluate_segment(segment)

    # 2. Вертикали
    for c in range(BOARD_SIZE):
        for r in range(BOARD_SIZE - WIN_LENGTH + 1):
            segment = [board[r + i][c] for i in range(WIN_LENGTH)]
            total_score += evaluate_segment(segment)

    # 3. Основные диагонали (\)
    for r in range(BOARD_SIZE - WIN_LENGTH + 1):
        for c in range(BOARD_SIZE - WIN_LENGTH + 1):
            segment = [board[r + i][c + i] for i in range(WIN_LENGTH)]
            total_score += evaluate_segment(segment)

    # 4. Обратные диагонали (/)
    for r in range(BOARD_SIZE - WIN_LENGTH + 1):
        for c in range(WIN_LENGTH - 1, BOARD_SIZE):
            segment = [board[r + i][c - i] for i in range(WIN_LENGTH)]
            total_score += evaluate_segment(segment)

    return total_score


# ============================================================
#               АЛГОРИТМ МИНИМАКС
# ============================================================

def minimax(board, depth, is_max_turn, max_depth):
    """
    Рекурсивный минимаксный поиск с ограничением глубины.
    Возвращает: (оценка, (row, col))
    """

    # --- 1. Проверка терминальных состояний ---
    # Добавляем depth к оценке, чтобы ИИ выбирал более быструю победу/защиту.
    if is_win(board, HUMAN):
        return 1000 - depth, None
    if is_win(board, AI):
        return -1000 + depth, None
    if is_draw(board):
        return 0, None

    # --- 2. Ограничение глубины ---
    if depth == max_depth:
        return heuristic(board), None

    valid_moves = get_valid_moves(board)

    # Если нет ходов (по идее, уже покрыто is_draw)
    if not valid_moves:
        return 0, None

    # --- 3. Ход Максимизатора (X) ---
    if is_max_turn:
        best_value = -math.inf
        best_move = None

        for move in valid_moves:
            r, c = move
            new_board = deepcopy(board)
            new_board[r][c] = HUMAN

            value, _ = minimax(new_board, depth + 1, False, max_depth)  # Рекурсия для минимизатора

            if value > best_value:
                best_value = value
                best_move = move

        return best_value, best_move

    # --- 4. Ход Минимизатора (O) ---
    else:  # is_max_turn == False (Ход ИИ/Минимизатора)
        best_value = math.inf
        best_move = None

        for move in valid_moves:
            r, c = move
            new_board = deepcopy(board)
            new_board[r][c] = AI

            value, _ = minimax(new_board, depth + 1, True, max_depth)  # Рекурсия для максимизатора

            if value < best_value:
                best_value = value
                best_move = move

        return best_value, best_move


# ============================================================
#               ИГРОВОЙ ЦИКЛ
# ============================================================

def ai_move(board, max_depth):
    """
    Делает ход ИИ, используя минимакс.
    """
    start_time = time.time()
    print("\nИИ думает...")

    # ИИ (O) - Минимизатор, поэтому начинаем с is_max_turn=False
    value, move = minimax(board, depth=0, is_max_turn=False, max_depth=max_depth)

    end_time = time.time()

    print(f"• Оценка позиции для ИИ: {value}")
    print(f"• ИИ выбирает ход: {move}")
    print(f"• Глубина поиска: {max_depth}")
    print(f"• Время на ход: {end_time - start_time:.4f} сек.")

    if move is not None:
        r, c = move
        board[r][c] = AI
    return board


def human_move(board):
    """Ход человека."""
    while True:
        try:
            r = int(input("Введите номер строки (0-3): "))
            c = int(input("Введите номер столбца (0-3): "))
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r][c] == EMPTY:
                board[r][c] = HUMAN
                return board
            else:
                print("Некорректный ход. Попробуйте снова.")
        except ValueError:
            print("Ошибка ввода. Введите целое число.")


def play_game(max_depth=2):
    """Основной игровой цикл."""
    board = create_board()
    print("Игра начинается! Вы играете за X.")
    print_board(board)

    while True:
        # --- Ход человека (X) ---
        board = human_move(board)
        print_board(board)

        if is_win(board, HUMAN):
            print("Вы победили! 🥳")
            break
        if is_draw(board):
            print("Ничья! 🤝")
            break

        # --- Ход ИИ (O) ---
        board = ai_move(board, max_depth=max_depth)
        print_board(board)

        if is_win(board, AI):
            print("ИИ победил! 🤖")
            break
        if is_draw(board):
            print("Ничья! 🤝")
            break


if __name__ == "__main__":
    # Для выполнения задания 4.a, 4.b, 4.c меняйте max_depth
    # Рекомендуется начать с depth=2
    TEST_DEPTH = 2
    print(f"*** ТЕСТИРОВАНИЕ С ГЛУБИНОЙ {TEST_DEPTH} ***")
    # play_game(max_depth=1) # Для задания 4.a
    # play_game(max_depth=3) # Для задания 4.c
    play_game(max_depth=TEST_DEPTH)  # Для задания 4.b
