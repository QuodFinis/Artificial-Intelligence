import time
import math
import random

class Player:
    def __init__(self, name):
        self.name = name
        self.letter = ""
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.win_rate_history = []
        self.variance_of_performance = 0
        self.game_time = 0
        self.avg_time = 0
        self.rating = 0

    def move(self, game):
        start_time = time.time()
        move = self.get_move(game)
        end_time = time.time()

        move_time = end_time - start_time

        self.game_time += move_time
        total_games = self.wins + self.losses + self.ties
        self.avg_time = self.game_time / total_games

        return move

    def get_move(self, game):
        pass

    def stats(self):
        return f"\n{self.name} stats: {self.wins} Wins, {self.ties} Ties, with avg time of {self.avg_time:.2f} " \
               f"seconds, and rating of {self.rating}"

    def update_stats(self, winner):
        if winner == self.letter:
            self.wins += 1
        elif winner is None:
            self.ties += 1
        else:
            self.losses += 1

        total_games = self.wins + self.losses + self.ties
        win_rate = self.wins / total_games

        self.win_rate_history.append(win_rate)

        if len(self.win_rate_history) < 2:
            variance_of_performance = 0
        else:
            mean = sum(self.win_rate_history) / len(self.win_rate_history)
            variance_of_performance = math.sqrt(
                sum((x - mean) ** 2 for x in self.win_rate_history) / len(self.win_rate_history))

        if len(self.win_rate_history) >= 10:
            last_10_games = self.win_rate_history[-10:]
            last_10_win_rate = sum(last_10_games) / 10
            trend_of_performance = last_10_win_rate - win_rate
        else:
            trend_of_performance = 0

        avg_time = self.game_time / total_games

        win_rate_weight = 0.6
        variance_of_performance_weight = 0.2
        trend_of_performance_weight = 0.1
        avg_time_weight = 0.1

        win_rate_normalized = win_rate

        variance_of_performance_normalized = 1 - variance_of_performance

        trend_of_performance_normalized = (trend_of_performance + 1) / 2

        avg_time_normalized = max(0, 1 - avg_time / 5)

        rating = (win_rate_weight * win_rate_normalized +
                  variance_of_performance_weight * variance_of_performance_normalized +
                  trend_of_performance_weight * trend_of_performance_normalized +
                  avg_time_weight * avg_time_normalized) * 10

        self.variance_of_performance = variance_of_performance
        self.rating = rating
        self.game_time = 0


class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def get_move(self, game):
        valid_square = False
        val = None

        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-8): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError

                valid_square = True

            except ValueError:
                print('Invalid square. Try again.')

        return val


class ComputerPlayer(Player):
    def __init__(self, name, initial_difficulty=0):
        super().__init__(name)
        self.difficulty = initial_difficulty

    def update_difficulty(self, opp_player):
        self.difficulty = opp_player.rating

    def get_move(self, game):
        if self.difficulty == 0:
            square = random.choice(game.available_moves())
        else:
            square = self.minimax_abp_depth_rand(game, -math.inf, math.inf, True)['position']
        return square

    def minimax_abp_depth_rand(self, state, alpha, beta, maximizing_player):
        max_player = self.letter
        other_player = 'O' if max_player == 'X' else 'X'

        if state.winner_letter == other_player:
            return {'position': None, 'score': 1 * (state.num_empty_squares() + 1)}

        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if maximizing_player:
            best = {'position': None, 'score': -math.inf}
        else:
            best = {'position': None, 'score': math.inf}

        moves = state.available_moves()
        best_moves = []  # list of moves with the same score as the best move

        for possible_move in moves:
            state.make_move(possible_move, max_player)
            sim_score = self.minimax_abp_depth_rand(state, alpha, beta, not maximizing_player)
            state.board[possible_move] = ' '
            state.winner_letter = None

            move_score = {'position': possible_move, 'score': sim_score['score']}
            if maximizing_player:
                if move_score['score'] > best['score']:  # new best move found
                    best = move_score
                    best_moves = [move_score['position']]  # reset the list of best moves
                elif move_score['score'] == best['score']:  # another move with the same score as the best move
                    best_moves.append(move_score['position'])  # add it to the list of best moves
                alpha = max(alpha, sim_score['score'])
            else:
                if move_score['score'] < best['score']:  # new best move found
                    best = move_score
                    best_moves = [move_score['position']]  # reset the list of best moves
                elif move_score['score'] == best['score']:  # another move with the same score as the best move
                    best_moves.append(move_score['position'])  # add it to the list of best moves
                beta = min(beta, sim_score['score'])

            if alpha >= beta:
                break

        best['position'] = random.choice(best_moves)  # randomly choose one of the best moves
        return best


class TicTacToe:
    def __init__(self, playerA, playerB):
        self.board = [' ' for _ in range(9)]
        self.winner_letter = None
        self.playerA = playerA
        self.playerB = playerB
        self.play()

    @staticmethod
    def print_board_nums():
        number_board = [[str(i) for i in range(j * 3, (j + 1) * 3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')
        print("\n")

    def print_board(self):
        for row in [self.board[i * 3:(i + 1) * 3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    def play(self):
        while True:
            if random.choice([True, False]):
                self.playerA.letter = 'X'
                self.playerB.letter = 'O'
            else:
                self.playerA.letter = 'O'
                self.playerB.letter = 'X'

            if isinstance(self.playerA, ComputerPlayer):
                self.playerA.update_difficulty(self.playerB)
            if isinstance(self.playerB, ComputerPlayer):
                self.playerB.update_difficulty(self.playerA)

            self.print_board_nums()

            letter = 'X'
            while self.empty_squares():
                current_player = self.playerA if self.playerA.letter == 'X' else self.playerB

                square = current_player.get_move(self)  # get the move from the current player

                if self.make_move(square, letter):
                    print(f'{current_player.name} ({letter}) makes a move to square {square}')  # print the name and the letter of the current player
                    self.print_board()
                    print('')

                    if self.winner_letter:
                        winner = self.playerA.name if self.winner_letter == self.playerA.letter else self.playerB.name
                        print(f"{winner} won!!!")
                        break

                    letter = 'O' if letter == 'X' else 'X'

            if not self.winner_letter:
                print('It\'s a tie!')

            self.playerA.update_stats(self.winner_letter)
            self.playerB.update_stats(self.winner_letter)

            play_again = input('Do you want to play again? (Y/N) ')
            if play_again.lower() != 'y':
                print(f'After {self.playerA.wins + self.playerA.losses + self.playerA.ties} game(s), \
                            {self.playerA.stats()}, {self.playerB.stats()}, \n')
                if self.playerA.wins > self.playerB.wins:
                    print(f'{self.playerA.name} wins!')
                elif self.playerA.wins < self.playerB.wins:
                    print(f'{self.playerB.name} wins!')
                else:
                    print('It\'s a tie!')
                break

            self.board = [' ' for _ in range(9)]
            self.winner_letter = None

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.winner_letter = letter
            return True

            # Move the winner check outside the if block
            self.winner_letter = letter if self.winner(square, letter) else None
            return True

        return False

    def winner(self, square, letter):
        row_ind = square // 3
        row = self.board[row_ind * 3: (row_ind + 1) * 3]
        if all([spot == letter for spot in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind + i * 3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False


if __name__ == '__main__':
    p1 = ComputerPlayer('b1', initial_difficulty=1)
    p2 = ComputerPlayer('b2', initial_difficulty=10)
    ttt = TicTacToe(p1, p2)
