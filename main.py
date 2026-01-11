import random
import time
import numpy as np
from collections import deque
from collections import defaultdict

class Cards:
    KING = 10
    QUEEN = 10
    JACK = 10

DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, Cards.KING, Cards.QUEEN, Cards.JACK] * 4

class Person:
    def __init__(self):
        self.hand = []
        self.total = 0
        self.aces = 0
        self.stand = False

    def check_total(self):
        total = 0
        aces = 0
        for card in self.hand:
            if card == 1:
                aces += 1
                total += 11
            else:
                total += card

        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        self.total = total
        self.aces = aces

class Player(Person):
    def __init__(self, exp_rate=0.2, learn_rate=0.1):
        super().__init__()
        self.states = []
        self.actions = [True, False]
        self.exploration_rate = exp_rate
        self.lr = learn_rate

        # State is a tuple: (player_total, dealer_upcard, usable_ace_or_aces_count)
        # action is whether stand is true or false
        # Key: (state, action) -> Value: dict of metrics
        self.q_values = defaultdict(lambda: {"count": 0, "success": 0.0})

    def action(self, d_total):

        self.check_total()
        state = (self.total, d_total, self.aces)

        # default actions
        # if total <= 11 drawing card is always optimal
        # if total > 21 prevent simulation from continue trying to draw cards
        if self.total <= 11:
            self.states.append((state, False))
            return
        elif self.total > 21:
            self.stand = True
            return

        # exploration - greedy policy to explore sample space
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(self.actions)
            self.stand = action
            self.states.append((state, action))

        # greedy for optimal policy
        else:
            self.stand = max(self.actions, key = lambda x: self.q_values[(state, x)]["success"])
            self.states.append((state, self.stand))

        return

    # go back through the seen states and update mean depending on the game outcome
    def reward(self, loss):
        if loss is None:
            update = 0
        elif loss:
            update = -1
        else:
            update = 1

        for visits in self.states:
            self.q_values[visits]["count"] += 1
            q = self.q_values[visits]["success"]
            self.q_values[visits]["success"] = q + self.lr * (update - q)

    def reset(self):
        self.hand = []
        self.total = 0
        self.aces = 0
        self.stand = False
        self.states = []


class Dealer(Person):
    def __init__(self):
        super().__init__()

    def action(self):
        self.check_total()
        if self.total >= 17:
            self.stand = True


class Game:
    def __init__(self, player = Player()):
        self.player = player
        self.dealer = Dealer()
        self.playing = True
        self.dealer_win = None
        self.deck = deque()
        self.deck_check()
        self.stats = {"Player Wins": 0, "Dealer Wins": 0, "Draws": 0}

    def deal(self, user):
        self.deck_check()

        card = self.deck.popleft()
        user.hand.append(card)


    def deck_check(self):
        if len(self.deck) < len(DECK):
            new_deck = DECK.copy() * 4
            random.shuffle(new_deck)
            self.deck = deque(new_deck)

    def start_deal(self):
        self.deal(self.player)
        self.deal(self.player)
        self.deal(self.dealer)
        self.deal(self.dealer)
        self.player.check_total()
        self.dealer.check_total()

    def score_check(self):
        if self.player.total > 21:
            self.dealer_win, self.playing = True, False

        elif self.dealer.total > 21:
            self.dealer_win, self.playing = False, False

        elif self.player.stand and self.dealer.stand:
            if self.player.total > self.dealer.total:
                self.dealer_win = False
            elif self.player.total == self.dealer.total:
                self.dealer_win = None
            else:
                self.dealer_win = True

            self.playing = False

    def play(self):
        self.player.reset()
        self.start_deal()
        self.player.action(self.dealer.hand[0])

        while not self.player.stand:
            self.deal(self.player)
            self.player.action(self.dealer.hand[0])
        self.score_check()

        if not self.playing:
            self.player.reward(self.dealer_win)
            return

        self.dealer.action()
        while not self.dealer.stand:
            self.deal(self.dealer)
            self.dealer.action()

        self.score_check()
        self.player.reward(self.dealer_win)

    def reset(self):
        if self.dealer_win is None:
            self.stats['Draws'] += 1
        elif self.dealer_win:
            self.stats['Dealer Wins'] += 1
        else:
            self.stats['Player Wins'] += 1

        self.dealer = Dealer()
        self.playing = True
        self.dealer_win = None
        self.deck = deque()
        self.deck_check()


if __name__ == '__main__':

    n = 1000000

    player = Player()
    game = Game(player)

    for i in range(n):
        if i % 100000 == 0:
            print(f'Game #{i}, Games Remaining {n - i}')
        game.play()
        game.reset()

    print(game.stats)
    print(player.q_values)
