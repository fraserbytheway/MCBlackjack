import random
from collections import deque, defaultdict
import multiprocessing as mp

class Cards:
    KING = 10
    QUEEN = 10
    JACK = 10

DECK = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, Cards.KING, Cards.QUEEN, Cards.JACK] * 4

def default_q():
    return {"count": 0, "success": 0.0}

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
        self.q_values = self.q_values = defaultdict(default_q)

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

        # epsilon soft policy MC
        # random state exploration
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.choice(self.actions)
            self.stand = action
            self.states.append((state, action))

        # greedy action
        else:
            self.stand = max(self.actions, key = lambda x: self.q_values[(state, x)]["success"])
            self.states.append((state, self.stand))

        return

    # updating action value function
    def reward(self, loss):
        if loss is None:
            update = 0
        elif loss:
            update = -1
        else:
            update = 1

        for visits in set(self.states):
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
    def __init__(self, player = None, num_decks = 6):
        self.player = player if player is not None else Player()
        self.dealer = Dealer()
        self.playing = True
        self.dealer_win = None
        self.num_decks = num_decks
        self.deck = deque()
        self.deck_check()
        self.stats = {"Player Wins": 0, "Dealer Wins": 0, "Draws": 0}

    def deal(self, user):
        self.deck_check()

        card = self.deck.popleft()
        user.hand.append(card)


    def deck_check(self):
        if len(self.deck) < len(DECK):
            new_deck = DECK.copy() * self.num_decks
            random.shuffle(new_deck)
            self.deck = deque(new_deck)

    def start_deal(self):
        for per in [self.player, self.dealer]:
            self.deal(per)
            self.deal(per)
            per.check_total()

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

def run_sim(args):
    n_trials, exp_rate, learn_rate = args
    sim = Simulation(exp_rate, learn_rate)
    sim.trials(n_trials)
    return sim.player.q_values


def merge_q_values(q_dicts):
    merged = defaultdict(lambda: {"count": 0, "success": 0.0})

    for q in q_dicts:
        for (state_action), data in q.items():
            total_count = merged[state_action]["count"] + data["count"]

            if total_count == 0:
                continue

            merged[state_action]["success"] = (
                merged[state_action]["success"] * merged[state_action]["count"]
                + data["success"] * data["count"]
            ) / total_count

            merged[state_action]["count"] = total_count

    return merged

class Simulation:
    def __init__(self, exp_rate = 0.2, learn_rate = 0.1):
        self.player = Player(exp_rate = exp_rate, learn_rate = learn_rate)
        self.game = Game(player=self.player)

    def trials(self, n = 10000):
        for i in range(n):
            self.game.play()
            self.game.reset()
        return self.player.q_values

    def results(self):
        return self.player.q_values, self.game.stats

def mp_sim(num_worker = 8, trials = 1000000, exp_rate = 0.2, learn_rate = 0.1):

    trials_per_worker = trials // num_worker

    with mp.Pool(num_worker) as p:
        q_values_list = p.map(
            run_sim,
            [(trials_per_worker, exp_rate, learn_rate)] * num_worker
        )

    return merge_q_values(q_values_list)


if __name__ == '__main__':

    res = mp_sim()
    print(res)

