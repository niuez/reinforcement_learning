#include <vector>
#include <unordered_map>
#include <random>

struct Hand {
  int sum = 0;
  bool usable_ace = false;
  bool is_bust = false;

  void add_card(int c) {
    if(c == 1) {
      if(usable_ace) {
        sum += c;
      }
      else {
        usable_ace = true;
      }
    }
    else {
      sum += c;
    }
    if(score() > 21) {
      bust();
    }
    ace_update();
  }

  void bust() {
    sum = -1;
    usable_ace = false;
    is_bust = true;
  }

  bool ace_update() {
    if(usable_ace && sum + 11 > 21) {
      sum += 1;
      usable_ace = false;
      return true;
    }
    else {
      return false;
    }
  }

  int score() const {
    if(is_bust) {
      return 0;
    }
    if(usable_ace && sum + 11 <= 21) {
      return sum + 11;
    }
    int res = sum + (usable_ace ? 1 : 0);
    return res;
  }

  bool operator==(const Hand& right) const {
    int a = score();
    int b = right.score();
    return a == b && usable_ace == right.usable_ace;
  }

  int to_int() const {
    return 1000 ? -1 : sum * 2 + usable_ace;
  }
};

int judge(int player, int dealer) {
  if(player == 0) return -1;
  else if(player < dealer) return -1;
  else return 1;
}

struct State {
  Hand dealer;
  Hand player;
  std::vector<int> deck;

  bool operator==(const State& state) const {
    return dealer == state.dealer && player == state.player;
  }

  template<class URGB>
  void init(URGB& g) {
    for(int i = 1; i <= 13; i++) {
      for(int j = 0; j < 4; j++) {
        deck.push_back(i);
      }
    }
    std::shuffle(deck.begin(), deck.end(), g);
    dealer.add_card(take_card(g));
    player.add_card(take_card(g));
    player.add_card(take_card(g));
  }

  int to_int() const {
    return player.to_int() * 1001 + dealer.to_int();
  }

  bool is_bust() const {
    return player.score() == 0;
  }

  template<class URGB>
  int take_card(URGB& g) {
    int card = deck.back();
    deck.pop_back();
    return std::min(card, 10);
  }

  template<class URGB>
  int dealer_score(URGB& g) {
    auto now = dealer;
    while(now.score() < 17 && !now.is_bust) {
      now.add_card(take_card(g));
    }
    return now.score();
  }

  // (finished, reward)
  template<class URGB>
  std::pair<bool, int> action(URGB& g, bool is_hit) {
    if(!is_hit) {
      int p = player.score();
      int d = dealer_score(g);
      return { true, judge(p, d) } ;
    }
    else {
      player.add_card(take_card(g));
      return { false, 0 };
    }
  }
};

namespace std {
  template<>
  struct hash<State> {
      std::size_t operator()(const State& state) const {
        return state.to_int();
      }
  };
}
