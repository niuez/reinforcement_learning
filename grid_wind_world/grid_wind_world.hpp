#include <vector>
#include <unordered_set>

struct State {
  int x;
  int y;

  bool operator==(const State& s) const {
    return x == s.x && y == s.y;
  }
};

namespace std {
  template<>
  struct hash<State> {
    std::size_t operator()(const State& state) const {
      return state.x * 10 + state.y;
    }
  };
}


struct GridWindWorld {
  int H;
  int W;
  std::vector<std::vector<std::pair<int, int>>> wind;
  int sx;
  int sy;
  int tx;
  int ty;

  void init_example() {
    H = 7;
    W = 10;
    wind.resize(H, std::vector<std::pair<int, int>>(W));

    std::vector<int> wy = { 0, 0 ,0, 1, 1, 1, 2, 2, 1, 0 };
    for(int i = 0; i < H; i++) {
      for(int j = 0; j < W; j++) {
        wind[i][j] = { -wy[j], 0 };
      }
    }
    sx = 3;
    sy = 0;
    tx = 3;
    ty = 7;
  }

  State init_state() const {
    return State { sx, sy };
  }
  bool is_goal(const State& state) const {
    return state.x == tx && state.y == ty;
  }
  void move_state(State& state, int dx, int dy) const {
    state.x = std::clamp(state.x + dx, 0, H - 1);
    state.y = std::clamp(state.y + dy, 0, W - 1);
  }
  double action(State& state, int action) const {
    const static std::vector<int> dd = { 0, 1, 0, -1 };
    auto [wx, wy] = wind[state.x][state.y];
    move_state(state, wx + dd[action], wy + dd[action ^ 1]);
    return -1;
  }
};

